import argparse
import datetime
import math
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import Dataset as HFDataset, load_dataset, load_from_disk
from torch.utils.data import DataLoader, Dataset

from discriminator import (
    DEFAULT_ENCODER_MODEL,
    MAX_ENCODER_SEQ_LEN,
    format_discriminator_input,
    get_encoder_discriminator,
    get_retrieval_ctx,
)
from generator import (
    MODEL_ID,
    GEN_MODE,
    OPENAI_MODEL,
    FakeNewsGenerator,
    search_wikipedia,
)


class TextDataset(Dataset):
    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


def _normalize_date(raw_date: Any) -> Any:
    if hasattr(raw_date, "date"):
        return raw_date
    if isinstance(raw_date, str):
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.datetime.strptime(raw_date, fmt)
            except ValueError:
                continue
        try:
            return datetime.datetime.fromisoformat(raw_date)
        except ValueError:
            return raw_date
    return raw_date if raw_date is not None else datetime.datetime.utcnow()


def extract_article(text: str) -> Tuple[str, str]:
    """
    Extract generated Title/Body from the generator output, removing system/user prompt.
    """
    if not isinstance(text, str):
        return "", ""
    title, body = "", text
    if "Title:" in text:
        parts = text.split("Title:", 1)[1]
        if "Body:" in parts:
            title_part, body_part = parts.split("Body:", 1)
            title = title_part.strip().splitlines()[0] if title_part.strip() else ""
            body = body_part.strip()
        else:
            body = parts.strip()
    return title, body


def _slice_length_from_split(split: str) -> Optional[int]:
    """
    Parse dataset split expressions such as "train[:200]" to infer how many
    examples are described by the slice. Returns None if the split cannot be
    interpreted.
    """
    match = re.search(r"\[\s*([0-9]*)?\s*:\s*([0-9]*)?\s*\]", split or "")
    if not match:
        return None
    start_str, end_str = match.group(1), match.group(2)
    start = int(start_str) if start_str else 0
    if end_str:
        end = int(end_str)
        if end <= start:
            return None
        return end - start
    return None


class GANTrainer:
    """
    Tie the RAG-enabled generator and discriminator into a simple GAN-style loop.
    The generator crafts fake news with retrieval context, while the discriminator
    learns to separate real and generated samples and provides feedback.

    Enhanced with:
    - Dynamic balancing: Skip D training when G is too weak
    - Few-shot learning: Include successful examples in feedback
    - Label smoothing: Soften D's learning signal
    """

    def __init__(self, args: argparse.Namespace, dataset: List[Dict[str, Any]]):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.generator = FakeNewsGenerator(model_id=args.generator_model)
        self.discriminator = get_encoder_discriminator(
            model_name=args.discriminator_model
        )
        self.discriminator.model.train()
        self.optimizer = torch.optim.AdamW(
            self.discriminator.model.parameters(), lr=args.lr
        )
        self.gen_sft_enabled = (
            getattr(args, "gen_sft_every_round", False)
            and GEN_MODE == "local"
        )
        self.gen_use_lora = getattr(
            getattr(self.generator, "engine", None), "use_lora", False
        )
        self.gen_model = None
        self.gen_tokenizer = None
        self.gen_base_model = None
        self.gen_optimizer = None
        self.gen_lambda_kl = getattr(args, "gen_sft_lambda_kl", 0.01)
        self.gen_max_grad_norm = getattr(args, "gen_sft_max_grad_norm", 1.0)
        self.gen_sft_steps = getattr(args, "gen_sft_steps", 2)
        self.gen_sft_batch_size = getattr(args, "gen_sft_batch_size", 1)
        self.gen_sft_max_length = getattr(args, "gen_sft_max_length", 512)
        self.gen_success_threshold = getattr(args, "gen_sft_success_threshold", 0.55)
        self.gen_sft_max_samples = getattr(args, "gen_sft_max_samples", 2)
        self.gen_sft_warmup_rounds = getattr(args, "gen_sft_warmup_rounds", 3)
        if self.gen_sft_enabled and self.gen_use_lora:
            (
                self.gen_model,
                self.gen_tokenizer,
                self.gen_base_model,
            ) = self.generator.get_trainable_components()
            if self.gen_base_model is not None:
                self.gen_base_model.to(next(self.gen_model.parameters()).device)
            params = [p for p in self.gen_model.parameters() if p.requires_grad]
            self.gen_optimizer = torch.optim.AdamW(params, lr=args.gen_sft_lr)
        self.best_loss = float("inf")
        self.best_path = None
        if not dataset:
            raise ValueError(
                "Training dataset is empty; at least one example is required."
            )
        self.dataset = dataset
        self.dataset_len = len(dataset)
        self.real_samples_per_round = self._determine_samples_per_round()
        self.max_unique_rounds = math.ceil(
            self.dataset_len / self.real_samples_per_round
        )
        self._wrapped_warned = False

        # === 新增：動態平衡與 Few-shot 學習 ===
        self.successful_examples = []  # 存儲騙過 D 的成功案例 (最多保留 5 個)
        self.last_fool_rate = 0.0  # 追蹤上一輪的 fool rate
        self.rounds_without_training = 0  # 連續幾輪沒訓練 D

        # 動態平衡參數（可透過 args 調整）
        self.min_fool_rate_to_train = getattr(
            args, "min_fool_rate_to_train", 0.05
        )  # 低於此值暫停訓練 D
        self.max_skip_rounds = getattr(args, "max_skip_rounds", 2)  # 最多連續跳過幾輪
        self.label_smoothing = getattr(
            args, "label_smoothing", 0.1
        )  # Label smoothing 係數

    def _build_context(self, example: Dict[str, Any]) -> str:
        # Respect generator_use_wiki flag: only fetch wiki context when enabled
        if self.args.rag_source == "none":
            return ""
        if self.args.rag_source == "wiki":
            if not self.args.generator_use_wiki:
                return ""
            query = example.get("title") or str(example.get("description", ""))[:50]
            return search_wikipedia(
                query, num_results=self.args.num_rag_results, lang=self.args.rag_lang
            )
        if self.args.rag_source == "google":
            return get_retrieval_ctx(example, prefix="", source="google")
        return ""

    def _collate(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts = [item["text"] for item in batch]
        labels = torch.tensor(
            [item["label"] for item in batch],
            dtype=torch.long,
            device=self.discriminator.device,
        )
        encoded = self.discriminator.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.args.max_length,
            return_tensors="pt",
        ).to(self.discriminator.device)
        encoded["labels"] = labels
        return encoded

    def _train_discriminator(
        self, samples: List[Dict[str, Any]], use_label_smoothing: bool = True
    ) -> float:
        """
        Train discriminator with optional label smoothing to slow down learning.
        Label smoothing: soft labels like 0.9/0.1 instead of 1.0/0.0
        """
        dataloader = DataLoader(
            TextDataset(samples),
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=self._collate,
        )
        epoch_loss = 0.0
        step = 0

        for batch in dataloader:
            outputs = self.discriminator.model(**batch)

            # Label smoothing: 讓 D 學得慢一點
            if use_label_smoothing and self.label_smoothing > 0:
                # 原本 loss 是用 hard labels，這裡用 soft labels 重新計算
                logits = outputs.logits
                labels = batch["labels"]
                num_classes = logits.shape[-1]

                # Soft labels: [0.1, 0.9] for real, [0.9, 0.1] for fake (if smoothing=0.1)
                smooth = self.label_smoothing
                soft_labels = torch.full_like(logits, smooth / (num_classes - 1))
                soft_labels.scatter_(1, labels.unsqueeze(1), 1.0 - smooth)

                # Cross entropy with soft labels
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                loss = -(soft_labels * log_probs).sum(dim=-1).mean()
            else:
                loss = outputs.loss

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            epoch_loss += loss.item()
            step += 1

        return epoch_loss / max(step, 1)

    def train_generator_sft(
        self,
        success_records: List[Dict[str, Any]],
        all_fake_samples: List[Dict[str, Any]],
        round_id: int,
    ) -> None:
        """
        Mini SFT on high-score fake samples using LoRA-adapted generator.
        """
        if (
            not self.gen_sft_enabled
            or not self.gen_use_lora
            or self.gen_model is None
        ):
            return

        device = next(self.gen_model.parameters()).device
        records = [
            r
            for r in success_records
            if r.get("fool")
            and r.get("prob_true", 0.0) >= self.gen_success_threshold
        ]

        warmup_used = False
        if not records and round_id <= self.gen_sft_warmup_rounds:
            warmup_used = True
            candidates = [
                f for f in all_fake_samples if f.get("label", 1) == 0
            ]
            candidates = sorted(
                candidates, key=lambda x: x.get("prob_true", 0.0), reverse=True
            )
            records = candidates[: self.gen_sft_max_samples]

        if not records:
            return

        samples = sorted(records, key=lambda x: x["prob_true"], reverse=True)[
            : self.gen_sft_max_samples
        ]

        r_val = None
        peft_cfg = getattr(self.gen_model, "peft_config", {})
        if isinstance(peft_cfg, dict) and peft_cfg:
            first_cfg = next(iter(peft_cfg.values()))
            r_val = getattr(first_cfg, "r", None)
        if r_val is None:
            r_val = getattr(self.gen_model, "lora_r", "n/a")

        print(
            f"    [LoRA SFT] Starting mini fine-tune on {len(samples)} samples "
            f"(r={r_val}, "
            f"warmup={'yes' if warmup_used else 'no'})"
        )
        self.gen_model.train()
        for step in range(self.gen_sft_steps):
            batch = random.sample(
                samples, k=min(self.gen_sft_batch_size, len(samples))
            )
            texts = []
            for rec in batch:
                prompt_text = rec.get("prompt_text", "")
                fake_news = rec.get("fake_news_text", "")
                combined = f"{prompt_text}\n{fake_news}".strip()
                texts.append(combined)

            enc = self.gen_tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.gen_sft_max_length,
            ).to(device)

            outputs = self.gen_model(**enc, labels=enc["input_ids"])
            ce_loss = outputs.loss

            # KL regularization against frozen base model to avoid drift
            kl_loss = 0.0
            if self.gen_base_model is not None:
                with torch.no_grad():
                    base_outputs = self.gen_base_model(**enc)
                kl_loss = torch.nn.functional.kl_div(
                    torch.nn.functional.log_softmax(outputs.logits, dim=-1),
                    torch.nn.functional.softmax(base_outputs.logits, dim=-1),
                    reduction="batchmean",
                )

            total_loss = ce_loss + self.gen_lambda_kl * kl_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.gen_model.parameters(), self.gen_max_grad_norm
            )
            if self.gen_optimizer is not None:
                self.gen_optimizer.step()
                self.gen_optimizer.zero_grad(set_to_none=True)
        self.gen_model.eval()
    def run_round(self, round_id: int) -> Dict[str, Any]:
        print(f"\n===== Round {round_id} =====")
        real_dataset = self._get_real_samples_for_round(round_id)
        real_samples: List[Dict[str, Any]] = []
        fake_samples: List[Dict[str, Any]] = []
        successful_records: List[Dict[str, Any]] = []
        collect_train_signals = self.gen_sft_enabled and self.gen_use_lora

        def _truncate_text(text: str, limit: int = 800) -> str:
            if not isinstance(text, str):
                text = str(text)
            return text if len(text) <= limit else text[:limit] + "... [truncated]"

        first_real_ref: Optional[Dict[str, str]] = None
        first_fake_ref: Optional[Dict[str, str]] = None

        for idx, example in enumerate(real_dataset, start=1):
            # Compose discriminator input for the real article.
            real_text = format_discriminator_input(
                example,
                rag=self.args.disc_use_rag,
                prefix="",
                source=self.args.rag_source,
            )
            real_samples.append({"text": real_text, "label": 1})
            if first_real_ref is None:
                first_real_ref = {
                    "title": (example.get("title") or "").strip(),
                    "article": (
                        example.get("description")
                        or example.get("article")
                        or ""
                    ).strip(),
                }

            # Build retrieval context for the generator.
            rag_context = self._build_context(example)
            if (
                self.args.filter_no_wiki
                and self.args.rag_source == "wiki"
                and not rag_context.strip()
            ):
                # skip samples with no wiki hits when filtering is enabled
                continue
            # Only allow Wikipedia fallback when rag_source explicitly asks for wiki
            use_wiki = (
                self.args.generator_use_wiki
                and self.args.rag_source == "wiki"
                and not rag_context
            )

            feedback = example.get("feedback_prompt")
            gen_output = self.generator.generate(
                title=example.get("title", ""),
                content=example.get("description", ""),
                feedback_prompt=feedback,
                use_rag=use_wiki,
                context_override=rag_context if rag_context else None,
                rag_query=example.get("title", ""),
                num_rag_results=self.args.num_rag_results,
                lang=self.args.rag_lang,
                train_mode=collect_train_signals,
            )

            if isinstance(gen_output, dict):
                fake = gen_output.get("text", "")
                prompt_text = gen_output.get("prompt", "")
                prompt_ids = gen_output.get("prompt_ids")
                generated_ids = gen_output.get("generated_ids")
            else:
                fake = gen_output
                prompt_text = ""
                prompt_ids = None
                generated_ids = None

            # 只保留生成的文章內容（Title/Body），避免把 prompt/指令餵給判別器
            gen_title, gen_body = extract_article(fake)
            clean_generated = f"{gen_title}\n{gen_body}".strip()
            if first_fake_ref is None:
                first_fake_ref = {
                    "title": gen_title.strip(),
                    "body": gen_body.strip(),
                }

            fake_disc_input = format_discriminator_input(
                example,
                rag=self.args.disc_use_rag,
                prefix="",
                description_override=clean_generated,
                source=self.args.rag_source,
            )

            # Temporarily switch to eval mode for stable inference; restore original mode after.
            prev_training = self.discriminator.model.training
            self.discriminator.model.eval()

            # 使用強化版的 detailed feedback
            detailed = self.discriminator.get_detailed_feedback(
                fake_disc_input, rag_context=rag_context, top_k=5
            )
            fake_prob_true = detailed["prob_true"]
            suspicious = ", ".join(
                [f"{w}({s:.2f})" for w, s in detailed["suspicious_words"]]
            )

            if prev_training:
                self.discriminator.model.train()

            # 強化版 Feedback Prompt - 這是 "Verbal Adversarial Feedback" 的核心
            feedback_lines = [
                f"=== DISCRIMINATOR FEEDBACK (Round {round_id}) ===",
                f"Detection Result: Your previous version was classified as FAKE",
                f"Confidence: {detailed['confidence'].upper()} ({detailed['prob_fake']:.1%} fake probability)",
                "",
                "Problems Identified:",
            ]

            for i, reason in enumerate(detailed["detection_reasons"], 1):
                reason_readable = reason.replace("_", " ").title()
                feedback_lines.append(f"  {i}. {reason_readable}")

            feedback_lines.append("")
            feedback_lines.append(f"Flagged Suspicious Terms: {suspicious}")
            feedback_lines.append("")
            feedback_lines.append("Improvement Instructions:")

            for i, suggestion in enumerate(detailed["improvement_suggestions"], 1):
                feedback_lines.append(f"  {i}. {suggestion}")

            # === 新增：Few-shot 成功案例 ===
            if self.successful_examples:
                feedback_lines.append("")
                feedback_lines.append("=" * 50)
                feedback_lines.append("SUCCESSFUL EXAMPLE (This fooled the detector!):")
                feedback_lines.append("=" * 50)
                # 取最近一個成功案例
                best_example = self.successful_examples[-1]
                # 只取前 500 字避免 prompt 太長
                feedback_lines.append(best_example["text"][:500])
                feedback_lines.append("...")
                feedback_lines.append(
                    f"[This achieved {best_example['prob_true']:.1%} real probability]"
                )
                feedback_lines.append("")
                feedback_lines.append(
                    "LEARN FROM THIS: Mimic the style and tone of the successful example above."
                )

            feedback_lines.append("")
            feedback_lines.append(
                "CRITICAL: Your rewrite MUST address these issues to pass detection."
            )

            example["feedback_prompt"] = "\n".join(feedback_lines)

            fake_samples.append(
                {
                    "text": fake_disc_input,
                    "label": 0,
                    "prob_true": fake_prob_true,
                    "prob_fake": detailed["prob_fake"],
                    "confidence": detailed["confidence"],
                    "suspicious_words": suspicious,
                    "detection_reasons": detailed["detection_reasons"],
                    "feedback_prompt": example["feedback_prompt"],
                    "generated_content": clean_generated,  # 保存生成內容，用於 few-shot
                    "prompt_text": prompt_text,
                    "prompt_ids": prompt_ids,
                    "generated_ids": generated_ids,
                    "fool": fake_prob_true > detailed["prob_fake"],
                }
            )

            if idx % self.args.log_interval == 0 or idx == len(real_dataset):
                total = len(real_dataset)
                print(f"  Processed {idx}/{total} samples (generation + disc prep)")

        # === 新增：先計算 fool_rate，收集成功案例 ===
        prob_trues = [s["prob_true"] for s in fake_samples]
        fooled_count = sum(1 for s in fake_samples if s.get("fool"))
        fool_rate = fooled_count / max(len(fake_samples), 1)

        # 收集成功騙過 D 的案例（需為假新聞且 fooled）
        new_successes = [
            s
            for s in fake_samples
            if s.get("label", 1) == 0
            and s.get("fool")
            and s.get("prob_true", 0.0) > self.gen_success_threshold
        ]
        if new_successes:
            # 按 prob_true 排序，取最好的
            new_successes.sort(key=lambda x: x["prob_true"], reverse=True)
            for s in new_successes[:3]:  # 最多加 3 個
                self.successful_examples.append(
                    {
                        "text": s.get("generated_content", s["text"][:500]),
                        "prob_true": s["prob_true"],
                    }
                )
                successful_records.append(
                    {
                        "prompt_text": s.get("prompt_text", ""),
                        "fake_news_text": s.get("generated_content", ""),
                        "prob_true": s["prob_true"],
                        "prompt_ids": s.get("prompt_ids"),
                        "generated_ids": s.get("generated_ids"),
                        "fool": s.get("fool", True),
                    }
                )
            # 只保留最近 5 個成功案例
            self.successful_examples = self.successful_examples[-5:]
            print(
                f"  ✓ Collected {len(new_successes)} successful examples (total: {len(self.successful_examples)})"
            )

        # === 新增：動態平衡 - 決定是否訓練 D ===
        skip_training = False
        training_reason = ""

        if fool_rate < self.min_fool_rate_to_train:
            if self.rounds_without_training < self.max_skip_rounds:
                skip_training = True
                self.rounds_without_training += 1
                training_reason = f"SKIPPED (fool_rate {fool_rate:.1%} < {self.min_fool_rate_to_train:.1%}, giving G time to improve)"
            else:
                training_reason = f"FORCED (skipped {self.rounds_without_training} rounds, must train)"
                self.rounds_without_training = 0
        else:
            self.rounds_without_training = 0
            training_reason = f"NORMAL (fool_rate {fool_rate:.1%} >= {self.min_fool_rate_to_train:.1%})"

        # Train discriminator on mixed real/fake samples.
        mixed_samples = real_samples + fake_samples
        avg_loss = 0.0

        if skip_training:
            print(f"  ⏸️  Discriminator training: {training_reason}")
            # 還是要計算 loss 用於統計，但不更新參數
            self.discriminator.model.eval()
            with torch.no_grad():
                for s in mixed_samples[: self.args.batch_size]:
                    inputs = self.discriminator.tokenizer(
                        s["text"],
                        truncation=True,
                        padding=True,
                        max_length=self.args.max_length,
                        return_tensors="pt",
                    ).to(self.discriminator.device)
                    inputs["labels"] = torch.tensor(
                        [s["label"]], device=self.discriminator.device
                    )
                    outputs = self.discriminator.model(**inputs)
                    avg_loss += outputs.loss.item()
                avg_loss /= min(len(mixed_samples), self.args.batch_size)
            self.discriminator.model.train()
        else:
            print(f"  ▶️  Discriminator training: {training_reason}")
            for epoch in range(self.args.discriminator_epochs):
                print(
                    f"  Training discriminator epoch {epoch + 1}/{self.args.discriminator_epochs} (with label smoothing={self.label_smoothing})"
                )
                avg_loss = self._train_discriminator(
                    mixed_samples, use_label_smoothing=True
                )

        if successful_records or (
            self.gen_sft_enabled and round_id <= self.gen_sft_warmup_rounds
        ):
            self.train_generator_sft(successful_records, fake_samples, round_id)

        self.last_fool_rate = fool_rate

        if self.args.output_dir:
            out_dir = Path(self.args.output_dir) / f"round_{round_id}"
            out_dir.mkdir(parents=True, exist_ok=True)
            HFDataset.from_list(fake_samples).save_to_disk(out_dir)
            print(f"  Saved generated fakes to {out_dir}")

        # 計算更詳細的統計指標（prob_trues, fooled_count, fool_rate 已在前面計算）
        mean_prob_true = sum(prob_trues) / max(len(prob_trues), 1)
        min_prob_true = min(prob_trues) if prob_trues else 0
        max_prob_true = max(prob_trues) if prob_trues else 0

        # 統計 confidence 分布
        confidence_dist = {"high": 0, "medium": 0, "low": 0}
        for s in fake_samples:
            conf = s.get("confidence", "high")
            confidence_dist[conf] = confidence_dist.get(conf, 0) + 1

        # 統計 detection reasons 分布
        reason_dist = {}
        for s in fake_samples:
            for reason in s.get("detection_reasons", []):
                reason_dist[reason] = reason_dist.get(reason, 0) + 1

        round_stats = {
            "round": round_id,
            "avg_disc_loss": avg_loss,
            "num_fake": len(fake_samples),
            "num_real": len(real_samples),
            # Generator 進步指標 (這些是關鍵！)
            "mean_fake_prob_true": float(mean_prob_true),
            "min_fake_prob_true": float(min_prob_true),
            "max_fake_prob_true": float(max_prob_true),
            "fool_rate": float(fool_rate),  # 成功欺騙 D 的比例
            "fooled_count": fooled_count,
            # Detection 分析
            "confidence_distribution": confidence_dist,
            "detection_reasons": reason_dist,
            # 動態平衡資訊
            "disc_training_skipped": skip_training,
            "successful_examples_count": len(self.successful_examples),
        }

        # 印出關鍵指標（方便觀察 G vs D 動態）
        if first_real_ref and first_fake_ref:
            print("  ┌─ Sample Inspection (first item of round)")
            print(f"  │ [REAL] Title: {first_real_ref['title']}")
            print(f"  │ [REAL] Article: {_truncate_text(first_real_ref['article'])}")
            print(f"  │ [FAKE] Title: {first_fake_ref['title']}")
            print(f"  │ [FAKE] Article: {_truncate_text(first_fake_ref['body'])}")
            print("  └─ End Sample Inspection\n")
        print(f"\n  ┌─────────────────────────────────────────────────────")
        print(f"  │ ROUND {round_id} SUMMARY - Verbal Adversarial Feedback")
        print(f"  ├─────────────────────────────────────────────────────")
        print(f"  │ Generator Performance:")
        print(f"  │   Mean P(real): {mean_prob_true:.3f}  (↑ = G improving)")
        print(
            f"  │   Fool Rate:    {fool_rate:.1%} ({fooled_count}/{len(prob_trues)} samples)"
        )
        print(f"  │   Range:        [{min_prob_true:.3f}, {max_prob_true:.3f}]")
        print(f"  │   Few-shot Examples: {len(self.successful_examples)} stored")
        print(f"  │ Discriminator:")
        print(f"  │   Training:     {'SKIPPED ⏸️' if skip_training else 'ACTIVE ▶️'}")
        print(f"  │   Avg Loss:     {avg_loss:.4f}  (↓ = D improving)")
        print(
            f"  │   Confidence:   H:{confidence_dist['high']} M:{confidence_dist['medium']} L:{confidence_dist['low']}"
        )
        print(f"  │ Detection Reasons: {reason_dist}")
        print(f"  └─────────────────────────────────────────────────────\n")

        # Save discriminator checkpoints (best + last)
        if self.args.output_dir:
            base_dir = Path(self.args.output_dir)
            round_ckpt = base_dir / f"disc_round_{round_id}"
            round_ckpt.mkdir(parents=True, exist_ok=True)
            self.discriminator.model.save_pretrained(round_ckpt)
            self.discriminator.tokenizer.save_pretrained(round_ckpt)
            round_stats["disc_checkpoint"] = str(round_ckpt)

            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.best_path = base_dir / "disc_best"
                self.best_path.mkdir(parents=True, exist_ok=True)
                self.discriminator.model.save_pretrained(self.best_path)
                self.discriminator.tokenizer.save_pretrained(self.best_path)
                round_stats["disc_best_checkpoint"] = str(self.best_path)

        print(f"  Round summary: {round_stats}")
        return round_stats

    def _determine_samples_per_round(self) -> int:
        if self.args.real_samples_per_round:
            return max(1, min(self.args.real_samples_per_round, self.dataset_len))
        inferred = _slice_length_from_split(self.args.dataset_split)
        if inferred and inferred > 0:
            return min(inferred, self.dataset_len)
        return self.dataset_len

    def _get_real_samples_for_round(self, round_id: int) -> List[Dict[str, Any]]:
        start = (round_id - 1) * self.real_samples_per_round
        if start >= self.dataset_len and not self._wrapped_warned:
            print(
                f"  Note: only {self.dataset_len} unique real samples are available "
                f"and each round uses {self.real_samples_per_round}; "
                f"rounds beyond {self.max_unique_rounds} will reuse examples."
            )
            self._wrapped_warned = True
        start_mod = start % self.dataset_len
        end = start_mod + self.real_samples_per_round
        if end <= self.dataset_len:
            return self.dataset[start_mod:end]
        return self.dataset[start_mod:] + self.dataset[: end - self.dataset_len]


def load_news_data(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.dataset_path and Path(args.dataset_path).exists():
        ds = load_from_disk(args.dataset_path)
    else:
        load_kwargs = {"split": args.dataset_split}
        if args.dataset_config:
            load_kwargs["name"] = args.dataset_config
        ds = load_dataset(args.dataset_name, **load_kwargs)

    # For CNN/DailyMail: keep the shortest 10k articles to speed training
    if args.dataset_name == "cnn_dailymail":

        def _len_fn(x):
            article = x.get("article", "") or x.get("description", "")
            return {"desc_len": len(article)}

        ds = ds.map(_len_fn, num_proc=1)
        ds = ds.sort("desc_len")
        ds = ds.select(range(min(len(ds), 10_000)))
        ds = ds.remove_columns(["desc_len"])
        ds = ds.shuffle(seed=42)

    examples = []
    for row in ds:
        example = dict(row)

        # Normalize common schemas
        if "cnn_dailymail" in args.dataset_name:
            # CNN/DailyMail: fields are article, highlights, id
            article = row.get("article", "")
            highlights = row.get("highlights", "")
            example["title"] = highlights.strip() or article[:80].strip() or "Untitled"
            example["description"] = article
            example["date_publish"] = datetime.datetime.utcnow()
        else:
            example["date_publish"] = _normalize_date(example.get("date_publish"))
            if "description" not in example and "text" in example:
                example["description"] = example["text"]

        examples.append(example)
    return examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RAG-GAN training loop for fake news detection."
    )
    parser.add_argument(
        "--dataset-name", type=str, default="sanxing/advfake_news_please"
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default=None,
        help="Optional dataset config/name (e.g., 3.0.0 for cnn_dailymail).",
    )
    parser.add_argument("--dataset-split", type=str, default="train")
    parser.add_argument(
        "--dataset-path", type=str, help="Optional local dataset path (load_from_disk)."
    )
    parser.add_argument("--num-rounds", type=int, default=2)
    parser.add_argument("--discriminator-epochs", type=int, default=1)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Discriminator batch size (reduce if CUDA OOM).",
    )
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max-length", type=int, default=MAX_ENCODER_SEQ_LEN)
    parser.add_argument(
        "--rag-source", type=str, choices=["google", "wiki", "none"], default="wiki"
    )
    parser.add_argument(
        "--disc-use-rag",
        action="store_true",
        help="Enable RAG for discriminator input.",
    )
    parser.add_argument(
        "--no-disc-rag",
        dest="disc_use_rag",
        action="store_false",
        help="Disable RAG for discriminator input.",
    )
    parser.add_argument(
        "--generator-use-wiki",
        action="store_true",
        help="Use Wikipedia when no external context is provided.",
    )
    parser.add_argument(
        "--no-generator-wiki",
        dest="generator_use_wiki",
        action="store_false",
        help="Disable Wikipedia fallback.",
    )
    parser.add_argument(
        "--filter-no-wiki",
        action="store_true",
        help="Skip samples whose wiki search returns empty context.",
    )
    parser.add_argument("--num-rag-results", type=int, default=3)
    parser.add_argument("--rag-lang", type=str, default="en")
    parser.add_argument("--generator-model", type=str, default=MODEL_ID)
    parser.add_argument(
        "--discriminator-model", type=str, default=DEFAULT_ENCODER_MODEL
    )
    parser.add_argument(
        "--gen-sft-every-round",
        action="store_true",
        default=os.environ.get("GEN_SFT_EVERY_ROUND", "0").lower()
        in ("1", "true", "yes"),
        help="Run a small SFT step on the generator after each round (LoRA only, local mode).",
    )
    parser.add_argument(
        "--no-gen-sft",
        dest="gen_sft_every_round",
        action="store_false",
        help="Disable generator SFT regardless of env flag.",
    )
    parser.add_argument(
        "--gen-sft-lr",
        type=float,
        default=5e-5,
        help="Learning rate for generator mini SFT (LoRA params).",
    )
    parser.add_argument(
        "--gen-sft-steps",
        type=int,
        default=2,
        help="Gradient steps per round for mini SFT.",
    )
    parser.add_argument(
        "--gen-sft-batch-size",
        type=int,
        default=1,
        help="Batch size for mini SFT samples.",
    )
    parser.add_argument(
        "--gen-sft-max-length",
        type=int,
        default=512,
        help="Max sequence length for generator SFT inputs.",
    )
    parser.add_argument(
        "--gen-sft-success-threshold",
        type=float,
        default=0.55,
        help="Minimum discriminator P(real) to treat a fake as successful.",
    )
    parser.add_argument(
        "--gen-sft-max-samples",
        type=int,
        default=2,
        help="Number of top successful samples to SFT on each round.",
    )
    parser.add_argument(
        "--gen-sft-lambda-kl",
        type=float,
        default=0.01,
        help="KL penalty weight to keep LoRA close to base model.",
    )
    parser.add_argument(
        "--gen-sft-max-grad-norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm for generator SFT.",
    )
    parser.add_argument(
        "--gen-sft-warmup-rounds",
        type=int,
        default=3,
        help="Rounds to allow warm-up SFT when no successful fake samples exist.",
    )
    parser.add_argument(
        "--output-dir", type=str, help="Where to save generated fakes per round."
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Number of samples between progress logs.",
    )
    parser.add_argument(
        "--real-samples-per-round",
        type=int,
        help="Number of real samples to cycle through each round to keep batches disjoint.",
    )

    # === 動態平衡參數 ===
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.1,
        help="Label smoothing coefficient for discriminator training (0.0 = no smoothing, 0.1 = recommended).",
    )
    parser.add_argument(
        "--min-fool-rate-to-train",
        type=float,
        default=0.05,
        help="Skip discriminator training if fool rate is below this threshold (gives G time to improve).",
    )
    parser.add_argument(
        "--max-skip-rounds",
        type=int,
        default=2,
        help="Maximum consecutive rounds to skip discriminator training.",
    )

    parser.set_defaults(disc_use_rag=False, generator_use_wiki=True)
    return parser.parse_args()


def main() -> None:
    import json

    args = parse_args()

    # 顯示 Generator 模式
    print("\n" + "=" * 60)
    print("RAG²AN - Verbal Adversarial Feedback Training")
    print("=" * 60)
    print(f"Generator Mode: {GEN_MODE.upper()}")
    if GEN_MODE == "api":
        print(f"  API Model: {OPENAI_MODEL}")
    else:
        print(f"  Local Model: {MODEL_ID}")
    print(f"Discriminator: {args.discriminator_model}")
    print("=" * 60 + "\n")

    dataset = load_news_data(args)

    print(f"Loaded {len(dataset)} examples for training.")
    if args.log_interval <= 0:
        raise ValueError("--log-interval must be positive.")

    trainer = GANTrainer(args, dataset)
    all_stats = []

    for round_id in range(1, args.num_rounds + 1):
        stats = trainer.run_round(round_id)
        all_stats.append(stats)

    # 保存完整的訓練歷史到 JSON
    if args.output_dir:
        history_path = Path(args.output_dir) / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(all_stats, f, indent=2, default=str)
        print(f"\nTraining history saved to: {history_path}")

        # 印出最終摘要
        print("\n" + "=" * 60)
        print("FINAL TRAINING SUMMARY")
        print("=" * 60)
        print(f"{'Round':<8} {'Mean P(real)':<14} {'Fool Rate':<12} {'Disc Loss':<12}")
        print("-" * 60)
        for s in all_stats:
            print(
                f"{s['round']:<8} {s['mean_fake_prob_true']:<14.3f} {s['fool_rate']:<12.1%} {s['avg_disc_loss']:<12.4f}"
            )
        print("=" * 60)

        # 計算整體趨勢
        if len(all_stats) >= 2:
            first_fool_rate = all_stats[0]["fool_rate"]
            last_fool_rate = all_stats[-1]["fool_rate"]
            first_mean_p = all_stats[0]["mean_fake_prob_true"]
            last_mean_p = all_stats[-1]["mean_fake_prob_true"]

            print(f"\nGenerator Trend:")
            print(
                f"  Fool Rate:    {first_fool_rate:.1%} → {last_fool_rate:.1%} ({'↑ IMPROVING' if last_fool_rate > first_fool_rate else '↓ declining'})"
            )
            print(
                f"  Mean P(real): {first_mean_p:.3f} → {last_mean_p:.3f} ({'↑ IMPROVING' if last_mean_p > first_mean_p else '↓ declining'})"
            )


if __name__ == "__main__":
    main()
