import argparse
import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple


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
from generator import MODEL_ID, FakeNewsGenerator, search_wikipedia


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


class GANTrainer:
    """
    Tie the RAG-enabled generator and discriminator into a simple GAN-style loop.
    The generator crafts fake news with retrieval context, while the discriminator
    learns to separate real and generated samples and provides feedback.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.generator = FakeNewsGenerator(model_id=args.generator_model)
        self.discriminator = get_encoder_discriminator(model_name=args.discriminator_model)
        self.discriminator.model.train()
        self.optimizer = torch.optim.AdamW(self.discriminator.model.parameters(), lr=args.lr)
        self.best_loss = float("inf")
        self.best_path = None

    def _build_context(self, example: Dict[str, Any]) -> str:
        if self.args.rag_source == "none":
            return ""
        if self.args.rag_source == "wiki":
            query = example.get("title") or str(example.get("description", ""))[:50]
            return search_wikipedia(query, num_results=self.args.num_rag_results, lang=self.args.rag_lang)
        if self.args.rag_source == "google":
            return get_retrieval_ctx(example, prefix="", source="google")
        return ""

    def _collate(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts = [item["text"] for item in batch]
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long, device=self.discriminator.device)
        encoded = self.discriminator.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.args.max_length,
            return_tensors="pt",
        ).to(self.discriminator.device)
        encoded["labels"] = labels
        return encoded

    def _train_discriminator(self, samples: List[Dict[str, Any]]) -> float:
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
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            with torch.no_grad():
                preds = outputs.logits.argmax(dim=-1)
                acc = (preds == batch["labels"]).float().mean().item()
            epoch_loss += loss.item()
            step += 1

        return epoch_loss / max(step, 1)

    def run_round(self, round_id: int, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        print(f"\n===== Round {round_id} =====")
        real_samples: List[Dict[str, Any]] = []
        fake_samples: List[Dict[str, Any]] = []

        for idx, example in enumerate(dataset, start=1):
            # Compose discriminator input for the real article.
            real_text = format_discriminator_input(
                example,
                rag=self.args.disc_use_rag,
                prefix="",
                source=self.args.rag_source,
            )
            real_samples.append({"text": real_text, "label": 1})

            # Build retrieval context for the generator.
            rag_context = self._build_context(example)
            if self.args.filter_no_wiki and self.args.rag_source == "wiki" and not rag_context.strip():
                # skip samples with no wiki hits when filtering is enabled
                continue
            # Only allow Wikipedia fallback when rag_source explicitly asks for wiki
            use_wiki = (
                self.args.generator_use_wiki
                and self.args.rag_source == "wiki"
                and not rag_context
            )

            feedback = example.get("feedback_prompt")
            fake = self.generator.generate(
                title=example.get("title", ""),
                content=example.get("description", ""),
                feedback_prompt=feedback,
                use_rag=use_wiki,
                context_override=rag_context if rag_context else None,
                rag_query=example.get("title", ""),
                num_rag_results=self.args.num_rag_results,
                lang=self.args.rag_lang,
            )

            # 只保留生成的文章內容（Title/Body），避免把 prompt/指令餵給判別器
            gen_title, gen_body = extract_article(fake)
            clean_generated = f"{gen_title}\n{gen_body}".strip()

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
            fake_prob_true = self.discriminator.predict_prob(fake_disc_input)
            suspicious = self.discriminator.get_suspicious_words(fake_disc_input)
            if prev_training:
                self.discriminator.model.train()

            # Feedback is stored back into the example for the next round.
            example["feedback_prompt"] = (
                f"Your last version looked suspicious because of: {suspicious}. "
                f"Rewrite to be more consistent with the retrieved context."
            )

            fake_samples.append(
                {
                    "text": fake_disc_input,
                    "label": 0,
                    "prob_true": fake_prob_true,
                    "suspicious_words": suspicious,
                    "feedback_prompt": example["feedback_prompt"],
                }
            )

            if idx % 10 == 0:
                total = len(dataset)
                print(f"  Processed {idx}/{total} samples (generation + disc prep)")

        # Train discriminator on mixed real/fake samples.
        mixed_samples = real_samples + fake_samples
        avg_loss = 0.0
        for epoch in range(self.args.discriminator_epochs):
            print(f"  Training discriminator epoch {epoch + 1}/{self.args.discriminator_epochs}")
            avg_loss = self._train_discriminator(mixed_samples)

        if self.args.output_dir:
            out_dir = Path(self.args.output_dir) / f"round_{round_id}"
            out_dir.mkdir(parents=True, exist_ok=True)
            HFDataset.from_list(fake_samples).save_to_disk(out_dir)
            print(f"  Saved generated fakes to {out_dir}")

        round_stats = {
            "avg_disc_loss": avg_loss,
            "num_fake": len(fake_samples),
            "num_real": len(real_samples),
            "mean_fake_prob_true": float(
                sum(s["prob_true"] for s in fake_samples) / max(len(fake_samples), 1)
            ),
        }

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


def load_news_data(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.dataset_path and Path(args.dataset_path).exists():
        ds = load_from_disk(args.dataset_path)
    else:
        load_kwargs = {"split": args.dataset_split}
        if args.dataset_config:
            load_kwargs["name"] = args.dataset_config
        ds = load_dataset(args.dataset_name, **load_kwargs)

    # For CNN/DailyMail: drop the longer half based on article/description length
    if args.dataset_name == "cnn_dailymail":
        def _len_fn(x):
            article = x.get("article", "") or x.get("description", "")
            return {"desc_len": len(article)}
        ds = ds.map(_len_fn, num_proc=1)
        ds = ds.sort("desc_len")
        ds = ds.select(range(len(ds) // 2))
        ds = ds.remove_columns(["desc_len"])

    if args.max_samples:
        ds = ds.select(range(args.max_samples))

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
    parser = argparse.ArgumentParser(description="RAG-GAN training loop for fake news detection.")
    parser.add_argument("--dataset-name", type=str, default="sanxing/advfake_news_please")
    parser.add_argument("--dataset-config", type=str, default=None, help="Optional dataset config/name (e.g., 3.0.0 for cnn_dailymail).")
    parser.add_argument("--dataset-split", type=str, default="train[:64]")
    parser.add_argument("--dataset-path", type=str, help="Optional local dataset path (load_from_disk).")
    parser.add_argument("--max-samples", type=int, default=64, help="Limit number of samples for a quick run.")
    parser.add_argument("--num-rounds", type=int, default=2)
    parser.add_argument("--discriminator-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max-length", type=int, default=MAX_ENCODER_SEQ_LEN)
    parser.add_argument("--rag-source", type=str, choices=["google", "wiki", "none"], default="wiki")
    parser.add_argument("--disc-use-rag", action="store_true", help="Enable RAG for discriminator input.")
    parser.add_argument("--no-disc-rag", dest="disc_use_rag", action="store_false", help="Disable RAG for discriminator input.")
    parser.add_argument("--generator-use-wiki", action="store_true", help="Use Wikipedia when no external context is provided.")
    parser.add_argument("--no-generator-wiki", dest="generator_use_wiki", action="store_false", help="Disable Wikipedia fallback.")
    parser.add_argument("--filter-no-wiki", action="store_true", help="Skip samples whose wiki search returns empty context.")
    parser.add_argument("--num-rag-results", type=int, default=3)
    parser.add_argument("--rag-lang", type=str, default="en")
    parser.add_argument("--generator-model", type=str, default=MODEL_ID)
    parser.add_argument("--discriminator-model", type=str, default=DEFAULT_ENCODER_MODEL)
    parser.add_argument("--output-dir", type=str, help="Where to save generated fakes per round.")

    parser.set_defaults(disc_use_rag=False, generator_use_wiki=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_news_data(args)

    print(f"Loaded {len(dataset)} examples for training.")
    trainer = GANTrainer(args)
    for round_id in range(1, args.num_rounds + 1):
        trainer.run_round(round_id, dataset)


if __name__ == "__main__":
    main()
