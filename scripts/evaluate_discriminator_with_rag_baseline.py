#!/usr/bin/env python
"""
Evaluate a single LLM discriminator on advfake WITH RAG context.

- Same logic as no-RAG version:
  - Only evaluates DEFAULT_DISCRIMINATOR_MODEL (or --model-path)
  - Uses BF16 on CUDA
  - Model is instructed to output ONLY: REAL or FAKE
  - P(REAL) computed via log-prob scoring of label candidates (no confidence parsing)

RAG:
- Builds RAG-augmented text using src.discriminator.format_discriminator_input(...)
- Caches built inputs under local/rag_cache/*.pt
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.discriminator import MAX_ENCODER_SEQ_LEN, format_discriminator_input

# Pick ONE default model here

#DEFAULT_DISCRIMINATOR_MODEL = "Qwen/Qwen2.5-7B-Instruct"
#DEFAULT_DISCRIMINATOR_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_DISCRIMINATOR_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"


def _clean_text(val) -> str:
    if isinstance(val, str):
        return val.strip()
    if val is None:
        return ""
    return str(val).strip()


def _cache_key(dataset_path: Path, rag_source: str, num_rag_results: int) -> str:
    return "|".join(
        [
            str(dataset_path.resolve()),
            f"source={rag_source}",
            f"k={num_rag_results}",
            "query=description_override",
        ]
    )


def _default_cache_path(dataset_path: Path, rag_source: str, num_rag_results: int) -> Path:
    cache_dir = Path("local/rag_cache")
    dataset_id = dataset_path.stem
    return cache_dir / f"{dataset_id}_source-{rag_source}_k-{num_rag_results}.pt"


def build_rag_inputs(
    dataset_path: Path,
    rag_source: str,
    num_rag_results: int,
    cache_path: Optional[Path] = None,
) -> Tuple[List[str], List[int]]:
    cache_path = cache_path or _default_cache_path(dataset_path, rag_source, num_rag_results)
    cache_key = _cache_key(dataset_path, rag_source, num_rag_results)

    if cache_path.exists():
        try:
            cached = torch.load(cache_path)
            meta = cached.get("meta", {})
            if meta.get("key") == cache_key:
                print(f"  Loaded cached RAG inputs from {cache_path}")
                return list(cached["texts"]), list(cached["labels"])
            print("  Cache found but metadata mismatch; rebuilding RAG inputs.")
        except Exception as exc:
            print(f"  Failed to load cache at {cache_path}: {exc}. Rebuilding.")

    ds = Dataset.from_file(str(dataset_path))
    texts: List[str] = []
    labels: List[int] = []
    processed = 0

    for row in ds:
        real = _clean_text(row.get("description", ""))
        fake = _clean_text(row.get("f_description", ""))

        if not real and not fake:
            continue

        processed += 1
        if processed % 20 == 0:
            print(f"  Built RAG inputs for {processed} examples...")

        if real:
            text = format_discriminator_input(
                row,
                rag=True,
                prefix="",
                source=rag_source,
                num_results=num_rag_results,
            )
            texts.append(text)
            labels.append(1)

        if fake:
            text = format_discriminator_input(
                row,
                rag=True,
                prefix="",
                description_override=fake,
                source=rag_source,
                num_results=num_rag_results,
            )
            texts.append(text)
            labels.append(0)

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"meta": {"key": cache_key}, "texts": texts, "labels": labels}, cache_path)
        print(f"  Saved RAG inputs to cache at {cache_path}")
    except Exception as exc:
        print(f"  Warning: failed to write cache at {cache_path}: {exc}")

    return texts, labels


class LLMDiscriminator:
    """
    Scores P(REAL) by comparing log-prob of two label candidates: " REAL" vs " FAKE".
    """

    def __init__(
        self,
        model_id: str,
        max_input_length: int,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        debug_samples: int = 0,
    ):
        self.model_id = model_id
        self.max_input_length = max_input_length
        self.debug_samples = debug_samples

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if getattr(self.tokenizer, "padding_side", None) is not None:
            self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        dtype = torch_dtype
        if dtype is None and self.device.type == "cuda":
            dtype = torch.bfloat16

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
        ).to(self.device)
        self.model.eval()

        self.cand_text_real = " REAL"
        self.cand_text_fake = " FAKE"
        self.cand_ids_real = self.tokenizer.encode(self.cand_text_real, add_special_tokens=False)
        self.cand_ids_fake = self.tokenizer.encode(self.cand_text_fake, add_special_tokens=False)
        self.max_cand_len = max(len(self.cand_ids_real), len(self.cand_ids_fake))
        self.prompt_max_len = max(1, self.max_input_length - self.max_cand_len)

    def _build_messages(self, rag_text: str) -> List[dict]:
        system_prompt = (
            "You are a news fact-checking classifier. "
            "The provided content may include retrieved context (RAG). "
            "You must output exactly one word: REAL or FAKE. "
            "Do not output anything else."
        )
        user_prompt = (
            f"{rag_text}\n\n"
            "Question: Is the News REAL or FAKE?\n"
            "Answer (one word only):"
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _format_prompt(self, rag_text: str) -> str:
        messages = self._build_messages(rag_text)
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return f"{messages[0]['content']}\n\n{messages[1]['content']}\n"

    @staticmethod
    def _left_pad(seqs: List[torch.Tensor], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        max_len = max(int(x.numel()) for x in seqs)
        batch = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
        attn = torch.zeros((len(seqs), max_len), dtype=torch.long)
        for i, s in enumerate(seqs):
            L = int(s.numel())
            batch[i, -L:] = s
            attn[i, -L:] = 1
        return batch, attn

    @torch.no_grad()
    def _batch_logprobs_for_candidate(
        self,
        prompt_ids_list: List[torch.Tensor],
        cand_ids: List[int],
    ) -> torch.Tensor:
        cand = torch.tensor(cand_ids, dtype=torch.long)
        seqs = [torch.cat([p, cand], dim=0) for p in prompt_ids_list]
        input_ids, attention_mask = self._left_pad(seqs, self.tokenizer.pad_token_id)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits

        B, T, _ = logits.shape
        cand_len = int(cand.numel())

        logps = []
        for i in range(B):
            seq_len = int(attention_mask[i].sum().item())
            prompt_len = int(prompt_ids_list[i].numel())
            pad_left = T - seq_len
            lp = 0.0
            for j in range(cand_len):
                pos = pad_left + (prompt_len + j - 1)
                tok = int(cand_ids[j])
                lp += torch.log_softmax(logits[i, pos, :], dim=-1)[tok].item()
            logps.append(lp)

        return torch.tensor(logps, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def predict_probs_true(self, texts: Sequence[str], batch_size: int) -> List[float]:
        probs: List[float] = []
        seen = 0

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            prompts = [self._format_prompt(t) for t in batch_texts]

            tok = self.tokenizer(
                prompts,
                truncation=True,
                padding=True,
                max_length=self.prompt_max_len,
                return_tensors="pt",
            )
            input_ids = tok["input_ids"]
            attn = tok["attention_mask"]

            prompt_ids_list: List[torch.Tensor] = []
            for i in range(input_ids.shape[0]):
                plen = int(attn[i].sum().item())
                prompt_ids_list.append(input_ids[i, -plen:].clone().cpu())

            logp_real = self._batch_logprobs_for_candidate(prompt_ids_list, self.cand_ids_real)
            logp_fake = self._batch_logprobs_for_candidate(prompt_ids_list, self.cand_ids_fake)

            stacked = torch.stack([logp_real, logp_fake], dim=-1)
            p_real = torch.softmax(stacked, dim=-1)[:, 0]

            for i in range(len(batch_texts)):
                p = float(p_real[i].item())
                probs.append(p)

                if self.debug_samples > 0 and (seen + i) < self.debug_samples:
                    pred = "REAL" if p >= 0.5 else "FAKE"
                    print(f"[debug] sample={seen+i} p_real={p:.3f} pred={pred}")

            seen += len(batch_texts)

        return probs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ONE discriminator on advfake with RAG.")
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_DISCRIMINATOR_MODEL,
        help=f"HF model id or local path. Default: {DEFAULT_DISCRIMINATOR_MODEL}",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("local/hf_datasets/advfake/advfake-train.arrow"),
        help="Path to advfake arrow file (402 rows).",
    )
    parser.add_argument(
        "--rag-source",
        type=str,
        default="dpr",
        choices=["dpr", "none"],
        help="Retrieval source used for building the context.",
    )
    parser.add_argument(
        "--num-rag-results",
        type=int,
        default=3,
        help="Number of retrieved documents to include.",
    )
    parser.add_argument(
        "--rag-cache-path",
        type=Path,
        default=None,
        help="Override cache path (default: local/rag_cache/<dataset>_source-*_k-*.pt)",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for scoring.")
    parser.add_argument(
        "--max-length",
        type=int,
        default=MAX_ENCODER_SEQ_LEN,
        help="Max token length for discriminator inputs.",
    )
    parser.add_argument(
        "--debug-samples",
        type=int,
        default=0,
        help="Print debug info for first N samples.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")

    cache_path = args.rag_cache_path or _default_cache_path(
        args.dataset_path, args.rag_source, args.num_rag_results
    )
    print(
        f"Building or loading RAG inputs from {args.dataset_path} "
        f"(rag_source={args.rag_source}, cache={cache_path}) ..."
    )
    texts, labels = build_rag_inputs(
        args.dataset_path,
        rag_source=args.rag_source,
        num_rag_results=args.num_rag_results,
        cache_path=cache_path,
    )
    n_real = sum(labels)
    n_fake = len(labels) - n_real
    print(f"Prepared {len(labels)} RAG-augmented samples (real={n_real}, fake={n_fake}).")

    print(f"\nEvaluating model (with RAG): {args.model_path}")
    disc = LLMDiscriminator(
        model_id=args.model_path,
        max_input_length=args.max_length,
        torch_dtype=torch.bfloat16,  # keep bf16 for CUDA usage
        debug_samples=args.debug_samples,
    )

    probs_true = disc.predict_probs_true(texts, batch_size=args.batch_size)
    preds = [1 if p >= 0.5 else 0 for p in probs_true]

    macro_f1 = f1_score(labels, preds, average="macro")
    roc_auc = roc_auc_score(labels, probs_true)
    accuracy = accuracy_score(labels, preds)

    print("\n=== Results (RAG) ===")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"ROC-AUC:  {roc_auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()