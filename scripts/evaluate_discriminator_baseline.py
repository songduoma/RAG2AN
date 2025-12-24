#!/usr/bin/env python
"""
Evaluate a single LLM discriminator on advfake (no RAG).

- Only evaluates DEFAULT_DISCRIMINATOR_MODEL (or --model-path).
- Uses BF16 (bfloat16) when on CUDA (kept "always on" for GPU usage).
- The LLM is instructed to output ONLY: REAL or FAKE (no confidence).
- We DO NOT parse generated confidence; instead, we score P(REAL) via log-prob of labels.

Dataset:
- Loads local/hf_datasets/advfake/advfake-train.arrow (402 rows)
- Uses 'description' as real, 'f_description' as fake => 804 samples total
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

from src.discriminator import MAX_ENCODER_SEQ_LEN

# Pick ONE default model here
#DEFAULT_DISCRIMINATOR_MODEL = "Qwen/Qwen2.5-7B-Instruct"
#DEFAULT_DISCRIMINATOR_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_DISCRIMINATOR_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def _clean_text(val) -> str:
    if isinstance(val, str):
        return val.strip()
    if val is None:
        return ""
    return str(val).strip()


def load_advfake_pairs(dataset_path: Path) -> Tuple[List[str], List[int]]:
    """
    Returns texts + labels where 1 = real description, 0 = fake rewrite.
    """
    ds = Dataset.from_file(str(dataset_path))
    texts: List[str] = []
    labels: List[int] = []

    for row in ds:
        real = _clean_text(row.get("description", ""))
        fake = _clean_text(row.get("f_description", ""))

        if real:
            texts.append(real)
            labels.append(1)
        if fake:
            texts.append(fake)
            labels.append(0)

    return texts, labels


class LLMDiscriminator:
    """
    Scores P(REAL) by comparing log-prob of two label candidates: " REAL" vs " FAKE".
    This avoids needing the model to output a numeric confidence.
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

        # "bf16 always on" (practically for CUDA; CPU bf16 may be unsupported/slow)
        dtype = torch_dtype
        if dtype is None and self.device.type == "cuda":
            dtype = torch.bfloat16

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
        ).to(self.device)
        self.model.eval()

        # Candidate labels (leading space tends to match natural continuation)
        self.cand_text_real = " REAL"
        self.cand_text_fake = " FAKE"
        self.cand_ids_real = self.tokenizer.encode(self.cand_text_real, add_special_tokens=False)
        self.cand_ids_fake = self.tokenizer.encode(self.cand_text_fake, add_special_tokens=False)
        self.max_cand_len = max(len(self.cand_ids_real), len(self.cand_ids_fake))

        # Keep prompt length <= (max_input_length - max_cand_len)
        self.prompt_max_len = max(1, self.max_input_length - self.max_cand_len)

    def _build_messages(self, news_text: str) -> List[dict]:
        system_prompt = (
            "You are a news fact-checking classifier. "
            "You must output exactly one word: REAL or FAKE. "
            "Do not output anything else."
        )
        user_prompt = (
            f"News:\n{news_text}\n\n"
            "Question: Is this news REAL or FAKE?\n"
            "Answer (one word only):"
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _format_prompt(self, news_text: str) -> str:
        messages = self._build_messages(news_text)
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        # Fallback if no chat template
        return f"{messages[0]['content']}\n\n{messages[1]['content']}\n"

    @staticmethod
    def _left_pad(seqs: List[torch.Tensor], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Left-pad variable-length 1D tensors into a batch tensor.
        Returns (input_ids, attention_mask).
        """
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
        """
        For each prompt_ids in the batch, compute log P(cand_ids | prompt).
        Returns tensor shape (B,).
        """
        cand = torch.tensor(cand_ids, dtype=torch.long)
        seqs = [torch.cat([p, cand], dim=0) for p in prompt_ids_list]
        input_ids, attention_mask = self._left_pad(seqs, self.tokenizer.pad_token_id)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits  # (B, T, V)

        B, T, _ = logits.shape
        cand_len = int(cand.numel())

        # Compute logprob token-by-token for the candidate tail
        logps = []
        for i in range(B):
            seq_len = int(attention_mask[i].sum().item())
            prompt_len = int(prompt_ids_list[i].numel())
            # Positions (within the unpadded sequence) where candidate tokens appear:
            # cand token j is at position (prompt_len + j)
            # its probability is from logits at position (prompt_len + j - 1)
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
        """
        Returns P(TRUE=REAL) for each input text.
        """
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

            # Extract the *unpadded* prompt ids per sample (since padding is left)
            prompt_ids_list: List[torch.Tensor] = []
            for i in range(input_ids.shape[0]):
                plen = int(attn[i].sum().item())
                prompt_ids_list.append(input_ids[i, -plen:].clone().cpu())

            logp_real = self._batch_logprobs_for_candidate(prompt_ids_list, self.cand_ids_real)
            logp_fake = self._batch_logprobs_for_candidate(prompt_ids_list, self.cand_ids_fake)

            # softmax over {REAL, FAKE}
            stacked = torch.stack([logp_real, logp_fake], dim=-1)  # (B,2)
            p_real = torch.softmax(stacked, dim=-1)[:, 0]  # (B,)

            for i in range(len(batch_texts)):
                p = float(p_real[i].item())
                probs.append(p)

                if self.debug_samples > 0 and (seen + i) < self.debug_samples:
                    pred = "REAL" if p >= 0.5 else "FAKE"
                    print(f"[debug] sample={seen+i} p_real={p:.3f} pred={pred}")

            seen += len(batch_texts)

        return probs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ONE discriminator on advfake (no RAG).")
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
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for scoring.")
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

    print(f"Loading advfake pairs from {args.dataset_path} ...")
    texts, labels = load_advfake_pairs(args.dataset_path)
    n_real = sum(labels)
    n_fake = len(labels) - n_real
    print(f"Loaded {len(labels)} samples (real={n_real}, fake={n_fake}).")

    print(f"\nEvaluating model: {args.model_path}")
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

    print("\n=== Results ===")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"ROC-AUC:  {roc_auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()