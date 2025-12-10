"""
Evaluate discriminator checkpoints on eval.csv and report:
- Accuracy (threshold=0.5)
- Macro-F1
- ROC-AUC
- Optional RAG (same path as training: format_discriminator_input + google/none/dpr)

Assumptions:
- Checkpoints are in local/rag_gan_runs/20251210_005557/disc_round_{1..9}
- eval.csv has columns: text, rating (True/False)
- Tokenizer/model are fully available locally (no downloads).
"""
import argparse
import json
import re
import sys
from pathlib import Path
from typing import List
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Allow importing project modules when running as a script
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from discriminator import format_discriminator_input, get_retrieval_ctx  # noqa: E402


EVAL_PATH = Path("FakeNews_Task3_2022/eval.csv")
CHECKPOINT_DIR = Path("local/rag_gan_runs/20251210_005557")
CHECKPOINTS = [CHECKPOINT_DIR / f"disc_round_{i}" for i in range(1, 12)]
BATCH_SIZE = 8
MAX_LEN = 512  # align with discriminator default


def first_sentence(text: str, fallback_len: int = 150) -> str:
    """Extract first sentence; fallback to leading chars."""
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if parts:
        return parts[0]
    return text[:fallback_len].strip()


class TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int]):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


def detect_positive_label_id(model) -> int:
    """Try to infer which label id corresponds to TRUE/REAL/ENTAIL/etc."""
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    for idx, label in id2label.items():
        label_lower = str(label).lower()
        if any(
            key in label_lower
            for key in ("true", "real", "entail", "support", "pos", "positive")
        ):
            return int(idx)
    # fallbacks
    if 1 in id2label:
        return 1
    return 0


def load_eval():
    df = pd.read_csv(EVAL_PATH)
    # 保險：只留乾淨的 True/False
    df = df[df["rating"].isin([True, False, "True", "False", "true", "false"])]
    labels = df["rating"].astype(str).str.lower().map({"true": 1, "false": 0}).tolist()
    texts = df["text"].astype(str).tolist()
    titles = df["title"].fillna("").astype(str).tolist() if "title" in df.columns else [
        ""
    ] * len(texts)
    return texts, titles, labels


def build_examples(texts: List[str], titles: List[str]) -> List[dict]:
    """Create examples consistent with training format_discriminator_input."""
    examples = []
    for text, title in zip(texts, titles):
        title_val = title if title else first_sentence(text)
        examples.append(
            {
                "title": title_val,
                "description": text,
                "date_publish": None,
                "url": "",
            }
        )
    return examples


def build_formatted_inputs(
    examples: List[dict],
    use_rag: bool,
    rag_source: str,
    cache_path: Path | None = None,
    save_cache: bool = False,
) -> List[str]:
    """Use the same formatter as training, with optional RAG cache."""

    cached_ctx: List[str] | None = None
    if cache_path and cache_path.exists():
        try:
            cached_ctx = [json.loads(line)["ctx"] for line in cache_path.read_text().splitlines()]
            if len(cached_ctx) != len(examples):
                print(f"[warn] Cache size {len(cached_ctx)} != examples {len(examples)}, ignoring cache.")
                cached_ctx = None
            else:
                print(f"[cache] Loaded {len(cached_ctx)} contexts from {cache_path}")
        except Exception as e:
            print(f"[warn] Failed to load cache {cache_path}: {e}")
            cached_ctx = None

    contexts_to_save: List[str] = []
    formatted: List[str] = []
    total = len(examples)
    for idx, ex in enumerate(examples, start=1):
        ctx = None
        if use_rag:
            if cached_ctx:
                ctx = cached_ctx[idx - 1]
            else:
                ctx = get_retrieval_ctx(ex, prefix="", source=rag_source)
                contexts_to_save.append(ctx)
                if idx % 10 == 0:
                    print(f"[RAG] fetched {idx}/{total} contexts", flush=True)
        if ctx:
            text = format_discriminator_input(
                ex, rag=False, prefix="", description_override=None, source=rag_source
            )
            formatted.append(ctx + ("\n\n" if not ctx.endswith("\n\n") else "") + text)
        else:
            formatted.append(
                format_discriminator_input(
                    ex, rag=use_rag, prefix="", description_override=None, source=rag_source
                )
            )
    if save_cache and contexts_to_save and cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w") as f:
            for ctx in contexts_to_save:
                f.write(json.dumps({"ctx": ctx}) + "\n")
        print(f"[cache] Saved {len(contexts_to_save)} contexts to {cache_path}")
    return formatted


def evaluate_checkpoint(path: Path, dataset: TextDataset):
    tokenizer = AutoTokenizer.from_pretrained(
        path,
        use_fast=True,
        local_files_only=True,
        fix_mistral_regex=True,  # avoid incorrect tokenization for mistral-style tokenizers
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        path,
        output_attentions=False,
        trust_remote_code=True,
        local_files_only=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    pos_id = detect_positive_label_id(model)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    all_labels: List[int] = []
    all_probs: List[float] = []

    with torch.no_grad():
        for texts, labels in dataloader:
            inputs = tokenizer(
                list(texts),
                truncation=True,
                padding=True,
                max_length=MAX_LEN,
                return_tensors="pt",
            ).to(device)
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[:, pos_id].cpu().tolist()
            all_probs.extend(probs)
            # labels 這邊是 list[int]
            all_labels.extend(labels)

    # 使用固定 threshold=0.5
    preds = [1 if p >= 0.5 else 0 for p in all_probs]

    acc = accuracy_score(all_labels, preds)
    macro_f1 = f1_score(all_labels, preds, average="macro")
    roc_auc = roc_auc_score(all_labels, all_probs)

    return acc, macro_f1, roc_auc


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate discriminator checkpoints.")
    parser.add_argument(
        "--disc-use-rag",
        action="store_true",
        help="Enable RAG for discriminator input (same as training flag).",
    )
    parser.add_argument(
        "--rag-source",
        default="google",
        choices=["google", "dpr", "none"],
        help="RAG source for discriminator (default: google).",
    )
    parser.add_argument(
        "--rag-cache",
        type=str,
        default=None,
        help="Path to cache search contexts (jsonl). If exists, reuse; if absent and RAG is on, fetch and save.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    texts, titles, labels = load_eval()
    examples = build_examples(texts, titles)

    cache_path = Path(args.rag_cache) if args.rag_cache else None
    prepared_texts = build_formatted_inputs(
        examples,
        use_rag=args.disc_use_rag,
        rag_source=args.rag_source,
        cache_path=cache_path,
        save_cache=args.disc_use_rag and cache_path is not None,
    )

    dataset = TextDataset(prepared_texts, labels)
    mode = f"RAG={args.disc_use_rag} source={args.rag_source}"
    print(f"Loaded eval set: {len(dataset)} samples from {EVAL_PATH} ({mode})")

    for ckpt in CHECKPOINTS:
        if not ckpt.exists():
            print(f"[skip] {ckpt} not found")
            continue
        try:
            acc, macro_f1, roc_auc = evaluate_checkpoint(ckpt, dataset)
            print(
                f"{ckpt.name}: "
                f"Acc={acc:.4f}, "
                f"Macro-F1={macro_f1:.4f}, "
                f"ROC-AUC={roc_auc:.4f}"
            )
        except Exception as e:
            print(f"[error] {ckpt}: {e}")


if __name__ == "__main__":
    main()
