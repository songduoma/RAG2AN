#!/usr/bin/env python
"""
Evaluate each discriminator checkpoint with retrieval context (RAG) on advfake.

Produces Macro-F1 / ROC-AUC scores after prefixing every real/fake text with the
same DPR context used during GAN training.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from discriminator import (
    EncoderDiscriminator,
    MAX_ENCODER_SEQ_LEN,
    format_discriminator_input,
)


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
            "query=description_override",  # ensure cache invalidation when query uses override
        ]
    )


def _default_cache_path(
    dataset_path: Path, rag_source: str, num_rag_results: int
) -> Path:
    cache_dir = Path("local/rag_cache")
    dataset_id = dataset_path.stem
    return cache_dir / f"{dataset_id}_source-{rag_source}_k-{num_rag_results}.pt"


def build_rag_inputs(
    dataset_path: Path,
    rag_source: str,
    num_rag_results: int,
    cache_path: Optional[Path] = None,
) -> Tuple[List[str], List[int]]:
    cache_path = cache_path or _default_cache_path(
        dataset_path, rag_source, num_rag_results
    )
    cache_key = _cache_key(dataset_path, rag_source, num_rag_results)

    if cache_path.exists():
        try:
            cached = torch.load(cache_path)
            meta = cached.get("meta", {})
            if meta.get("key") == cache_key:
                print(f"  Loaded cached RAG inputs from {cache_path}")
                return list(cached["texts"]), list(cached["labels"])
            print("  Cache found but metadata mismatch; rebuilding RAG inputs.")
        except Exception as exc:  # pragma: no cover - defensive
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
        torch.save(
            {"meta": {"key": cache_key}, "texts": texts, "labels": labels},
            cache_path,
        )
        print(f"  Saved RAG inputs to cache at {cache_path}")
    except Exception as exc:  # pragma: no cover - defensive
        print(f"  Warning: failed to write cache at {cache_path}: {exc}")

    return texts, labels


@torch.no_grad()
def predict_probs(
    disc: EncoderDiscriminator, texts: Sequence[str], batch_size: int
) -> List[float]:
    probs: List[float] = []
    model = disc.model
    model.eval()

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        inputs = disc.tokenizer(
            list(batch_texts),
            truncation=True,
            padding=True,
            max_length=disc.max_length,
            return_tensors="pt",
        ).to(disc.device)

        logits = model(**inputs).logits
        batch_probs = torch.softmax(logits, dim=-1)[:, disc.positive_label_id]
        probs.extend(batch_probs.cpu().tolist())

    return probs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate discriminators using RAG-enabled inputs on advfake."
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("local/rag_gan_runs/G+D+_neither"),
        help="Directory containing disc_round_* checkpoints.",
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
        help=(
            "Path to cache the RAG-augmented texts/labels. "
            "Defaults to local/rag_cache/<dataset>_source-*_k-*.pt"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for inference (use smaller size if GPU memory limited).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=MAX_ENCODER_SEQ_LEN,
        help="Max token length for discriminator inputs.",
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

    base = args.models_dir
    if not base.exists():
        raise FileNotFoundError(f"Models dir not found: {base}")
    model_paths: Dict[str, str] = {
        sub.name: str(sub)
        for sub in sorted(base.iterdir())
        if sub.is_dir() and sub.name.startswith("disc_round_")
    }
    if not model_paths:
        raise FileNotFoundError(f"No disc_round_* folders found under {base}")

    results = []
    for name, path in model_paths.items():
        print(f"\nEvaluating {name} with RAG @ {path}")
        disc = EncoderDiscriminator(model_name=path, max_length=args.max_length)

        probs_true = predict_probs(disc, texts, batch_size=args.batch_size)
        preds = [1 if p >= 0.5 else 0 for p in probs_true]

        macro_f1 = f1_score(labels, preds, average="macro")
        roc_auc = roc_auc_score(labels, probs_true)
        accuracy = accuracy_score(labels, preds)
        print(f"  Macro-F1: {macro_f1:.4f}")
        print(f"  ROC-AUC:  {roc_auc:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        results.append((name, macro_f1, roc_auc, accuracy))

    print("\n=== RAG Evaluation Summary ===")
    for name, f1, auc, acc in sorted(results, key=lambda item: int(item[0].rsplit("_", 1)[-1])):
        print(
            f"{name:20s}  Macro-F1: {f1:.4f}  ROC-AUC: {auc:.4f}  Accuracy: {acc:.4f}"
        )


if __name__ == "__main__":
    main()
