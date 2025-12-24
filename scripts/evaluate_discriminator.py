#!/usr/bin/env python
"""
Evaluate a trained discriminator on the advfake dataset.

The script pulls 402 real descriptions and their paired 402 fake rewrites
(`f_description`) from sanxing/advfake, runs the discriminator, and reports
Macro-F1 and ROC-AUC.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.discriminator import (
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


def load_advfake_pairs(
    dataset_path: Path,
    disc_use_rag: bool,
    rag_source: str,
    num_rag_results: int,
) -> Tuple[List[str], List[int]]:
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
            texts.append(
                format_discriminator_input(
                    row,
                    rag=disc_use_rag,
                    prefix="",
                    source=rag_source,
                    num_results=num_rag_results,
                )
            )
            labels.append(1)
        if fake:
            texts.append(
                format_discriminator_input(
                    row,
                    rag=disc_use_rag,
                    prefix="",
                    description_override=fake,
                    source=rag_source,
                    num_results=num_rag_results,
                )
            )
            labels.append(0)

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
    env_disc_use_rag = os.environ.get("DISC_USE_RAG", "0")
    default_disc_use_rag = str(env_disc_use_rag).strip() == "1"
    default_rag_source = os.environ.get("RAG_SOURCE", "dpr")
    default_num_rag_results = int(os.environ.get("NUM_RAG_RESULTS", "3"))
    parser = argparse.ArgumentParser(
        description="Evaluate discriminator checkpoint(s) on advfake."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path or model id for a single discriminator checkpoint.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("local/rag_gan_runs/20251223_132301"),
        help="Directory containing per-round discriminator folders (disc_round_*).",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("local/hf_datasets/advfake/advfake-train.arrow"),
        help="Path to advfake arrow file (contains 402 rows).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--disc-use-rag",
        action="store_true",
        default=default_disc_use_rag,
        help="Match training distribution by including RAG context.",
    )
    parser.add_argument(
        "--no-disc-rag",
        action="store_false",
        dest="disc_use_rag",
        help="Disable RAG context (still applies discriminator prompt prefix).",
    )
    parser.add_argument(
        "--rag-source",
        type=str,
        default=default_rag_source,
        choices=["dpr", "none"],
        help="Retrieval source used for building the context.",
    )
    parser.add_argument(
        "--num-rag-results",
        type=int,
        default=default_num_rag_results,
        help="Number of retrieved documents to include.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=True,
        help="Load the discriminator with bfloat16 weights.",
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

    print(f"Loading advfake pairs from {args.dataset_path} ...")
    texts, labels = load_advfake_pairs(
        args.dataset_path,
        disc_use_rag=args.disc_use_rag,
        rag_source=args.rag_source,
        num_rag_results=args.num_rag_results,
    )
    n_real = sum(labels)
    n_fake = len(labels) - n_real
    print(f"Loaded {len(labels)} samples (real={n_real}, fake={n_fake}).")

    model_paths: Dict[str, str] = {}
    if args.model_path:
        model_paths[Path(args.model_path).name] = args.model_path
    else:
        base = Path(args.models_dir)
        if not base.exists():
            raise FileNotFoundError(f"Models dir not found: {base}")
        # collect disc_round_* folders
        for sub in sorted(base.iterdir()):
            if sub.is_dir() and sub.name.startswith("disc_round_"):
                model_paths[sub.name] = str(sub)
        if not model_paths:
            raise FileNotFoundError(
                f"No disc_round_* folders found under {base}. "
                "Provide --model-path for single evaluation."
            )

    results = []
    torch_dtype = torch.bfloat16 if args.bf16 else None
    for name, path in model_paths.items():
        print(f"\nEvaluating {name} @ {path}")
        disc = EncoderDiscriminator(
            model_name=path,
            max_length=args.max_length,
            torch_dtype=torch_dtype,
        )

        probs_true = predict_probs(disc, texts, batch_size=args.batch_size)
        preds = [1 if p >= 0.5 else 0 for p in probs_true]

        macro_f1 = f1_score(labels, preds, average="macro")
        roc_auc = roc_auc_score(labels, probs_true)
        accuracy = accuracy_score(labels, preds)

        print(f"  Macro-F1: {macro_f1:.4f}")
        print(f"  ROC-AUC:  {roc_auc:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        results.append((name, macro_f1, roc_auc, accuracy))

    if len(results) > 1:
        print("\n=== Summary ===")
        for name, f1, auc, acc in sorted(results, key=lambda item: int(item[0].rsplit("_", 1)[-1])):
            print(
                f"{name:20s}  Macro-F1: {f1:.4f}  ROC-AUC: {auc:.4f}  Accuracy: {acc:.4f}"
            )


if __name__ == "__main__":
    main()
