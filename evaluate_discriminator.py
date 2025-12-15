#!/usr/bin/env python
"""
Evaluate a trained discriminator on the advfake dataset.

The script pulls 402 real descriptions and their paired 402 fake rewrites
(`f_description`) from sanxing/advfake, runs the discriminator, and reports
Macro-F1 and ROC-AUC.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from discriminator import EncoderDiscriminator, MAX_ENCODER_SEQ_LEN


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
        default=Path("local/rag_gan_runs/gen_on_dis_off"),
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
    texts, labels = load_advfake_pairs(args.dataset_path)
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
    for name, path in model_paths.items():
        print(f"\nEvaluating {name} @ {path}")
        disc = EncoderDiscriminator(
            model_name=path,
            max_length=args.max_length,
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
    
    
"""
G+ / D−
=== Summary ===
disc_round_1          Macro-F1: 0.3774  ROC-AUC: 0.7731  Accuracy: 0.5187
disc_round_2          Macro-F1: 0.7015  ROC-AUC: 0.7796  Accuracy: 0.7139
disc_round_3          Macro-F1: 0.6962  ROC-AUC: 0.7809  Accuracy: 0.6965
disc_round_4          Macro-F1: 0.3848  ROC-AUC: 0.7087  Accuracy: 0.5187
disc_round_5          Macro-F1: 0.3979  ROC-AUC: 0.7157  Accuracy: 0.5261
disc_round_6          Macro-F1: 0.7199  ROC-AUC: 0.7980  Accuracy: 0.7201
disc_round_7          Macro-F1: 0.7023  ROC-AUC: 0.7805  Accuracy: 0.7027
disc_round_8          Macro-F1: 0.6732  ROC-AUC: 0.7837  Accuracy: 0.6928
disc_round_9          Macro-F1: 0.6954  ROC-AUC: 0.7667  Accuracy: 0.6965
disc_round_10         Macro-F1: 0.7275  ROC-AUC: 0.7946  Accuracy: 0.7289

G− / D−
=== Summary ===
disc_round_1          Macro-F1: 0.6741  ROC-AUC: 0.7454  Accuracy: 0.6766
disc_round_2          Macro-F1: 0.5576  ROC-AUC: 0.7722  Accuracy: 0.6007
disc_round_3          Macro-F1: 0.5486  ROC-AUC: 0.7673  Accuracy: 0.5933
disc_round_4          Macro-F1: 0.6424  ROC-AUC: 0.7104  Accuracy: 0.6443
disc_round_5          Macro-F1: 0.5539  ROC-AUC: 0.7574  Accuracy: 0.5970
disc_round_6          Macro-F1: 0.7078  ROC-AUC: 0.7754  Accuracy: 0.7114
disc_round_7          Macro-F1: 0.6678  ROC-AUC: 0.7633  Accuracy: 0.6704
disc_round_8          Macro-F1: 0.6873  ROC-AUC: 0.7618  Accuracy: 0.6978
disc_round_9          Macro-F1: 0.6678  ROC-AUC: 0.7813  Accuracy: 0.6716
disc_round_10         Macro-F1: 0.6756  ROC-AUC: 0.7825  Accuracy: 0.6803
"""