#!/usr/bin/env python3
"""
Download CNN/DailyMail, clean newlines and extra whitespace, and save to disk.

Usage examples:
  # Clean train split and save locally
  python scripts/prepare_cnn_dailymail.py --split train \
      --output-dir local/hf_datasets/cnn_dailymail_3.0.0_clean/train

  # Clean all splits and save as one DatasetDict
  python scripts/prepare_cnn_dailymail.py --split all \
      --output-dir local/hf_datasets/cnn_dailymail_3.0.0_clean

  # Then run training using the local dataset
  DATASET_NAME=cnn_dailymail DATASET_CONFIG=3.0.0 \
  DATASET_SPLIT=train DATASET_PATH=local/hf_datasets/cnn_dailymail_3.0.0_clean/train \
  ./train.sh
"""

import argparse
import re
from pathlib import Path

from datasets import load_dataset, DatasetDict


def clean_text(s: str) -> str:
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    # Replace common breaks/tabs with spaces
    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    # Normalize non-breaking spaces and collapse whitespace
    s = s.replace("\u00A0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def clean_cnn_dm_split(ds):
    """Clean a single split of CNN/DailyMail dataset."""
    def _clean(example):
        article = clean_text(example.get("article", ""))
        highlights = clean_text(example.get("highlights", ""))
        example["article"] = article
        example["highlights"] = highlights
        return example

    return ds.map(_clean, desc="Cleaning CNN/DailyMail (strip newlines/whitespace)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare cleaned CNN/DailyMail dataset (remove newlines, collapse spaces).")
    p.add_argument("--config", default="3.0.0", help="Dataset config/version (default: 3.0.0)")
    p.add_argument("--split", default="train", help="Split to process: train | validation | test | all")
    p.add_argument("--max-samples", type=int, default=None, help="Optionally limit number of samples")
    p.add_argument("--output-dir", required=True, help="Directory to save cleaned dataset")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.split == "all":
        print("Loading all splits of cnn_dailymail...")
        dsd: DatasetDict = load_dataset("cnn_dailymail", args.config)
        cleaned = DatasetDict()
        for sp in ("train", "validation", "test"):
            ds_sp = dsd[sp]
            if args.max_samples:
                ds_sp = ds_sp.select(range(min(args.max_samples, len(ds_sp))))
            cleaned[sp] = clean_cnn_dm_split(ds_sp)
        print(f"Saving cleaned DatasetDict to: {out}")
        cleaned.save_to_disk(str(out))
    else:
        print(f"Loading split: {args.split}")
        ds = load_dataset("cnn_dailymail", args.config, split=args.split)
        if args.max_samples:
            ds = ds.select(range(min(args.max_samples, len(ds))))
        ds = clean_cnn_dm_split(ds)
        print(f"Saving cleaned dataset to: {out}")
        ds.save_to_disk(str(out))

    print("Done.")


if __name__ == "__main__":
    main()

