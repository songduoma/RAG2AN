"""
Merge English splits and keep only samples with <=330 tokens using the DeBERTa
tokenizer. Labels are normalized to lowercase `true`/`false`; other labels are
dropped. Output columns: `title`, `text`, `rating`.
"""
from pathlib import Path
from typing import List

import pandas as pd
from transformers import AutoTokenizer

# Use the same tokenizer as the discriminator; stay offline to avoid downloads.
TOKENIZER_MODEL = "microsoft/deberta-v3-base"
MAX_TOKENS = 330

DATASETS: List[Path] = [
    Path("FakeNews_Task3_2022/Task3_train_dev/Task3_english_training.csv"),
    Path("FakeNews_Task3_2022/Task3_train_dev/Task3_english_dev.csv"),
    Path("FakeNews_Task3_2022/Task3_Test/English_data_test_release_with_rating.csv"),
]

OUTPUT_PATH = Path("FakeNews_Task3_2022/eval.csv")


def normalize_label(value: str):
    """Return normalized true/false label or None for unwanted labels."""
    label = str(value).strip().lower()
    if label == "true":
        return "true"
    if label == "false":
        return "false"
    return None


def main():
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_MODEL, use_fast=True, local_files_only=True
    )

    frames = []
    for path in DATASETS:
        df = pd.read_csv(path)
        label_col = "our rating" if "our rating" in df.columns else "rating"

        token_counts = df["text"].astype(str).apply(
            lambda text: len(tokenizer(text).input_ids)
        )
        within_limit = token_counts <= MAX_TOKENS

        filtered = (
            df.loc[within_limit, ["title", "text", label_col]]
            .assign(
                rating=lambda x: x[label_col].map(normalize_label),
                title=lambda x: x["title"].fillna("").astype(str),
            )
            .dropna(subset=["rating"])[["title", "text", "rating"]]
        )
        frames.append(filtered)

    combined = pd.concat(frames, ignore_index=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(combined)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
