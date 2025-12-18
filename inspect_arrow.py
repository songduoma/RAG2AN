#!/usr/bin/env python
import argparse
from pathlib import Path
from pprint import pprint

from datasets import Dataset, load_from_disk


def load_dataset(target: Path):
    """Load either a dataset directory or a standalone Arrow file."""
    if target.is_dir():
        return load_from_disk(str(target))
    if target.suffix == ".arrow":
        return Dataset.from_file(str(target))
    raise ValueError(
        f"Unsupported path {target}. Provide a dataset directory or an .arrow file."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Print the first N rows stored in a HuggingFace Arrow dataset."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="local/rag_gan_runs/gen_on_dis_on/round_1/data-00000-of-00001.arrow",
        help="Path to the dataset directory or to a data-*.arrow file.",
    )
    parser.add_argument(
        "-n",
        "--num-records",
        type=int,
        default=5,
        help="How many rows to show (default: 10).",
    )
    args = parser.parse_args()

    dataset_path = Path(args.path).expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"{dataset_path} does not exist.")

    ds = load_dataset(dataset_path)
    num_to_show = min(args.num_records, len(ds))

    for idx in range(num_to_show):
        print(f"--- Row {idx} ---")
        row = dict(ds[idx])
        row.pop("generated_ids", None)
        row.pop("prompt_ids", None)
        pprint(row)
        print("\n\n")


if __name__ == "__main__":
    main()
