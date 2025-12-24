#!/usr/bin/env python
"""
Simple attention/token-salience listing for EncoderDiscriminator.

Prints per-token [CLS] attention scores to stdout.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.discriminator import DEFAULT_ENCODER_MODEL, EncoderDiscriminator


def _read_text(args: argparse.Namespace) -> str:
    if args.text:
        return args.text.strip()
    if args.text_file:
        return Path(args.text_file).read_text(encoding="utf-8").strip()
    raise ValueError("Provide --text or --text-file.")


def _normalize_scores(scores: List[float]) -> List[float]:
    if not scores:
        return []
    max_score = max(scores)
    if max_score <= 0:
        return [0.0 for _ in scores]
    return [s / max_score for s in scores]


def _build_token_spans(
    disc: EncoderDiscriminator,
    text: str,
) -> List[Dict]:
    tokenizer = disc.tokenizer
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=disc.max_length,
        return_tensors="pt",
        return_offsets_mapping=True,
    ).to(disc.device)

    inputs_for_model = {k: v for k, v in inputs.items() if k != "offset_mapping"}
    outputs = disc.model(
        **inputs_for_model, output_attentions=True, return_dict=True
    )
    attentions = outputs.attentions[-1].cpu()
    avg_att = torch.mean(attentions, dim=1).squeeze(0)  # (seq, seq)
    cls_att = avg_att[0, :]

    input_ids = inputs["input_ids"][0].cpu().tolist()
    offsets = inputs["offset_mapping"][0].cpu().tolist()

    special_mask = [
        tok_id in (tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id)
        for tok_id in input_ids
    ]
    noise_mask = []
    for tok_id in input_ids:
        word = tokenizer.decode([tok_id])
        noise_mask.append(disc._is_noise_token(word))
    combined_mask = [s or n for s, n in zip(special_mask, noise_mask)]
    cls_att[combined_mask] = 0

    token_spans: List[Dict] = []
    for idx, (start, end) in enumerate(offsets):
        if start == end:
            continue
        token_spans.append(
            {
                "start": start,
                "end": end,
                "score": float(cls_att[idx].item()),
                "is_top": False,
            }
        )

    token_spans.sort(key=lambda item: item["start"])
    return token_spans


def _build_segments(text: str, token_spans: List[Dict]) -> List[Dict]:
    segments: List[Dict] = []
    cursor = 0
    for span in token_spans:
        start = span["start"]
        end = span["end"]
        if cursor < start:
            segments.append(
                {
                    "text": text[cursor:start],
                    "score": 0.0,
                    "is_top": False,
                }
            )
        segments.append(
            {
                "text": text[start:end],
                "score": span["score"],
                "is_top": span["is_top"],
            }
        )
        cursor = end
    if cursor < len(text):
        segments.append({"text": text[cursor:], "score": 0.0, "is_top": False})
    return segments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print per-token [CLS] attention scores."
    )
    parser.add_argument("--text", type=str, default=None, help="Input text string.")
    parser.add_argument(
        "--text-file", type=str, default=None, help="Path to a text file."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="local/rag_gan_runs/gen_on_dis_on/disc_round_5",
        help="Model name or path for discriminator encoder.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Max sequence length (defaults to discriminator setting).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = _read_text(args)

    disc = EncoderDiscriminator(model_name=args.model)
    if args.max_length:
        disc.max_length = args.max_length

    token_spans = _build_token_spans(disc, text)
    segments = _build_segments(text, token_spans)
    scores = [seg["score"] for seg in segments]
    norm_scores = _normalize_scores(scores)

    ranked = sorted(
        zip(segments, norm_scores), key=lambda item: item[1], reverse=True
    )

    print("token\tnorm_attention")
    for seg, norm in ranked:
        token_text = seg["text"].replace("\t", " ").replace("\n", "\\n")
        print(f"{token_text}\t{norm:.6f}")



if __name__ == "__main__":
    main()
