#!/usr/bin/env python3
"""Compute legacy-compatible per-event KL from GT/GEN token counts."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

EVENTS = (
    (4, "piano_pitch", "piano*", "piano"),
    (5, "piano_duration", "Duration_piano_*", "Duration_piano_"),
    (6, "guitar_pitch", "guitar*", "guitar"),
    (7, "guitar_duration", "Duration_guitar_*", "Duration_guitar_"),
    (8, "bass_pitch", "bass*", "bass"),
    (9, "bass_duration", "Duration_bass_*", "Duration_bass_"),
    (10, "string_pitch", "string*", "string"),
    (11, "string_duration", "Duration_string_*", "Duration_string_"),
    (12, "wind_pitch", "wind*", "wind"),
    (13, "wind_duration", "Duration_wind_*", "Duration_wind_"),
    (14, "drum_pitch", "drum*", "drum"),
    (15, "drum_duration", "Duration_drum_*", "Duration_drum_"),
)
SPECIALS = {"<pad>", "<UNK>", "<MASK>"}
FIELDS = (
    "row_index_0_based",
    "event_type",
    "token_rule",
    "union_vocabulary_size",
    "gt_non_special_count",
    "gen_non_special_count",
    "gt_positive_gen_zero_tokens",
    "strict_kl_gt_to_gen_nats",
    "laplace_add_one_kl_gt_to_gen_nats",
    "laplace_add_one_kl_gt_to_gen_bits",
)


def read_counts(path: Path) -> dict[str, int]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        counts = {row["token"]: int(row["count"]) for row in rows}
    if not counts:
        raise ValueError(f"no token counts in {path}")
    return counts


def calculate(gt: dict[str, int], gen: dict[str, int]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    all_tokens = set(gt) | set(gen)
    for row_index, event_type, token_rule, prefix in EVENTS:
        tokens = sorted(
            token
            for token in all_tokens
            if token not in SPECIALS and token.startswith(prefix)
        )
        if not tokens:
            raise ValueError(f"empty event vocabulary for {event_type}")
        gt_total = sum(gt.get(token, 0) for token in tokens)
        gen_total = sum(gen.get(token, 0) for token in tokens)
        if gt_total == 0 or gen_total == 0:
            raise ValueError(f"zero event total for {event_type}")
        missing = sum(
            gt.get(token, 0) > 0 and gen.get(token, 0) == 0
            for token in tokens
        )
        if missing:
            strict = math.inf
        else:
            strict = sum(
                (gt[token] / gt_total)
                * math.log((gt[token] / gt_total) / (gen[token] / gen_total))
                for token in tokens
                if gt.get(token, 0) > 0
            )
        vocabulary_size = len(tokens)
        smooth = sum(
            ((gt.get(token, 0) + 1) / (gt_total + vocabulary_size))
            * math.log(
                ((gt.get(token, 0) + 1) / (gt_total + vocabulary_size))
                / ((gen.get(token, 0) + 1) / (gen_total + vocabulary_size))
            )
            for token in tokens
        )
        rows.append(
            {
                "row_index_0_based": row_index,
                "event_type": event_type,
                "token_rule": token_rule,
                "union_vocabulary_size": vocabulary_size,
                "gt_non_special_count": gt_total,
                "gen_non_special_count": gen_total,
                "gt_positive_gen_zero_tokens": missing,
                "strict_kl_gt_to_gen_nats": strict,
                "laplace_add_one_kl_gt_to_gen_nats": smooth,
                "laplace_add_one_kl_gt_to_gen_bits": smooth / math.log(2.0),
            }
        )
    return rows


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            formatted = dict(row)
            for field in (
                "strict_kl_gt_to_gen_nats",
                "laplace_add_one_kl_gt_to_gen_nats",
                "laplace_add_one_kl_gt_to_gen_bits",
            ):
                value = float(formatted[field])
                formatted[field] = "inf" if math.isinf(value) else f"{value:.12f}"
            writer.writerow(formatted)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_csv", required=True)
    parser.add_argument("--gen_csv", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = calculate(read_counts(Path(args.gt_csv)), read_counts(Path(args.gen_csv)))
    write_rows(Path(args.out), rows)
    for row in rows:
        strict = row["strict_kl_gt_to_gen_nats"]
        strict_text = "inf" if math.isinf(float(strict)) else f"{float(strict):.6f}"
        print(
            f"{row['row_index_0_based']:>2} {row['event_type']:<18} "
            f"strict={strict_text:<10} "
            f"add1={float(row['laplace_add_one_kl_gt_to_gen_nats']):.6f} nats"
        )


if __name__ == "__main__":
    main()
