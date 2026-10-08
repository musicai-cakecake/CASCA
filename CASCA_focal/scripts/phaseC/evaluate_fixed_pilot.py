"""Evaluate only the PAD/event metrics required by the fixed Phase C pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PAD_TOKEN = "<pad>"
MASK_TOKEN = "<MASK>"
UNK_TOKEN = "<UNK>"
GRID_SIZE = 16 * 512


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def decode_mask(rle: list[list[int]]) -> list[bool]:
    mask = [False] * GRID_SIZE
    for start, length in rle:
        if start < 0 or length < 0 or start + length > GRID_SIZE:
            raise ValueError(f"invalid generation-mask run: {(start, length)}")
        mask[start:start + length] = [True] * length
    return mask


def load_records(path: Path) -> dict[int, dict]:
    records: dict[int, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            index = int(record["valid_index"])
            if index in records:
                raise ValueError(f"duplicate valid_index={index} in {path}:{line_number}")
            records[index] = record
    if not records:
        raise ValueError(f"no records in {path}")
    return records


def flatten(grid: list[list[str]]) -> list[str]:
    tokens = [token for row in grid for token in row]
    if len(tokens) != GRID_SIZE:
        raise ValueError(f"expected {GRID_SIZE} grid tokens, got {len(tokens)}")
    return tokens


def summarize(records: dict[int, dict]) -> dict:
    counts = {
        "cells": 0,
        "reference_pad": 0,
        "generated_pad": 0,
        "event_true_positive": 0,
        "event_false_positive": 0,
        "event_false_negative": 0,
        "event_true_negative": 0,
        "exact_token_matches": 0,
        "generated_mask_tokens": 0,
        "generated_unk_tokens": 0,
        "structural_pass_samples": 0,
    }
    for record in records.values():
        reference = flatten(record["ground_truth"])
        generated = flatten(record["gen"])
        mask = decode_mask(record["generation_mask_rle"])
        counts["structural_pass_samples"] += int(
            bool(record.get("structural_audit", {}).get("all_pass", False))
        )
        for selected, target, prediction in zip(mask, reference, generated):
            if not selected:
                continue
            target_pad = target == PAD_TOKEN
            prediction_pad = prediction == PAD_TOKEN
            counts["cells"] += 1
            counts["reference_pad"] += int(target_pad)
            counts["generated_pad"] += int(prediction_pad)
            counts["exact_token_matches"] += int(target == prediction)
            counts["generated_mask_tokens"] += int(prediction == MASK_TOKEN)
            counts["generated_unk_tokens"] += int(prediction == UNK_TOKEN)
            counts["event_true_positive"] += int(not target_pad and not prediction_pad)
            counts["event_false_positive"] += int(target_pad and not prediction_pad)
            counts["event_false_negative"] += int(not target_pad and prediction_pad)
            counts["event_true_negative"] += int(target_pad and prediction_pad)

    cells = counts["cells"]
    reference_event = cells - counts["reference_pad"]
    generated_event = cells - counts["generated_pad"]
    tp = counts["event_true_positive"]
    fp = counts["event_false_positive"]
    fn = counts["event_false_negative"]
    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    f1 = ratio(2 * precision * recall, precision + recall)
    reference_pad_ratio = ratio(counts["reference_pad"], cells)
    generated_pad_ratio = ratio(counts["generated_pad"], cells)
    return {
        "sample_count": len(records),
        **counts,
        "reference_event": reference_event,
        "generated_event": generated_event,
        "reference_pad_ratio": reference_pad_ratio,
        "generated_pad_ratio": generated_pad_ratio,
        "pad_gap_percentage_points": 100 * (generated_pad_ratio - reference_pad_ratio),
        "reference_event_density": ratio(reference_event, cells),
        "generated_event_density": ratio(generated_event, cells),
        "event_precision": precision,
        "event_recall": recall,
        "event_f1": f1,
        "true_event_predicted_as_pad_ratio": ratio(fn, reference_event),
        "exact_token_match_ratio": ratio(counts["exact_token_matches"], cells),
        "structural_pass_rate": ratio(counts["structural_pass_samples"], len(records)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--focal", type=Path, required=True)
    parser.add_argument("--control_checkpoint", type=Path, required=True)
    parser.add_argument("--focal_checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    control_records = load_records(args.control)
    focal_records = load_records(args.focal)
    if set(control_records) != set(focal_records):
        raise ValueError("control and focal valid indices differ")
    for index in control_records:
        control = control_records[index]
        focal = focal_records[index]
        for field in ("sample_seed", "ground_truth", "generation_mask_rle"):
            if control[field] != focal[field]:
                raise ValueError(f"control/focal mismatch at valid_index={index}, field={field}")

    control_metrics = summarize(control_records)
    focal_metrics = summarize(focal_records)
    if control_metrics["reference_pad"] != focal_metrics["reference_pad"]:
        raise ValueError("control and focal reference PAD counts differ")

    control_checkpoint = json.loads(args.control_checkpoint.read_text(encoding="utf-8"))
    focal_checkpoint = json.loads(args.focal_checkpoint.read_text(encoding="utf-8"))
    control_abs_gap = abs(control_metrics["pad_gap_percentage_points"])
    focal_abs_gap = abs(focal_metrics["pad_gap_percentage_points"])
    payload = {
        "schema_version": "bd3lms-music-v2-focal.phaseC-fixed-pilot.v1",
        "protocol": {
            "valid_indices": [min(control_records), max(control_records)],
            "sample_count": len(control_records),
            "seed": 1,
            "diffusion_steps": 1000,
            "nucleus_p": 0.95,
            "weight_source": "ema",
            "optimizer_steps": 125,
            "global_batch_size": 64,
            "learning_rate": 3e-6,
            "warmup_steps": 100,
        },
        "checkpoint_identity": {
            "control": control_checkpoint,
            "focal": focal_checkpoint,
        },
        "control": control_metrics,
        "focal": focal_metrics,
        "comparison": {
            "generated_pad_ratio_delta_percentage_points": 100 * (
                focal_metrics["generated_pad_ratio"]
                - control_metrics["generated_pad_ratio"]
            ),
            "absolute_pad_gap_improvement_percentage_points": control_abs_gap - focal_abs_gap,
            "event_density_delta_percentage_points": 100 * (
                focal_metrics["generated_event_density"]
                - control_metrics["generated_event_density"]
            ),
            "event_precision_delta": focal_metrics["event_precision"]
            - control_metrics["event_precision"],
            "event_recall_delta": focal_metrics["event_recall"]
            - control_metrics["event_recall"],
            "event_f1_delta": focal_metrics["event_f1"]
            - control_metrics["event_f1"],
            "true_event_predicted_as_pad_delta": focal_metrics[
                "true_event_predicted_as_pad_ratio"
            ]
            - control_metrics["true_event_predicted_as_pad_ratio"],
            "primary_pad_target_met": focal_abs_gap < control_abs_gap,
            "sampling_structure_safe": (
                control_metrics["structural_pass_rate"] == 1.0
                and focal_metrics["structural_pass_rate"] == 1.0
                and control_metrics["generated_mask_tokens"] == 0
                and focal_metrics["generated_mask_tokens"] == 0
                and control_metrics["generated_unk_tokens"] == 0
                and focal_metrics["generated_unk_tokens"] == 0
            ),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["comparison"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
