"""Full-validation PAD/event evaluation for the Focal V2 generation region."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from xmidi_getmusic.constants import ROW_NAMES  # noqa: E402
from xmidi_getmusic.eval_protocol import (  # noqa: E402
    INFERENCE_SCHEMA_VERSION,
    iter_selected_token_pairs,
    validate_manifest_provenance,
)

PAD = "<pad>"
UNK = "<UNK>"
MASK = "<MASK>"


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _empty_counts() -> dict[str, int]:
    return {
        "cells": 0,
        "reference_pad": 0,
        "generated_pad": 0,
        "event_true_positive": 0,
        "event_false_positive": 0,
        "event_false_negative": 0,
        "event_true_negative": 0,
        "exact_token_matches": 0,
        "generated_mask": 0,
        "generated_unk": 0,
    }


def _update(counts: dict[str, int], target: str, prediction: str) -> None:
    target_pad = target == PAD
    prediction_pad = prediction == PAD
    counts["cells"] += 1
    counts["reference_pad"] += int(target_pad)
    counts["generated_pad"] += int(prediction_pad)
    counts["event_true_positive"] += int(not target_pad and not prediction_pad)
    counts["event_false_positive"] += int(target_pad and not prediction_pad)
    counts["event_false_negative"] += int(not target_pad and prediction_pad)
    counts["event_true_negative"] += int(target_pad and prediction_pad)
    counts["exact_token_matches"] += int(target == prediction)
    counts["generated_mask"] += int(prediction == MASK)
    counts["generated_unk"] += int(prediction == UNK)


def _metrics(counts: dict[str, int]) -> dict[str, Any]:
    cells = counts["cells"]
    reference_events = cells - counts["reference_pad"]
    generated_events = cells - counts["generated_pad"]
    tp = counts["event_true_positive"]
    fp = counts["event_false_positive"]
    fn = counts["event_false_negative"]
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    return {
        **counts,
        "reference_events": reference_events,
        "generated_events": generated_events,
        "reference_pad_ratio": _ratio(counts["reference_pad"], cells),
        "generated_pad_ratio": _ratio(counts["generated_pad"], cells),
        "pad_gap_percentage_points": 100.0
        * _ratio(counts["generated_pad"] - counts["reference_pad"], cells),
        "reference_event_density": _ratio(reference_events, cells),
        "generated_event_density": _ratio(generated_events, cells),
        "event_precision": precision,
        "event_recall": recall,
        "event_f1": _ratio(2.0 * precision * recall, precision + recall),
        "true_event_predicted_as_pad_ratio": _ratio(fn, reference_events),
        "exact_token_match_ratio": _ratio(counts["exact_token_matches"], cells),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--out_json", type=Path, required=True)
    parser.add_argument("--out_csv", type=Path, required=True)
    parser.add_argument("--expect_valid_size", type=int, default=6817)
    args = parser.parse_args()

    jsonl = args.jsonl.resolve()
    manifest_path = Path(str(jsonl) + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest_provenance(manifest)
    if manifest.get("status") != "completed":
        raise RuntimeError(f"inference manifest status is {manifest.get('status')!r}")
    provenance_id = manifest["provenance_id"]

    total = _empty_counts()
    per_row: dict[int, dict[str, int]] = defaultdict(_empty_counts)
    seen: set[int] = set()
    structural_pass = 0
    with jsonl.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("schema_version") != INFERENCE_SCHEMA_VERSION:
                raise RuntimeError(f"invalid schema at line {line_number}")
            if record.get("provenance_id") != provenance_id:
                raise RuntimeError(f"provenance mismatch at line {line_number}")
            valid_index = int(record["valid_index"])
            if valid_index in seen:
                raise RuntimeError(f"duplicate valid_index={valid_index}")
            seen.add(valid_index)
            audit = record.get("structural_audit", {})
            structural_pass += int(bool(audit.get("all_pass", False)))
            if not audit.get("all_pass", False):
                raise RuntimeError(f"structural audit failed at valid_index={valid_index}")
            for row, _, target, prediction in iter_selected_token_pairs(record):
                _update(total, target, prediction)
                _update(per_row[row], target, prediction)

    expected = set(range(args.expect_valid_size))
    if seen != expected:
        raise RuntimeError(
            f"expected valid indices 0..{args.expect_valid_size - 1}; "
            f"observed {len(seen)} unique indices"
        )
    if total["generated_mask"] or total["generated_unk"]:
        raise RuntimeError(
            "generated special tokens remain in G: "
            f"MASK={total['generated_mask']}, UNK={total['generated_unk']}"
        )

    payload = {
        "schema_version": "bd3lms-music-v2-focal.pad-event-evaluation.v1",
        "protocol": {
            "input_jsonl": str(jsonl),
            "manifest": str(manifest_path),
            "provenance_id": provenance_id,
            "checkpoint": manifest.get("checkpoint"),
            "sample_count": len(seen),
            "region": "explicit per-record generation_mask_rle (G=1)",
            "event_definition": "token != <pad>; generated <MASK>/<UNK> must be zero",
            "structural_pass_rate": _ratio(structural_pass, len(seen)),
        },
        "aggregate": _metrics(total),
        "per_target_row": {
            str(row): {"row_name": ROW_NAMES[row], **_metrics(per_row[row])}
            for row in sorted(per_row)
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["row", "row_name", *_metrics(_empty_counts()).keys()]
    with args.out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({"row": "aggregate", "row_name": "all_target_rows", **payload["aggregate"]})
        for row, values in payload["per_target_row"].items():
            writer.writerow({"row": row, **values})
    print(json.dumps(payload["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
