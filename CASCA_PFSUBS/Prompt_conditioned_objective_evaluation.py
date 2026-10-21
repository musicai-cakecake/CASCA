#!/usr/bin/env python3
"""Prompt-conditioned D_P, D_D, and D_IOI evaluation for BD3LMs.

The script evaluates the V2 and final Focal SUBS full-validation inference runs.
For every validation record it:

1. strictly extracts global ``style``, ``mood``, and ``tempo`` values from all
   active target row prompts and verifies row-to-row agreement;
2. computes piece-level D_P, D_D, D_IOI over all selected target cells;
3. computes instrument-level D_P, D_D, D_IOI on each active pitch/duration row
   pair; and
4. groups scores by instrument, style, mood, and tempo.

Within each model and prompt item, the highest floor(0.95*N) samples are retained
independently for D_P, D_D, and D_IOI. Their intersection is the reported 95%
confidence subset, so one poor metric is sufficient to exclude a sample.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import Objective_evaluation_for_musical_pieces as base


INSTRUMENT_SPECS = (
    ("piano", 4, 5),
    ("bass", 8, 9),
    ("wind", 12, 13),
    ("drum", 14, 15),
    ("guitar", 6, 7),
    ("string", 10, 11),
)
REQUESTED_STYLES = ("classical", "country", "jazz", "pop", "rock", "traditional")
REQUESTED_MOODS = ("angry", "happy", "quiet", "sad")
CATEGORIES = ("instrument", "style", "mood", "tempo")
METRICS = ("D_P", "D_D", "D_IOI")
OUTPUT_DIR_DEFAULT = base.SAFE_OUTPUT_ROOT / "prompt_conditioned_objective_evaluation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute instrument/style/mood/tempo conditioned D_P, D_D, D_IOI "
            "for V2 and final Focal SUBS full-validation inference."
        )
    )
    parser.add_argument("--v2-jsonl", type=Path, default=base.V2_JSONL_DEFAULT)
    parser.add_argument("--focal-jsonl", type=Path, default=base.FOCAL_JSONL_DEFAULT)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR_DEFAULT)
    parser.add_argument("--expected-records", type=int, default=6817)
    parser.add_argument("--grid-step", type=float, default=0.25)
    parser.add_argument("--min-bandwidth", type=float, default=0.5)
    parser.add_argument("--confidence-retain-fraction", type=float, default=0.95)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.expected_records <= 0:
        parser.error("--expected-records must be positive")
    if args.grid_step <= 0 or args.min_bandwidth <= 0:
        parser.error("--grid-step and --min-bandwidth must be positive")
    if not 0 < args.confidence_retain_fraction <= 1:
        parser.error("--confidence-retain-fraction must be in (0, 1]")
    if args.max_records is not None and args.max_records <= 0:
        parser.error("--max-records must be positive")
    return args


def extract_prompt_value(prompt: Sequence[str], key: str) -> str:
    positions = [index for index, token in enumerate(prompt) if token == key]
    if len(positions) != 1:
        raise RuntimeError(f"prompt must contain exactly one {key!r}, got {prompt}")
    index = positions[0]
    if index + 1 >= len(prompt):
        raise RuntimeError(f"prompt key {key!r} has no value: {prompt}")
    value = str(prompt[index + 1])
    if value in base.SPECIAL_TOKENS or not value:
        raise RuntimeError(f"invalid prompt value after {key!r}: {value!r}")
    return value


def prompt_metadata(
    record: Mapping[str, Any],
) -> tuple[str, str, str, tuple[str, ...]]:
    gt = record["ground_truth"]
    gen = record["gen"]
    row_unk = record.get("row_unk_mask")
    if not isinstance(row_unk, list) or len(row_unk) != base.NUM_ROWS:
        raise RuntimeError("row_unk_mask must contain 16 entries")

    style_values: set[str] = set()
    mood_values: set[str] = set()
    tempo_values: set[str] = set()
    active: list[str] = []
    for instrument, pitch_row, duration_row in INSTRUMENT_SPECS:
        pitch_all_unk = all(token == "<UNK>" for token in gt[pitch_row])
        duration_all_unk = all(token == "<UNK>" for token in gt[duration_row])
        if pitch_all_unk != bool(row_unk[pitch_row]):
            raise RuntimeError(f"row_unk_mask mismatch on row {pitch_row}")
        if duration_all_unk != bool(row_unk[duration_row]):
            raise RuntimeError(f"row_unk_mask mismatch on row {duration_row}")
        if pitch_all_unk != duration_all_unk:
            raise RuntimeError(f"active/inactive row-pair mismatch for {instrument}")
        for row in (pitch_row, duration_row):
            if gt[row][: base.NUM_PROMPT_COLS] != gen[row][: base.NUM_PROMPT_COLS]:
                raise RuntimeError(f"generated prompt differs from GT on row {row}")
        if pitch_all_unk:
            continue

        active.append(instrument)
        expected_tail = [instrument] * 8
        for row in (pitch_row, duration_row):
            prompt = gt[row][: base.NUM_PROMPT_COLS]
            if prompt[8:16] != expected_tail:
                raise RuntimeError(
                    f"instrument prompt mismatch for {instrument} row {row}: {prompt}"
                )
            style_values.add(extract_prompt_value(prompt, "style"))
            mood_values.add(extract_prompt_value(prompt, "mood"))
            tempo_values.add(extract_prompt_value(prompt, "tempo"))

    if not active:
        raise RuntimeError("sample contains no active target instrument")
    if len(style_values) != 1 or len(mood_values) != 1 or len(tempo_values) != 1:
        raise RuntimeError(
            "inconsistent prompt metadata: "
            f"style={style_values}, mood={mood_values}, tempo={tempo_values}"
        )
    return (
        next(iter(style_values)),
        next(iter(mood_values)),
        next(iter(tempo_values)),
        tuple(active),
    )


def score_grid(
    grid_gt: Sequence[Sequence[str]],
    grid_gen: Sequence[Sequence[str]],
    mask: Any,
    grid_step: float,
    min_bandwidth: float,
) -> dict[str, float]:
    gt_features = base.extract_features(grid_gt, mask)
    gen_features = base.extract_features(grid_gen, mask)
    scores: dict[str, float] = {}
    for feature, metric in (("pitch", "D_P"), ("duration", "D_D"), ("ioi", "D_IOI")):
        score = base.score_feature(
            base.feature_values(gt_features, feature),
            base.feature_values(gen_features, feature),
            grid_step,
            min_bandwidth,
        )
        if score.overlap is None:
            raise RuntimeError(f"undefined {metric} after explicit empty-set handling")
        scores[metric] = float(score.overlap)
    return scores


def confidence_summary(
    rows: Sequence[Mapping[str, Any]], retain_fraction: float
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    if not rows:
        empty_metrics = {
            metric: {"n": 0, "mean": None, "std": None, "median": None, "min": None, "max": None}
            for metric in METRICS
        }
        return (
            {
                "total_samples": 0,
                "retained_per_metric": 0,
                "joint_retained_samples": 0,
                "joint_excluded_samples": 0,
                "cutoff_minimum_retained_score": {metric: None for metric in METRICS},
                "metrics": empty_metrics,
            },
            {},
        )

    retain_count = max(1, math.floor(len(rows) * retain_fraction))
    retained_by_metric: dict[str, set[int]] = {}
    ranks: dict[str, dict[int, int]] = {}
    cutoffs: dict[str, float] = {}
    for metric in METRICS:
        ranked = sorted(rows, key=lambda row: (-float(row[metric]), int(row["valid_index"])))
        retained_by_metric[metric] = {
            int(row["valid_index"]) for row in ranked[:retain_count]
        }
        ranks[metric] = {
            int(row["valid_index"]): rank
            for rank, row in enumerate(ranked, start=1)
        }
        cutoffs[metric] = float(ranked[retain_count - 1][metric])

    joint = set.intersection(*(retained_by_metric[metric] for metric in METRICS))
    retained_rows = [row for row in rows if int(row["valid_index"]) in joint]
    details: dict[int, dict[str, Any]] = {}
    for row in rows:
        valid_index = int(row["valid_index"])
        details[valid_index] = {
            "rank_D_P": ranks["D_P"][valid_index],
            "rank_D_D": ranks["D_D"][valid_index],
            "rank_D_IOI": ranks["D_IOI"][valid_index],
            "top95_D_P": valid_index in retained_by_metric["D_P"],
            "top95_D_D": valid_index in retained_by_metric["D_D"],
            "top95_D_IOI": valid_index in retained_by_metric["D_IOI"],
            "confidence95_retained": valid_index in joint,
        }
    return (
        {
            "total_samples": len(rows),
            "retain_fraction_per_metric": retain_fraction,
            "retained_per_metric": retain_count,
            "joint_retained_samples": len(retained_rows),
            "joint_excluded_samples": len(rows) - len(retained_rows),
            "cutoff_minimum_retained_score": cutoffs,
            "metrics": {
                metric: base.summary_statistics(
                    [float(row[metric]) for row in retained_rows]
                )
                for metric in METRICS
            },
        },
        details,
    )


def requested_items(observed: Mapping[str, set[str]]) -> dict[str, list[str]]:
    result = {
        "instrument": [item[0] for item in INSTRUMENT_SPECS],
        "style": list(REQUESTED_STYLES),
        "mood": list(REQUESTED_MOODS),
        "tempo": sorted(observed["tempo"]),
    }
    for category in ("style", "mood"):
        extras = sorted(observed[category] - set(result[category]))
        result[category].extend(extras)
    return result


def process_run(
    label: str,
    jsonl: Path,
    out_root: Path,
    expected_records: int,
    grid_step: float,
    min_bandwidth: float,
    retain_fraction: float,
    max_records: int | None,
    progress_every: int,
    expected_references: Mapping[int, str] | None = None,
) -> tuple[dict[str, Any], dict[int, str]]:
    jsonl = jsonl.resolve()
    manifest = base.load_manifest(jsonl)
    if max_records is None and int(manifest.get("records_written", -1)) != expected_records:
        raise RuntimeError(f"{label}: manifest record count mismatch")

    run_dir = out_root / label
    run_dir.mkdir(parents=True, exist_ok=True)
    per_sample_path = run_dir / "per_sample_prompt_metrics.csv"
    selection_path = run_dir / "prompt_group_confidence95_membership.csv"
    references: dict[int, str] = {}
    seen: set[int] = set()
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    observed: dict[str, set[str]] = {category: set() for category in CATEGORIES}
    sample_rows: list[dict[str, Any]] = []

    fieldnames = [
        "model",
        "valid_index",
        "prompt_category",
        "prompt_item",
        "style",
        "mood",
        "tempo",
        "active_instruments",
        "D_P",
        "D_D",
        "D_IOI",
    ]
    with per_sample_path.open("w", encoding="utf-8", newline="") as output_handle:
        writer = csv.DictWriter(output_handle, fieldnames=fieldnames)
        writer.writeheader()
        with jsonl.open(encoding="utf-8") as input_handle:
            for line_number, line in enumerate(input_handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                valid_index, mask = base.validate_record(record, manifest, line_number)
                if valid_index in seen:
                    raise RuntimeError(f"{label}: duplicate valid_index={valid_index}")
                seen.add(valid_index)
                signature = base.reference_signature(record)
                references[valid_index] = signature
                if expected_references is not None:
                    if expected_references.get(valid_index) != signature:
                        raise RuntimeError(
                            f"{label}: GT/mask/seed mismatch at valid_index={valid_index}"
                        )

                style, mood, tempo, active_instruments = prompt_metadata(record)
                observed["style"].add(style)
                observed["mood"].add(mood)
                observed["tempo"].add(tempo)
                active_text = "|".join(active_instruments)
                overall_scores = score_grid(
                    record["ground_truth"],
                    record["gen"],
                    mask,
                    grid_step,
                    min_bandwidth,
                )
                for category, item in (
                    ("style", style),
                    ("mood", mood),
                    ("tempo", tempo),
                ):
                    row = {
                        "model": label,
                        "valid_index": valid_index,
                        "prompt_category": category,
                        "prompt_item": item,
                        "style": style,
                        "mood": mood,
                        "tempo": tempo,
                        "active_instruments": active_text,
                        **overall_scores,
                    }
                    buckets[(category, item)].append(row)
                    sample_rows.append(row)
                    writer.writerow(row)

                for instrument, pitch_row, duration_row in INSTRUMENT_SPECS:
                    if instrument not in active_instruments:
                        continue
                    instrument_mask = mask.copy()
                    instrument_mask[:] = False
                    instrument_mask[pitch_row] = mask[pitch_row]
                    instrument_mask[duration_row] = mask[duration_row]
                    instrument_scores = score_grid(
                        record["ground_truth"],
                        record["gen"],
                        instrument_mask,
                        grid_step,
                        min_bandwidth,
                    )
                    row = {
                        "model": label,
                        "valid_index": valid_index,
                        "prompt_category": "instrument",
                        "prompt_item": instrument,
                        "style": style,
                        "mood": mood,
                        "tempo": tempo,
                        "active_instruments": active_text,
                        **instrument_scores,
                    }
                    observed["instrument"].add(instrument)
                    buckets[("instrument", instrument)].append(row)
                    sample_rows.append(row)
                    writer.writerow(row)

                if progress_every > 0 and len(seen) % progress_every == 0:
                    print(f"[{label}] processed {len(seen)} records", flush=True)
                if max_records is not None and len(seen) >= max_records:
                    break

    if not seen:
        raise RuntimeError(f"{label}: no records processed")
    if max_records is None and seen != set(range(expected_records)):
        raise RuntimeError(f"{label}: full-validation index coverage mismatch")
    if expected_references is not None and seen != set(expected_references):
        raise RuntimeError(f"{label}: record coverage differs from V2")

    items = requested_items(observed)
    group_summaries: list[dict[str, Any]] = []
    membership: dict[tuple[str, str, int], dict[str, Any]] = {}
    requested_sets = {
        "instrument": {item[0] for item in INSTRUMENT_SPECS},
        "style": set(REQUESTED_STYLES),
        "mood": set(REQUESTED_MOODS),
        "tempo": set(),
    }
    for category in CATEGORIES:
        for item in items[category]:
            rows = buckets.get((category, item), [])
            confidence, details = confidence_summary(rows, retain_fraction)
            for valid_index, detail in details.items():
                membership[(category, item, valid_index)] = detail
            group_summaries.append(
                {
                    "model": label,
                    "prompt_category": category,
                    "prompt_item": item,
                    "item_status": (
                        "observed"
                        if category == "tempo"
                        else "requested"
                        if item in requested_sets[category]
                        else "observed_extra"
                    ),
                    "all_samples": {
                        metric: base.summary_statistics(
                            [float(row[metric]) for row in rows]
                        )
                        for metric in METRICS
                    },
                    "confidence95": confidence,
                }
            )

    with selection_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            fieldnames
            + [
                "rank_D_P",
                "rank_D_D",
                "rank_D_IOI",
                "top95_D_P",
                "top95_D_D",
                "top95_D_IOI",
                "confidence95_retained",
            ]
        )
        for row in sample_rows:
            key = (
                str(row["prompt_category"]),
                str(row["prompt_item"]),
                int(row["valid_index"]),
            )
            detail = membership[key]
            writer.writerow([row[name] for name in fieldnames] + list(detail.values()))

    summary = {
        "schema_version": "bd3lms.prompt-conditioned-objective.v1",
        "model": label,
        "input_jsonl": str(jsonl),
        "input_jsonl_sha256": base.file_sha256(jsonl),
        "input_manifest": str(Path(str(jsonl) + ".manifest.json")),
        "input_manifest_sha256": base.file_sha256(
            Path(str(jsonl) + ".manifest.json")
        ),
        "evaluator_path": str(Path(__file__).resolve()),
        "evaluator_sha256": base.file_sha256(Path(__file__).resolve()),
        "inference_provenance_id": manifest.get("provenance_id"),
        "checkpoint": manifest.get("checkpoint"),
        "records_processed": len(seen),
        "observed_prompt_values": {
            category: sorted(values) for category, values in observed.items()
        },
        "requested_prompt_values": {
            "instrument": [item[0] for item in INSTRUMENT_SPECS],
            "style": list(REQUESTED_STYLES),
            "mood": list(REQUESTED_MOODS),
            "tempo": [],
        },
        "per_sample_prompt_metrics_csv": str(per_sample_path),
        "confidence95_membership_csv": str(selection_path),
        "groups": group_summaries,
    }
    base.atomic_write_json(run_dir / "summary.json", summary)
    return summary, references


def flatten_long_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in summary["groups"]:
        confidence = group["confidence95"]
        metrics = confidence["metrics"]
        rows.append(
            {
                "model": summary["model"],
                "prompt_category": group["prompt_category"],
                "prompt_item": group["prompt_item"],
                "item_status": group["item_status"],
                "n_samples": confidence["total_samples"],
                "retained_per_metric": confidence["retained_per_metric"],
                "n_confidence95": confidence["joint_retained_samples"],
                "D_P": metrics["D_P"]["mean"],
                "D_D": metrics["D_D"]["mean"],
                "D_IOI": metrics["D_IOI"]["mean"],
                "cutoff_D_P": confidence["cutoff_minimum_retained_score"]["D_P"],
                "cutoff_D_D": confidence["cutoff_minimum_retained_score"]["D_D"],
                "cutoff_D_IOI": confidence["cutoff_minimum_retained_score"]["D_IOI"],
            }
        )
    return rows


def write_tables(
    out_dir: Path, v2_summary: Mapping[str, Any], focal_summary: Mapping[str, Any]
) -> dict[str, str]:
    long_rows = flatten_long_rows(v2_summary) + flatten_long_rows(focal_summary)
    long_path = out_dir / "prompt_metrics_95_long.csv"
    long_fields = list(long_rows[0])
    with long_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=long_fields)
        writer.writeheader()
        writer.writerows(long_rows)

    by_key = {
        (row["model"], row["prompt_category"], row["prompt_item"]): row
        for row in long_rows
    }
    category_order = {category: index for index, category in enumerate(CATEGORIES)}
    item_order: dict[tuple[str, str], int] = {}
    for category, items in {
        "instrument": [item[0] for item in INSTRUMENT_SPECS],
        "style": list(REQUESTED_STYLES),
        "mood": list(REQUESTED_MOODS),
    }.items():
        for index, item in enumerate(items):
            item_order[(category, item)] = index
    keys = sorted(
        {(row["prompt_category"], row["prompt_item"]) for row in long_rows},
        key=lambda key: (
            category_order[key[0]],
            item_order.get(key, 10_000),
            key[1],
        ),
    )
    wide_path = out_dir / "prompt_metrics_95_table.csv"
    wide_fields = [
        "prompt_category",
        "prompt_item",
        "item_status",
        "V2_n_samples",
        "V2_n_confidence95",
        "V2_D_P",
        "V2_D_D",
        "V2_D_IOI",
        "Focal_n_samples",
        "Focal_n_confidence95",
        "Focal_D_P",
        "Focal_D_D",
        "Focal_D_IOI",
        "Focal_minus_V2_D_P",
        "Focal_minus_V2_D_D",
        "Focal_minus_V2_D_IOI",
    ]
    wide_rows: list[dict[str, Any]] = []
    for category, item in keys:
        v2 = by_key.get(("V2", category, item))
        focal = by_key.get(("Focal_SUBS", category, item))
        if v2 is None or focal is None:
            raise RuntimeError(f"missing model group for {category}/{item}")

        def difference(metric: str) -> float | None:
            if v2[metric] is None or focal[metric] is None:
                return None
            return float(focal[metric]) - float(v2[metric])

        wide_rows.append(
            {
                "prompt_category": category,
                "prompt_item": item,
                "item_status": (
                    "observed"
                    if v2["item_status"] == focal["item_status"] == "observed"
                    else "requested"
                    if v2["item_status"] == focal["item_status"] == "requested"
                    else "observed_extra"
                ),
                "V2_n_samples": v2["n_samples"],
                "V2_n_confidence95": v2["n_confidence95"],
                "V2_D_P": v2["D_P"],
                "V2_D_D": v2["D_D"],
                "V2_D_IOI": v2["D_IOI"],
                "Focal_n_samples": focal["n_samples"],
                "Focal_n_confidence95": focal["n_confidence95"],
                "Focal_D_P": focal["D_P"],
                "Focal_D_D": focal["D_D"],
                "Focal_D_IOI": focal["D_IOI"],
                "Focal_minus_V2_D_P": difference("D_P"),
                "Focal_minus_V2_D_D": difference("D_D"),
                "Focal_minus_V2_D_IOI": difference("D_IOI"),
            }
        )
    with wide_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=wide_fields)
        writer.writeheader()
        writer.writerows(wide_rows)

    observed_paths: dict[str, str] = {}
    for category in ("mood", "tempo"):
        observed_path = out_dir / f"{category}_metrics_95_table.csv"
        observed_rows = [
            {**row, "item_status": "observed"}
            for row in wide_rows
            if row["prompt_category"] == category
            and int(row["V2_n_samples"]) > 0
            and int(row["Focal_n_samples"]) > 0
        ]
        with observed_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=wide_fields)
            writer.writeheader()
            writer.writerows(observed_rows)
        observed_paths[category] = str(observed_path)

    return {
        "long_csv": str(long_path),
        "comparison_table_csv": str(wide_path),
        "observed_mood_table_csv": observed_paths["mood"],
        "observed_tempo_table_csv": observed_paths["tempo"],
    }


def main() -> None:
    args = parse_args()
    base.load_manifest(args.v2_jsonl.resolve())
    base.load_manifest(args.focal_jsonl.resolve())
    out_dir = base.prepare_output_dir(args.out_dir, args.overwrite)

    methodology = {
        "evaluation_unit": "one canonical validation record (valid_index)",
        "style_mood_tempo": (
            "read style, mood, and tempo key/value pairs from columns 0-15 of "
            "every active target row; all active rows must agree"
        ),
        "instrument": (
            "fixed target pitch/duration row pairs; activity is defined by the "
            "validated all-UNK row pair"
        ),
        "features": (
            "same histogram, Gaussian KDE, pitch-class, duration-class, and IOI "
            "definitions as Objective_evaluation_for_musical_pieces.py"
        ),
        "confidence95": (
            "within each model and prompt item, retain floor(0.95*N) highest "
            "samples independently for D_P, D_D, D_IOI and report their intersection"
        ),
        "requested_styles": list(REQUESTED_STYLES),
        "requested_moods": list(REQUESTED_MOODS),
        "tempo_values": "dynamically discovered from the full validation set",
        "unexpected_observed_values": (
            "observed values outside requested lists are preserved as observed_extra; "
            "requested values absent from validation are retained with n=0 and null metrics"
        ),
        "grid_step": args.grid_step,
        "minimum_bandwidth": args.min_bandwidth,
    }
    base.atomic_write_json(out_dir / "methodology.json", methodology)

    print(f"Output directory: {out_dir}", flush=True)
    v2_summary, references = process_run(
        "V2",
        args.v2_jsonl,
        out_dir,
        args.expected_records,
        args.grid_step,
        args.min_bandwidth,
        args.confidence_retain_fraction,
        args.max_records,
        args.progress_every,
    )
    focal_summary, _ = process_run(
        "Focal_SUBS",
        args.focal_jsonl,
        out_dir,
        args.expected_records,
        args.grid_step,
        args.min_bandwidth,
        args.confidence_retain_fraction,
        args.max_records,
        args.progress_every,
        expected_references=references,
    )
    tables = write_tables(out_dir, v2_summary, focal_summary)
    combined = {
        "schema_version": "bd3lms.prompt-conditioned-objective.comparison.v1",
        "methodology": methodology,
        "ground_truth_masks_and_seeds_identical_across_runs": True,
        "tables": tables,
        "models": {"V2": v2_summary, "Focal_SUBS": focal_summary},
    }
    base.atomic_write_json(out_dir / "combined_summary.json", combined)
    print(f"Comparison table: {tables['comparison_table_csv']}")
    print(f"Observed Mood table: {tables['observed_mood_table_csv']}")
    print(f"Observed Tempo table: {tables['observed_tempo_table_csv']}")
    print(f"Combined summary: {out_dir / 'combined_summary.json'}")


if __name__ == "__main__":
    main()
