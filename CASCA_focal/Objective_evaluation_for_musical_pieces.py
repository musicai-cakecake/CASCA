#!/usr/bin/env python3
"""Evaluate musical-feature distribution overlap on full-validation inference.

For every canonical inference record, this script extracts three numeric feature
samples from the exact generation region G=1:

* Pitch (P): 12 pitch classes from target pitch rows. Each encoded pitch is
  reduced modulo 12; a compound chord token contributes every pitch regardless
  of the order in which pitches appear in the token.
* Duration (D): encoded duration-class integers from target duration rows.
* Inter-Onset Interval (IOI): original-column distance between adjacent valid
  onset cells in a target pitch row, computed within one contiguous G=1 segment.

For each feature and validation record, aligned GT/GEN integer histograms are
formed first. A shared-bandwidth Gaussian kernel is then applied to the two
histograms to obtain numerical PDFs. Their overlapped area is

    OA = integral min(PDF_GT(x), PDF_GEN(x)) dx.

D_P, D_D, and D_IOI are macro averages of the corresponding per-record OAs.
They are similarities in [0, 1] (larger is closer), despite the historical D_*
notation. A record with empty GT and GEN feature sets receives OA=1; if exactly
one side is empty, it receives OA=0, preserving a common macro-average denominator. The complementary distances 1-D_* are also reported.

The script streams the large JSONL files and writes per-record metrics, corpus
histograms, corpus KDE curves, per-run summaries, and one combined summary.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:
    from scipy.ndimage import gaussian_filter1d
except ImportError as exc:  # pragma: no cover - environment guard
    raise RuntimeError(
        "SciPy is required for Gaussian KDE smoothing; install scipy first"
    ) from exc


NUM_ROWS = 16
NUM_COLS = 512
NUM_SOURCE_ROWS = 4
NUM_PROMPT_COLS = 16
SEQ_LEN = NUM_ROWS * NUM_COLS
INFERENCE_SCHEMA_VERSION = "bd3lms-music-v2.inference.v3"
MANIFEST_SCHEMA_VERSION = "bd3lms-music-v2.manifest.v3"
PROJECT_ROOT = Path(__file__).resolve().parent
SAFE_OUTPUT_ROOT = (PROJECT_ROOT / "artifacts").resolve()
SPECIAL_TOKENS = frozenset({"<pad>", "<UNK>", "<MASK>"})

V2_JSONL_DEFAULT = Path(os.environ.get(
    "CASCA_BASE_JSONL", str(PROJECT_ROOT.parent / "CASCA_base" / "outputs" / "CASCA_base" / "inference_valid_best_ema" / "valid_inference.jsonl")))
FOCAL_JSONL_DEFAULT = Path(os.environ.get(
    "CASCA_FOCAL_JSONL", str(PROJECT_ROOT / "outputs" / "CASCA_focal" / "inference_valid_best_ema" / "valid_inference.jsonl")))
OUTPUT_DIR_DEFAULT = SAFE_OUTPUT_ROOT / "objective_evaluation_for_musical_pieces"

PITCH_ROW_SPECS = {
    4: ("piano", "piano"),
    6: ("guitar", "guitar"),
    8: ("bass", "bass"),
    10: ("string", "string"),
    12: ("wind", "wind"),
    14: ("drum", "drum"),
}
DURATION_ROW_SPECS = {
    pitch_row + 1: (instrument, f"Duration_{instrument}_")
    for pitch_row, (instrument, _) in PITCH_ROW_SPECS.items()
}
PITCH_ROWS = tuple(PITCH_ROW_SPECS)
DURATION_ROWS = tuple(DURATION_ROW_SPECS)
PITCH_PREFIXES = tuple(prefix for _, prefix in PITCH_ROW_SPECS.values())
DURATION_PREFIXES = tuple(prefix for _, prefix in DURATION_ROW_SPECS.values())
FEATURES = ("pitch", "duration", "ioi")
_INTEGER_RE = re.compile(r"^-?\d+$")


@dataclass
class ExtractedFeatures:
    pitch: list[int] = field(default_factory=list)
    duration: list[int] = field(default_factory=list)
    ioi: list[int] = field(default_factory=list)
    invalid_pitch_tokens: int = 0
    invalid_duration_tokens: int = 0


@dataclass
class FeatureScore:
    overlap: float | None
    status: str
    gt_count: int
    gen_count: int
    bandwidth: float | None


@dataclass
class RunAccumulator:
    label: str
    jsonl: Path
    records: int = 0
    seen_indices: set[int] = field(default_factory=set)
    overlap_values: dict[str, list[float]] = field(
        default_factory=lambda: {feature: [] for feature in FEATURES}
    )
    status_counts: dict[str, Counter[str]] = field(
        default_factory=lambda: {feature: Counter() for feature in FEATURES}
    )
    corpus_gt: dict[str, Counter[int]] = field(
        default_factory=lambda: {feature: Counter() for feature in FEATURES}
    )
    corpus_gen: dict[str, Counter[int]] = field(
        default_factory=lambda: {feature: Counter() for feature in FEATURES}
    )
    invalid_gt_pitch_tokens: int = 0
    invalid_gen_pitch_tokens: int = 0
    invalid_gt_duration_tokens: int = 0
    invalid_gen_duration_tokens: int = 0
    total_selected_positions: int = 0
    per_sample_scores: list[dict[str, float | int]] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-sample KDE distribution overlap D_P, D_D, and D_IOI "
            "for V2 and final Focal SUBS full-validation inference outputs."
        )
    )
    parser.add_argument("--v2-jsonl", type=Path, default=V2_JSONL_DEFAULT)
    parser.add_argument("--focal-jsonl", type=Path, default=FOCAL_JSONL_DEFAULT)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR_DEFAULT)
    parser.add_argument("--expected-records", type=int, default=6817)
    parser.add_argument(
        "--grid-step",
        type=float,
        default=0.25,
        help="Numerical KDE grid spacing in feature-class units.",
    )
    parser.add_argument(
        "--min-bandwidth",
        type=float,
        default=0.5,
        help="Minimum shared Gaussian KDE bandwidth per sample.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional smoke-test limit; disables full-set coverage checks.",
    )
    parser.add_argument(
        "--progress-every", type=int, default=250, help="Progress print interval."
    )
    parser.add_argument(
        "--confidence-retain-fraction",
        type=float,
        default=0.95,
        help="Fraction retained independently by each metric before intersection.",
    )
    parser.add_argument(
        "--demo-top-k",
        type=int,
        default=20,
        help="Number of top retained Focal samples exported for demos.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing output directory."
    )
    args = parser.parse_args()
    if args.expected_records <= 0:
        parser.error("--expected-records must be positive")
    if args.grid_step <= 0:
        parser.error("--grid-step must be positive")
    if args.min_bandwidth <= 0:
        parser.error("--min-bandwidth must be positive")
    if args.max_records is not None and args.max_records <= 0:
        parser.error("--max-records must be positive")
    if not 0.0 < args.confidence_retain_fraction <= 1.0:
        parser.error("--confidence-retain-fraction must be in (0, 1]")
    if args.demo_top_k <= 0:
        parser.error("--demo-top-k must be positive")
    return args


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path, chunk_size: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def reference_signature(record: Mapping[str, Any]) -> str:
    payload = {
        "ground_truth": record["ground_truth"],
        "generation_mask_rle": record["generation_mask_rle"],
        "sample_seed": int(record["sample_seed"]),
    }
    return canonical_sha256(payload)


def load_manifest(jsonl: Path) -> dict[str, Any]:
    manifest_path = Path(str(jsonl) + ".manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("manifest_schema_version") != MANIFEST_SCHEMA_VERSION:
        raise RuntimeError(f"unexpected manifest schema in {manifest_path}")
    if manifest.get("status") != "completed":
        raise RuntimeError(f"manifest is not completed: {manifest_path}")
    if manifest.get("inference_schema_version") != INFERENCE_SCHEMA_VERSION:
        raise RuntimeError(f"unexpected inference schema in {manifest_path}")
    identity = manifest.get("identity")
    if not isinstance(identity, dict):
        raise RuntimeError(f"manifest has no identity mapping: {manifest_path}")
    if manifest.get("provenance_id") != canonical_sha256(identity):
        raise RuntimeError(f"manifest identity/provenance mismatch: {manifest_path}")
    resolved_output = Path(manifest.get("output_jsonl", "")).resolve()
    if resolved_output != jsonl.resolve():
        raise RuntimeError(f"manifest output_jsonl differs from input: {manifest_path}")
    return manifest


def decode_mask(runs: Sequence[Sequence[int]]) -> np.ndarray:
    mask = np.zeros(SEQ_LEN, dtype=np.bool_)
    previous_end = 0
    for run_index, run in enumerate(runs):
        if not isinstance(run, list) or len(run) != 2:
            raise ValueError(f"RLE run {run_index} must be [start, length]")
        start, length = int(run[0]), int(run[1])
        end = start + length
        if length <= 0 or start < previous_end or start < 0 or end > SEQ_LEN:
            raise ValueError(f"invalid/overlapping RLE run {run_index}: {run}")
        mask[start:end] = True
        previous_end = end
    grid = mask.reshape(NUM_ROWS, NUM_COLS)
    if grid[:NUM_SOURCE_ROWS].any():
        raise ValueError("generation mask selects source rows")
    if grid[NUM_SOURCE_ROWS:, :NUM_PROMPT_COLS].any():
        raise ValueError("generation mask selects target prompts")
    return grid


def validate_record(
    record: Mapping[str, Any], manifest: Mapping[str, Any], line_number: int
) -> tuple[int, np.ndarray]:
    if record.get("schema_version") != INFERENCE_SCHEMA_VERSION:
        raise RuntimeError(f"invalid schema at line {line_number}")
    if record.get("provenance_id") != manifest.get("provenance_id"):
        raise RuntimeError(f"provenance mismatch at line {line_number}")
    valid_index = int(record["valid_index"])
    gt = record.get("ground_truth")
    gen = record.get("gen")
    if not isinstance(gt, list) or not isinstance(gen, list):
        raise RuntimeError(f"missing token grids at line {line_number}")
    if len(gt) != NUM_ROWS or len(gen) != NUM_ROWS:
        raise RuntimeError(f"grid row count mismatch at line {line_number}")
    if any(len(row) != NUM_COLS for row in gt) or any(
        len(row) != NUM_COLS for row in gen
    ):
        raise RuntimeError(f"grid column count mismatch at line {line_number}")
    audit = record.get("structural_audit")
    if not isinstance(audit, dict) or not audit.get("all_pass", False):
        raise RuntimeError(f"structural audit failed/missing at line {line_number}")
    runs = record.get("generation_mask_rle")
    if not isinstance(runs, list):
        raise RuntimeError(f"missing generation_mask_rle at line {line_number}")
    if record.get("generation_mask_sha256") != canonical_sha256({"rle": runs}):
        raise RuntimeError(f"generation-mask hash mismatch at line {line_number}")
    mask = decode_mask(runs)
    gt_array = np.asarray(gt, dtype=object)
    if np.any(mask & (gt_array == "<UNK>")):
        raise RuntimeError(f"generation mask selects structural <UNK> at line {line_number}")
    selected = int(mask.sum())
    if int(record.get("generated_positions", -1)) != selected:
        raise RuntimeError(f"generated-position mismatch at line {line_number}")
    return valid_index, mask


def parse_pitch_token(token: str, row: int) -> list[int] | None:
    if token in SPECIAL_TOKENS:
        return []
    if row not in PITCH_ROW_SPECS:
        return None
    prefix = next((item for item in PITCH_PREFIXES if token.startswith(item)), None)
    if prefix is None:
        return None
    payload = token[len(prefix) :].strip()
    parts = payload.split()
    if not parts or any(_INTEGER_RE.fullmatch(part) is None for part in parts):
        return None
    return [int(part) % 12 for part in parts]


def parse_duration_token(token: str, row: int) -> list[int] | None:
    if token in SPECIAL_TOKENS:
        return []
    if row not in DURATION_ROW_SPECS:
        return None
    prefix = next((item for item in DURATION_PREFIXES if token.startswith(item)), None)
    if prefix is None:
        return None
    payload = token[len(prefix) :].strip()
    if _INTEGER_RE.fullmatch(payload) is None:
        return None
    return [int(payload)]


def true_segments(row_mask: np.ndarray) -> Iterable[np.ndarray]:
    columns = np.flatnonzero(row_mask)
    if columns.size == 0:
        return
    split_points = np.flatnonzero(np.diff(columns) > 1) + 1
    for segment in np.split(columns, split_points):
        if segment.size:
            yield segment


def extract_features(
    grid: Sequence[Sequence[str]], mask: np.ndarray
) -> ExtractedFeatures:
    features = ExtractedFeatures()

    for row in PITCH_ROWS:
        for col in np.flatnonzero(mask[row]):
            parsed = parse_pitch_token(grid[row][int(col)], row)
            if parsed is None:
                features.invalid_pitch_tokens += 1
            else:
                features.pitch.extend(parsed)

    for row in DURATION_ROWS:
        for col in np.flatnonzero(mask[row]):
            parsed = parse_duration_token(grid[row][int(col)], row)
            if parsed is None:
                features.invalid_duration_tokens += 1
            else:
                features.duration.extend(parsed)

    for row in PITCH_ROWS:
        for segment in true_segments(mask[row]):
            onset_columns: list[int] = []
            for col in segment:
                parsed = parse_pitch_token(grid[row][int(col)], row)
                if parsed:
                    onset_columns.append(int(col))
            features.ioi.extend(
                right - left
                for left, right in zip(onset_columns, onset_columns[1:])
            )

    return features


def counter_from(values: Sequence[int]) -> Counter[int]:
    return Counter(int(value) for value in values)


def weighted_mean_std(counter: Counter[int]) -> tuple[float, float, int]:
    n = int(sum(counter.values()))
    if n == 0:
        return float("nan"), float("nan"), 0
    mean = sum(value * count for value, count in counter.items()) / n
    variance = sum(count * (value - mean) ** 2 for value, count in counter.items()) / n
    return float(mean), float(math.sqrt(max(variance, 0.0))), n


def weighted_quantile(counter: Counter[int], quantile: float) -> float:
    if not counter:
        return float("nan")
    total = sum(counter.values())
    threshold = quantile * max(total - 1, 0)
    cumulative = 0
    for value in sorted(counter):
        next_cumulative = cumulative + counter[value]
        if threshold < next_cumulative:
            return float(value)
        cumulative = next_cumulative
    return float(max(counter))


def select_bandwidth(
    gt: Counter[int], gen: Counter[int], min_bandwidth: float
) -> float:
    pooled = gt + gen
    _, std, n = weighted_mean_std(pooled)
    q25 = weighted_quantile(pooled, 0.25)
    q75 = weighted_quantile(pooled, 0.75)
    robust = (q75 - q25) / 1.349 if math.isfinite(q25) and math.isfinite(q75) else 0.0
    positive_scales = [scale for scale in (std, robust) if scale > 0 and math.isfinite(scale)]
    scale = min(positive_scales) if positive_scales else 0.0
    silverman = 0.9 * scale * (n ** -0.2) if n > 1 and scale > 0 else 0.0
    return float(max(min_bandwidth, silverman))


def kde_pdfs_from_histograms(
    gt: Counter[int],
    gen: Counter[int],
    grid_step: float,
    min_bandwidth: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if not gt or not gen:
        raise ValueError("both histograms must be nonempty for KDE")
    bandwidth = select_bandwidth(gt, gen, min_bandwidth)
    minimum = min(min(gt), min(gen))
    maximum = max(max(gt), max(gen))
    margin = max(4.0 * bandwidth, 1.0)
    lower = math.floor((minimum - margin) / grid_step) * grid_step
    upper = math.ceil((maximum + margin) / grid_step) * grid_step
    grid = np.arange(lower, upper + 0.5 * grid_step, grid_step, dtype=np.float64)

    def smooth(counter: Counter[int]) -> np.ndarray:
        density = np.zeros(grid.size, dtype=np.float64)
        total = float(sum(counter.values()))
        for value, count in counter.items():
            index = int(round((value - lower) / grid_step))
            if not 0 <= index < density.size:
                raise RuntimeError("histogram value fell outside KDE grid")
            density[index] += count / total / grid_step
        density = gaussian_filter1d(
            density,
            sigma=bandwidth / grid_step,
            mode="constant",
            cval=0.0,
            truncate=4.0,
        )
        area = float(np.trapz(density, grid))
        if not math.isfinite(area) or area <= 0:
            raise RuntimeError("KDE normalization failed")
        return density / area

    return grid, smooth(gt), smooth(gen), bandwidth


def score_feature(
    gt_values: Sequence[int],
    gen_values: Sequence[int],
    grid_step: float,
    min_bandwidth: float,
) -> FeatureScore:
    gt = counter_from(gt_values)
    gen = counter_from(gen_values)
    if not gt and not gen:
        return FeatureScore(1.0, "both_empty", 0, 0, None)
    if not gt:
        return FeatureScore(0.0, "gt_empty", 0, sum(gen.values()), None)
    if not gen:
        return FeatureScore(0.0, "gen_empty", sum(gt.values()), 0, None)
    grid, gt_pdf, gen_pdf, bandwidth = kde_pdfs_from_histograms(
        gt, gen, grid_step, min_bandwidth
    )
    overlap = float(np.trapz(np.minimum(gt_pdf, gen_pdf), grid))
    overlap = min(1.0, max(0.0, overlap))
    return FeatureScore(overlap, "scored", sum(gt.values()), sum(gen.values()), bandwidth)


def feature_values(features: ExtractedFeatures, name: str) -> list[int]:
    return getattr(features, name)


def summary_statistics(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "std": None, "median": None, "min": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def confidence_selection(
    rows: Sequence[Mapping[str, float | int]],
    retain_fraction: float,
    out_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, float | int]]]:
    if not rows:
        raise ValueError("confidence selection requires at least one sample")
    metric_names = ("D_P", "D_D", "D_IOI")
    retain_count = max(1, math.floor(len(rows) * retain_fraction))
    retained_by_metric: dict[str, set[int]] = {}
    ranks: dict[str, dict[int, int]] = {}
    cutoffs: dict[str, float] = {}

    for metric in metric_names:
        ranked = sorted(rows, key=lambda row: (-float(row[metric]), int(row["valid_index"])))
        retained = ranked[:retain_count]
        retained_by_metric[metric] = {int(row["valid_index"]) for row in retained}
        ranks[metric] = {
            int(row["valid_index"]): rank
            for rank, row in enumerate(ranked, start=1)
        }
        cutoffs[metric] = float(retained[-1][metric])

    retained_ids = set.intersection(*(retained_by_metric[name] for name in metric_names))
    retained_rows: list[dict[str, float | int]] = []
    selection_path = out_dir / "confidence95_selection.csv"
    with selection_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "valid_index",
                "D_P",
                "D_D",
                "D_IOI",
                "rank_D_P",
                "rank_D_D",
                "rank_D_IOI",
                "top95_D_P",
                "top95_D_D",
                "top95_D_IOI",
                "confidence95_retained",
                "composite_mean",
                "minimum_metric",
            ]
        )
        for source in sorted(rows, key=lambda row: int(row["valid_index"])):
            row = dict(source)
            valid_index = int(row["valid_index"])
            keep_flags = {
                metric: valid_index in retained_by_metric[metric]
                for metric in metric_names
            }
            retained = all(keep_flags.values())
            row["composite_mean"] = float(
                sum(float(row[metric]) for metric in metric_names) / len(metric_names)
            )
            row["minimum_metric"] = float(
                min(float(row[metric]) for metric in metric_names)
            )
            if retained:
                retained_rows.append(row)
            writer.writerow(
                [
                    valid_index,
                    row["D_P"],
                    row["D_D"],
                    row["D_IOI"],
                    ranks["D_P"][valid_index],
                    ranks["D_D"][valid_index],
                    ranks["D_IOI"][valid_index],
                    keep_flags["D_P"],
                    keep_flags["D_D"],
                    keep_flags["D_IOI"],
                    retained,
                    row["composite_mean"],
                    row["minimum_metric"],
                ]
            )

    retained_rows.sort(
        key=lambda row: (
            -float(row["composite_mean"]),
            -float(row["minimum_metric"]),
            int(row["valid_index"]),
        )
    )
    metric_summaries = {
        metric: summary_statistics([float(row[metric]) for row in retained_rows])
        for metric in metric_names
    }
    summary = {
        "retain_fraction_per_metric": retain_fraction,
        "ranking_rule": (
            "retain exactly floor(N*retain_fraction) highest scores for each metric; "
            "ties are resolved by ascending valid_index"
        ),
        "joint_rule": "retain the intersection; failure on any metric excludes the sample",
        "total_samples": len(rows),
        "retained_per_metric": retain_count,
        "excluded_per_metric": len(rows) - retain_count,
        "joint_retained_samples": len(retained_rows),
        "joint_excluded_samples": len(rows) - len(retained_rows),
        "cutoff_minimum_retained_score": cutoffs,
        "metrics": metric_summaries,
        "selection_csv": str(selection_path),
    }
    return summary, retained_rows


def write_demo_records(
    jsonl: Path,
    ranked_rows: Sequence[Mapping[str, float | int]],
    out_dir: Path,
    top_k: int,
) -> dict[str, Any]:
    selected_rows = list(ranked_rows[:top_k])
    selected_by_index = {int(row["valid_index"]): dict(row) for row in selected_rows}
    selected_indices = set(selected_by_index)
    csv_path = out_dir / f"top{top_k}_demo_samples.csv"
    json_path = out_dir / f"top{top_k}_demo_samples.json"
    jsonl_path = out_dir / f"top{top_k}_demo_records.jsonl"

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["rank", "valid_index", "composite_mean", "minimum_metric", "D_P", "D_D", "D_IOI"]
        )
        for rank, row in enumerate(selected_rows, start=1):
            writer.writerow(
                [
                    rank,
                    row["valid_index"],
                    row["composite_mean"],
                    row["minimum_metric"],
                    row["D_P"],
                    row["D_D"],
                    row["D_IOI"],
                ]
            )

    ranked_json = [
        {"rank": rank, **dict(row)}
        for rank, row in enumerate(selected_rows, start=1)
    ]
    atomic_write_json(
        json_path,
        {
            "ranking": (
                "descending mean(D_P,D_D,D_IOI), then descending minimum metric, "
                "then ascending valid_index; candidates are joint 95%-confidence samples"
            ),
            "samples": ranked_json,
        },
    )

    records_by_index: dict[int, dict[str, Any]] = {}
    with jsonl.open(encoding="utf-8") as input_handle:
        for line in input_handle:
            if not line.strip():
                continue
            record = json.loads(line)
            valid_index = int(record["valid_index"])
            if valid_index in selected_indices:
                record["objective_distribution_metrics"] = selected_by_index[valid_index]
                records_by_index[valid_index] = record
    if set(records_by_index) != selected_indices:
        raise RuntimeError(
            f"missing top-demo records: {sorted(selected_indices - set(records_by_index))}"
        )
    with jsonl_path.open("w", encoding="utf-8") as output_handle:
        for row in selected_rows:
            record = records_by_index[int(row["valid_index"])]
            output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "top_k": len(selected_rows),
        "ranking_csv": str(csv_path),
        "ranking_json": str(json_path),
        "records_jsonl": str(jsonl_path),
        "samples": ranked_json,
    }


def write_distribution_files(
    out_dir: Path,
    feature: str,
    gt: Counter[int],
    gen: Counter[int],
    grid_step: float,
    min_bandwidth: float,
) -> dict[str, Any]:
    histogram_path = out_dir / f"{feature}_corpus_histogram.csv"
    with histogram_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["value", "gt_count", "gen_count", "gt_probability", "gen_probability"])
        gt_total = sum(gt.values())
        gen_total = sum(gen.values())
        for value in sorted(set(gt) | set(gen)):
            writer.writerow(
                [
                    value,
                    gt[value],
                    gen[value],
                    gt[value] / gt_total if gt_total else 0.0,
                    gen[value] / gen_total if gen_total else 0.0,
                ]
            )

    result: dict[str, Any] = {
        "gt_observations": int(sum(gt.values())),
        "gen_observations": int(sum(gen.values())),
        "histogram_csv": str(histogram_path),
    }
    if gt and gen:
        grid, gt_pdf, gen_pdf, bandwidth = kde_pdfs_from_histograms(
            gt, gen, grid_step, min_bandwidth
        )
        kde_path = out_dir / f"{feature}_corpus_kde.csv"
        with kde_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["x", "gt_pdf", "gen_pdf", "overlap_pdf"])
            writer.writerows(zip(grid, gt_pdf, gen_pdf, np.minimum(gt_pdf, gen_pdf)))
        result.update(
            {
                "bandwidth": bandwidth,
                "kde_csv": str(kde_path),
                "corpus_pooled_overlap": float(
                    np.trapz(np.minimum(gt_pdf, gen_pdf), grid)
                ),
            }
        )
    else:
        result.update({"bandwidth": None, "kde_csv": None, "corpus_pooled_overlap": None})
    return result


def process_run(
    label: str,
    jsonl: Path,
    out_root: Path,
    expected_records: int,
    grid_step: float,
    min_bandwidth: float,
    max_records: int | None,
    progress_every: int,
    confidence_retain_fraction: float,
    expected_references: Mapping[int, str] | None = None,
) -> tuple[dict[str, Any], dict[int, str], list[dict[str, float | int]]]:
    jsonl = jsonl.resolve()
    if not jsonl.is_file():
        raise FileNotFoundError(jsonl)
    manifest = load_manifest(jsonl)
    manifest_records = int(manifest.get("records_written", -1))
    if max_records is None and manifest_records != expected_records:
        raise RuntimeError(
            f"{label}: manifest records_written={manifest_records}, expected={expected_records}"
        )

    run_dir = out_root / label
    run_dir.mkdir(parents=True, exist_ok=True)
    per_sample_path = run_dir / "per_sample_overlap.csv"
    accumulator = RunAccumulator(label=label, jsonl=jsonl)
    references: dict[int, str] = {}

    fieldnames = [
        "model",
        "valid_index",
        "D_P",
        "D_D",
        "D_IOI",
        "pitch_status",
        "duration_status",
        "ioi_status",
        "pitch_gt_count",
        "pitch_gen_count",
        "duration_gt_count",
        "duration_gen_count",
        "ioi_gt_count",
        "ioi_gen_count",
        "pitch_bandwidth",
        "duration_bandwidth",
        "ioi_bandwidth",
        "invalid_gt_pitch_tokens",
        "invalid_gen_pitch_tokens",
        "invalid_gt_duration_tokens",
        "invalid_gen_duration_tokens",
    ]

    with per_sample_path.open("w", encoding="utf-8", newline="") as output_handle:
        writer = csv.DictWriter(output_handle, fieldnames=fieldnames)
        writer.writeheader()
        with jsonl.open(encoding="utf-8") as input_handle:
            for line_number, line in enumerate(input_handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                valid_index, mask = validate_record(record, manifest, line_number)
                if valid_index in accumulator.seen_indices:
                    raise RuntimeError(f"{label}: duplicate valid_index={valid_index}")
                accumulator.seen_indices.add(valid_index)
                accumulator.records += 1
                accumulator.total_selected_positions += int(mask.sum())

                signature = reference_signature(record)
                references[valid_index] = signature
                if expected_references is not None:
                    expected = expected_references.get(valid_index)
                    if expected is None or expected != signature:
                        raise RuntimeError(
                            f"{label}: GT/mask differs from V2 at valid_index={valid_index}"
                        )

                gt_features = extract_features(record["ground_truth"], mask)
                gen_features = extract_features(record["gen"], mask)
                accumulator.invalid_gt_pitch_tokens += gt_features.invalid_pitch_tokens
                accumulator.invalid_gen_pitch_tokens += gen_features.invalid_pitch_tokens
                accumulator.invalid_gt_duration_tokens += gt_features.invalid_duration_tokens
                accumulator.invalid_gen_duration_tokens += gen_features.invalid_duration_tokens

                scores: dict[str, FeatureScore] = {}
                for feature in FEATURES:
                    gt_values = feature_values(gt_features, feature)
                    gen_values = feature_values(gen_features, feature)
                    accumulator.corpus_gt[feature].update(gt_values)
                    accumulator.corpus_gen[feature].update(gen_values)
                    score = score_feature(
                        gt_values, gen_values, grid_step, min_bandwidth
                    )
                    scores[feature] = score
                    accumulator.status_counts[feature][score.status] += 1
                    if score.overlap is not None:
                        accumulator.overlap_values[feature].append(score.overlap)

                output_row = {
                    "model": label,
                    "valid_index": valid_index,
                    "D_P": scores["pitch"].overlap,
                    "D_D": scores["duration"].overlap,
                    "D_IOI": scores["ioi"].overlap,
                    "pitch_status": scores["pitch"].status,
                    "duration_status": scores["duration"].status,
                    "ioi_status": scores["ioi"].status,
                    "pitch_gt_count": scores["pitch"].gt_count,
                    "pitch_gen_count": scores["pitch"].gen_count,
                    "duration_gt_count": scores["duration"].gt_count,
                    "duration_gen_count": scores["duration"].gen_count,
                    "ioi_gt_count": scores["ioi"].gt_count,
                    "ioi_gen_count": scores["ioi"].gen_count,
                    "pitch_bandwidth": scores["pitch"].bandwidth,
                    "duration_bandwidth": scores["duration"].bandwidth,
                    "ioi_bandwidth": scores["ioi"].bandwidth,
                    "invalid_gt_pitch_tokens": gt_features.invalid_pitch_tokens,
                    "invalid_gen_pitch_tokens": gen_features.invalid_pitch_tokens,
                    "invalid_gt_duration_tokens": gt_features.invalid_duration_tokens,
                    "invalid_gen_duration_tokens": gen_features.invalid_duration_tokens,
                }
                writer.writerow(output_row)
                accumulator.per_sample_scores.append(
                    {
                        "valid_index": valid_index,
                        "D_P": float(scores["pitch"].overlap),
                        "D_D": float(scores["duration"].overlap),
                        "D_IOI": float(scores["ioi"].overlap),
                    }
                )

                if progress_every > 0 and accumulator.records % progress_every == 0:
                    print(f"[{label}] processed {accumulator.records} records", flush=True)
                if max_records is not None and accumulator.records >= max_records:
                    break

    if accumulator.records == 0:
        raise RuntimeError(f"{label}: no records were processed")
    if max_records is None:
        expected_indices = set(range(expected_records))
        if accumulator.seen_indices != expected_indices:
            missing = sorted(expected_indices - accumulator.seen_indices)[:10]
            extra = sorted(accumulator.seen_indices - expected_indices)[:10]
            raise RuntimeError(f"{label}: index coverage mismatch; missing={missing}, extra={extra}")
    if expected_references is not None and set(expected_references) != accumulator.seen_indices:
        raise RuntimeError(f"{label}: record coverage differs from V2")

    confidence95, retained_rows = confidence_selection(
        accumulator.per_sample_scores, confidence_retain_fraction, run_dir
    )

    metric_names = {"pitch": "D_P", "duration": "D_D", "ioi": "D_IOI"}
    metrics: dict[str, Any] = {}
    corpus: dict[str, Any] = {}
    for feature in FEATURES:
        stats = summary_statistics(accumulator.overlap_values[feature])
        mean = stats["mean"]
        metrics[metric_names[feature]] = {
            **stats,
            "distance_1_minus_mean": None if mean is None else 1.0 - float(mean),
            "status_counts": dict(accumulator.status_counts[feature]),
        }
        corpus[feature] = write_distribution_files(
            run_dir,
            feature,
            accumulator.corpus_gt[feature],
            accumulator.corpus_gen[feature],
            grid_step,
            min_bandwidth,
        )

    summary = {
        "schema_version": "bd3lms.objective-musical-pieces.v2",
        "model": label,
        "input_jsonl": str(jsonl),
        "input_jsonl_sha256": file_sha256(jsonl),
        "input_manifest": str(Path(str(jsonl) + ".manifest.json")),
        "input_manifest_sha256": file_sha256(Path(str(jsonl) + ".manifest.json")),
        "evaluator_path": str(Path(__file__).resolve()),
        "evaluator_sha256": file_sha256(Path(__file__).resolve()),
        "inference_provenance_id": manifest.get("provenance_id"),
        "checkpoint": manifest.get("checkpoint"),
        "records_processed": accumulator.records,
        "selected_positions": accumulator.total_selected_positions,
        "per_sample_csv": str(per_sample_path),
        "metrics_all_samples": metrics,
        "confidence95": confidence95,
        "corpus_distributions": corpus,
        "invalid_token_counts": {
            "gt_pitch": accumulator.invalid_gt_pitch_tokens,
            "gen_pitch": accumulator.invalid_gen_pitch_tokens,
            "gt_duration": accumulator.invalid_gt_duration_tokens,
            "gen_duration": accumulator.invalid_gen_duration_tokens,
        },
    }
    atomic_write_json(run_dir / "summary.json", summary)
    return summary, references, retained_rows


def prepare_output_dir(path: Path, overwrite: bool) -> Path:
    path = path.resolve()
    if path == SAFE_OUTPUT_ROOT or SAFE_OUTPUT_ROOT not in path.parents:
        raise ValueError(f"output directory must be a child of {SAFE_OUTPUT_ROOT}, got {path}")
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"output directory is nonempty: {path}; pass --overwrite to replace results"
        )
    if path.exists() and overwrite:
        for child in path.iterdir():
            if child.is_file() or child.is_symlink():
                child.unlink()
            else:
                import shutil

                shutil.rmtree(child)
    path.mkdir(parents=True, exist_ok=True)
    return path


def main() -> None:
    args = parse_args()
    load_manifest(args.v2_jsonl.resolve())
    load_manifest(args.focal_jsonl.resolve())
    out_dir = prepare_output_dir(args.out_dir, args.overwrite)

    methodology = {
        "evaluation_unit": "one canonical validation record (valid_index)",
        "region": "exact generation_mask_rle cells (G=1) only",
        "pitch": (
            "12 pitch classes computed as each encoded pitch integer modulo 12; "
            "compound chord tokens are split and pitch order is ignored; "
            "pad/UNK/MASK excluded"
        ),
        "duration": (
            "integer class payload in target duration-row tokens; "
            "pad/UNK/MASK excluded"
        ),
        "ioi": (
            "difference in original column indices between adjacent valid onset "
            "cells within one target pitch-row contiguous G=1 segment"
        ),
        "histogram": "integer feature-count histogram per record and corpus",
        "kde": (
            "Gaussian smoothing of histogram impulse masses; GT and GEN share a "
            "per-record Silverman bandwidth with a configured lower bound"
        ),
        "overlap_area": "numerical integral of min(PDF_GT, PDF_GEN)",
        "aggregation": (
            "macro mean over all records; both-empty records receive OA=1 and "
            "one-side-empty records receive OA=0"
        ),
        "confidence_selection": (
            "within each model, retain the highest floor(N*0.95) samples for each "
            "of D_P, D_D, and D_IOI, then retain their intersection; one poor metric "
            "is sufficient to exclude a sample"
        ),
        "demo_ranking": (
            "among retained Focal samples, descending mean(D_P,D_D,D_IOI), then "
            "descending minimum metric, then ascending valid_index"
        ),
        "interpretation": "D_P, D_D, D_IOI are OA similarities; higher is closer",
        "grid_step": args.grid_step,
        "minimum_bandwidth": args.min_bandwidth,
    }
    atomic_write_json(out_dir / "methodology.json", methodology)

    print(f"Output directory: {out_dir}", flush=True)
    v2_summary, references, _ = process_run(
        "V2",
        args.v2_jsonl,
        out_dir,
        args.expected_records,
        args.grid_step,
        args.min_bandwidth,
        args.max_records,
        args.progress_every,
        args.confidence_retain_fraction,
    )
    focal_summary, _, focal_retained_rows = process_run(
        "Focal_SUBS",
        args.focal_jsonl,
        out_dir,
        args.expected_records,
        args.grid_step,
        args.min_bandwidth,
        args.max_records,
        args.progress_every,
        args.confidence_retain_fraction,
        expected_references=references,
    )
    focal_demo = write_demo_records(
        args.focal_jsonl.resolve(),
        focal_retained_rows,
        out_dir / "Focal_SUBS",
        args.demo_top_k,
    )
    focal_summary["top_demo_samples"] = focal_demo
    atomic_write_json(out_dir / "Focal_SUBS" / "summary.json", focal_summary)

    combined = {
        "schema_version": "bd3lms.objective-musical-pieces.comparison.v2",
        "methodology": methodology,
        "ground_truth_masks_and_seeds_identical_across_runs": True,
        "models": {"V2": v2_summary, "Focal_SUBS": focal_summary},
        "comparison_confidence95": {},
        "comparison_all_samples": {},
        "focal_top_demo_samples": focal_demo,
    }
    for metric in ("D_P", "D_D", "D_IOI"):
        for comparison_key, source_key in (
            ("comparison_confidence95", "confidence95"),
            ("comparison_all_samples", "metrics_all_samples"),
        ):
            if source_key == "confidence95":
                v2_mean = v2_summary[source_key]["metrics"][metric]["mean"]
                focal_mean = focal_summary[source_key]["metrics"][metric]["mean"]
            else:
                v2_mean = v2_summary[source_key][metric]["mean"]
                focal_mean = focal_summary[source_key][metric]["mean"]
            combined[comparison_key][metric] = {
                "V2": v2_mean,
                "Focal_SUBS": focal_mean,
                "Focal_minus_V2": (
                    None
                    if v2_mean is None or focal_mean is None
                    else float(focal_mean) - float(v2_mean)
                ),
            }
    atomic_write_json(out_dir / "combined_summary.json", combined)

    def display(value: float | None, *, signed: bool = False) -> str:
        if value is None:
            return "undefined"
        return f"{value:+.6f}" if signed else f"{value:.6f}"

    print("\nJoint 95%-confidence KDE overlap areas (higher is closer):")
    for metric in ("D_P", "D_D", "D_IOI"):
        v2 = combined["comparison_confidence95"][metric]["V2"]
        focal = combined["comparison_confidence95"][metric]["Focal_SUBS"]
        delta = combined["comparison_confidence95"][metric]["Focal_minus_V2"]
        print(
            f"  {metric}: V2={display(v2)}  Focal_SUBS={display(focal)}  "
            f"delta={display(delta, signed=True)}"
        )
    print(f"Combined summary: {out_dir / 'combined_summary.json'}")


if __name__ == "__main__":
    main()
