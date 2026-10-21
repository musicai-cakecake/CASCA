#!/usr/bin/env python3
"""Produce one independent read-only Phase 0 identity/PAD-prior scan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from xmidi_getmusic.constants import (  # noqa: E402
    BLOCK_SIZE,
    MASK_ID,
    NUM_COLS,
    NUM_PROMPT_COLS,
    NUM_ROWS,
    NUM_SOURCE_ROWS,
    PAD_ID,
    ROW_NAMES,
    UNK_ID,
    VOCAB_SIZE,
)
from xmidi_getmusic.eval_protocol import (  # noqa: E402
    derive_sample_seed,
    inspect_v2_checkpoint,
)
from xmidi_getmusic.phase0_prior import (  # noqa: E402
    assert_expected_v2_counts,
    combine_counts,
    counts_to_json,
    load_offsets,
    load_sample,
    scan_split,
    sha256_file,
    write_hashed_json,
)
from xmidi_getmusic.runtime_requirements import (  # noqa: E402
    EXPECTED_WANDB_PROJECT,
    require_project_path,
    runtime_snapshot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-id", required=True, choices=("a", "b"))
    parser.add_argument(
        "--data-root", type=Path, required=True
    )
    parser.add_argument(
        "--source-data-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--v2-inference-manifest",
        type=Path,
        default=(
            REPO.parent
            / "CASCA_base"
            / "outputs"
            / "CASCA_base"
            / "inference_valid_best_ema"
            / "valid_inference.jsonl.manifest.json"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--reference-checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir", type=Path, default=REPO / "artifacts" / "phase0"
    )
    parser.add_argument("--pilot-size", type=int, default=512)
    parser.add_argument("--safety-size", type=int, default=64)
    parser.add_argument("--pilot-selection-seed", type=int, default=20260729)
    parser.add_argument("--safety-selection-seed", type=int, default=20260730)
    parser.add_argument("--sample-base-seed", type=int, default=1)
    return parser.parse_args()


def file_record(path: Path, source_path: Path | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "mtime_used_for_identity": False,
        "read_only_during_scan": True,
    }
    if source_path is not None:
        source_path = source_path.resolve()
        source_copy = {
            "path": str(source_path),
            "bytes": source_path.stat().st_size,
            "sha256": sha256_file(source_path),
        }
        source_copy["content_identical"] = (
            source_copy["bytes"] == record["bytes"]
            and source_copy["sha256"] == record["sha256"]
        )
        record["source_copy"] = source_copy
    return record


def dictionary_metadata(path: Path) -> dict[str, Any]:
    tokens = path.read_text(encoding="utf-8").splitlines()
    if len(tokens) != UNK_ID:
        raise ValueError(f"dictionary has {len(tokens)} entries, expected {UNK_ID}")
    if tokens[PAD_ID] != "<pad>":
        raise ValueError(f"dictionary token {PAD_ID} is {tokens[PAD_ID]!r}")
    token_sequence = "\n".join(tokens).encode("utf-8")
    return {
        "entries": len(tokens),
        "pad_id": PAD_ID,
        "pad_token": tokens[PAD_ID],
        "unk_id": UNK_ID,
        "mask_id": MASK_ID,
        "vocab_size": VOCAB_SIZE,
        "file_sha256": sha256_file(path),
        "token_sequence_serialization": "UTF-8 tokens joined by LF without terminal LF",
        "token_sequence_sha256": hashlib.sha256(token_sequence).hexdigest(),
    }


def validate_v2_historical_dataset_evidence(
    inputs: dict[str, Path], manifest_path: Path
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_files = manifest["identity"]["dataset"]["files"]
    mapping = {
        "valid_data": dataset_files["data"],
        "valid_idx": dataset_files["index"],
        "valid_length": dataset_files["length"],
    }
    checks: dict[str, bool] = {}
    records: dict[str, Any] = {}
    for name, expected in mapping.items():
        observed = {
            "path": str(inputs[name].resolve()),
            "bytes": inputs[name].stat().st_size,
            "sha256": sha256_file(inputs[name]),
        }
        records[name] = {"expected_from_v2_manifest": expected, "observed": observed}
        checks[f"{name}_byte_exact"] = (
            observed["bytes"] == expected["size_bytes"]
            and observed["sha256"] == expected["sha256"]
        )

    expected_dictionary = dataset_files["dictionary"]
    dictionary_bytes = inputs["dictionary"].read_bytes()
    normalized_bytes = (
        dictionary_bytes[:-1] if dictionary_bytes.endswith(b"\n") else dictionary_bytes
    )
    dictionary_record = {
        "expected_from_v2_manifest": expected_dictionary,
        "observed": {
            "path": str(inputs["dictionary"].resolve()),
            "bytes": len(dictionary_bytes),
            "sha256": hashlib.sha256(dictionary_bytes).hexdigest(),
        },
        "without_one_terminal_lf": {
            "bytes": len(normalized_bytes),
            "sha256": hashlib.sha256(normalized_bytes).hexdigest(),
        },
        "identity_policy": (
            "V2 manifest bytes are authoritative; one terminal LF is ignored only for "
            "dictionary token-sequence identity."
        ),
    }
    records["dictionary"] = dictionary_record
    checks["dictionary_token_sequence_exact"] = (
        len(normalized_bytes) == expected_dictionary["size_bytes"]
        and hashlib.sha256(normalized_bytes).hexdigest()
        == expected_dictionary["sha256"]
    )
    checks["manifest_checkpoint_exact"] = (
        manifest["identity"]["checkpoint_sha256"]
        == "d1edd7e7342c3fc952add33d70124369d69c1dde14208ed570b4fe50ca8dcd37"
    )
    return {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "records": records,
        "checks": checks,
        "passed": all(checks.values()),
        "scope_note": (
            "The V2 full-valid manifest directly authenticates valid data/index/length "
            "and dictionary token sequence. Train bytes are cross-checked against the "
            "original data-production copy and validated by exact V2 sample/PAD counts."
        ),
    }


def sample_descriptors(
    data_path: Path, idx_path: Path, indices: np.ndarray
) -> dict[int, dict[str, Any]]:
    offsets = load_offsets(data_path, idx_path)
    descriptors: dict[int, dict[str, Any]] = {}
    with data_path.open("rb") as handle:
        for index_value in indices:
            index = int(index_value)
            sample = load_sample(handle, offsets[index], offsets[index + 1], index)
            target = sample[NUM_SOURCE_ROWS:, NUM_PROMPT_COLS:]
            generation = target != UNK_ID
            pad = (target == PAD_ID) & generation
            generation_cells = int(generation.sum())
            event_cells = generation_cells - int(pad.sum())
            inactive_rows = [
                row_index
                for row_index in range(NUM_SOURCE_ROWS, NUM_ROWS)
                if not bool(generation[row_index - NUM_SOURCE_ROWS].any())
            ]
            descriptors[index] = {
                "generation_cells": generation_cells,
                "pad": int(pad.sum()),
                "non_pad": event_cells,
                "pad_ratio": float(pad.sum() / generation_cells) if generation_cells else 0.0,
                "inactive_target_rows": inactive_rows,
                "active_target_row_count": len(TARGET_ROW_NAMES) - len(inactive_rows),
            }
    return descriptors


TARGET_ROW_NAMES = ROW_NAMES[NUM_SOURCE_ROWS:]


def subset_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    cells = [item["generation_cells"] for item in samples]
    ratios = [item["pad_ratio"] for item in samples]
    inactive_coverage = sorted(
        {row for item in samples for row in item["inactive_target_rows"]}
    )
    return {
        "size": len(samples),
        "generation_cells_min": min(cells),
        "generation_cells_max": max(cells),
        "generation_cells_mean": float(np.mean(cells)),
        "pad_ratio_min": min(ratios),
        "pad_ratio_max": max(ratios),
        "pad_ratio_mean": float(np.mean(ratios)),
        "inactive_target_row_coverage": inactive_coverage,
        "active_target_row_count_histogram": {
            str(value): sum(item["active_target_row_count"] == value for item in samples)
            for value in sorted({item["active_target_row_count"] for item in samples})
        },
    }


def main() -> None:
    args = parse_args()
    os.environ.setdefault("WANDB_PROJECT", EXPECTED_WANDB_PROJECT)
    runtime = runtime_snapshot(require_tmux=True, require_wandb=True)
    data_root = args.data_root.resolve()
    source_root = args.source_data_root.resolve()
    v2_inference_manifest = args.v2_inference_manifest.resolve()
    output_root = require_project_path(args.output_dir, label="Phase 0 output")
    scan_dir = output_root / f"scan_{args.scan_id}"
    scan_dir.mkdir(parents=True, exist_ok=True)

    inputs = {
        "train_data": data_root / "train.data",
        "train_idx": data_root / "train.idx",
        "train_length": data_root / "train_length.npy",
        "valid_data": data_root / "valid.data",
        "valid_idx": data_root / "valid.idx",
        "valid_length": data_root / "valid_length.npy",
        "dictionary": data_root / "whole_dict.txt",
    }
    source_files = {
        name: source_root / path.name for name, path in inputs.items()
    }
    required = [*inputs.values(), *source_files.values(), v2_inference_manifest]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing Phase 0 input/evidence: {missing}")

    created_at = datetime.now(timezone.utc).isoformat()
    before = {
        name: file_record(path, source_files[name]) for name, path in inputs.items()
    }
    train = scan_split(inputs["train_data"], inputs["train_idx"])
    valid = scan_split(inputs["valid_data"], inputs["valid_idx"])
    combined = combine_counts((train, valid))
    assert_expected_v2_counts("train", train)
    assert_expected_v2_counts("valid", valid)
    v2_dataset_evidence = validate_v2_historical_dataset_evidence(
        inputs, v2_inference_manifest
    )
    after = {
        name: file_record(path, source_files[name]) for name, path in inputs.items()
    }
    if before != after:
        raise RuntimeError("input/source content changed during read-only scan")
    if not all(record["source_copy"]["content_identical"] for record in after.values()):
        raise RuntimeError("new-project data differs from original data-production copy")
    if not v2_dataset_evidence["passed"]:
        raise RuntimeError("new-project data fails V2 historical manifest validation")

    checkpoint = inspect_v2_checkpoint(
        args.checkpoint, require_ema=True, compute_sha256=True
    )
    reference_checkpoint = inspect_v2_checkpoint(
        args.reference_checkpoint, require_ema=True, compute_sha256=True
    )
    checkpoint_match = (
        checkpoint["size_bytes"] == reference_checkpoint["size_bytes"]
        and checkpoint["sha256"] == reference_checkpoint["sha256"]
    )
    if not checkpoint_match:
        raise RuntimeError("new-project best.ckpt differs from V2 reference checkpoint")

    identity = {
        "schema_version": "bd3lms-music-v2-focal.phase0.identity.v1",
        "scan_id": args.scan_id,
        "created_at_utc": created_at,
        "project": EXPECTED_WANDB_PROJECT,
        "data_root": str(data_root),
        "source_data_root": str(source_root),
        "mtime_identity_policy": "mtime is explicitly excluded; SHA-256/bytes/counts are authoritative",
        "files": after,
        "v2_historical_dataset_evidence": v2_dataset_evidence,
        "dictionary": dictionary_metadata(inputs["dictionary"]),
        "grid": {
            "num_rows": NUM_ROWS,
            "num_cols": NUM_COLS,
            "num_source_rows": NUM_SOURCE_ROWS,
            "num_prompt_cols": NUM_PROMPT_COLS,
            "block_size": BLOCK_SIZE,
            "row_names": list(ROW_NAMES),
        },
        "checkpoint": checkpoint,
        "reference_checkpoint": reference_checkpoint,
        "checkpoint_content_identical": checkpoint_match,
        "runtime": runtime,
    }
    identity_sha = write_hashed_json(scan_dir / "data_checkpoint_identity.json", identity)

    prior = {
        "schema_version": "bd3lms-music-v2-focal.phase0.pad-prior.v1",
        "scan_id": args.scan_id,
        "created_at_utc": created_at,
        "project": EXPECTED_WANDB_PROJECT,
        "source_identity_sha256": identity_sha,
        "normalization_domain": "canonical generation cells only",
        "region_definition": "G=1 iff row>=4, column>=16, token!=<UNK>",
        "known_v2_count_check": True,
        "splits": {
            "train": counts_to_json(train),
            "valid": counts_to_json(valid),
            "combined": counts_to_json(combined),
        },
    }
    prior_sha = write_hashed_json(scan_dir / "pad_prior.json", prior)

    if not 0 < args.safety_size <= args.pilot_size <= valid.examples:
        raise ValueError("require 0 < safety_size <= pilot_size <= valid examples")
    pilot_rng = np.random.default_rng(args.pilot_selection_seed)
    pilot_indices = np.sort(
        pilot_rng.choice(valid.examples, size=args.pilot_size, replace=False)
    ).astype(np.int64)
    safety_rng = np.random.default_rng(args.safety_selection_seed)
    safety_indices = np.sort(
        safety_rng.choice(pilot_indices, size=args.safety_size, replace=False)
    ).astype(np.int64)
    descriptors = sample_descriptors(
        inputs["valid_data"], inputs["valid_idx"], pilot_indices
    )
    samples = [
        {
            "valid_index": int(index),
            "sample_seed": derive_sample_seed(args.sample_base_seed, int(index)),
            **descriptors[int(index)],
        }
        for index in pilot_indices
    ]
    safety_set = set(int(index) for index in safety_indices)
    safety_samples = [item for item in samples if item["valid_index"] in safety_set]
    pilot = {
        "schema_version": "bd3lms-music-v2-focal.phase0.pilot.v1",
        "scan_id": args.scan_id,
        "created_at_utc": created_at,
        "project": EXPECTED_WANDB_PROJECT,
        "source_split": "valid",
        "source_identity_sha256": identity_sha,
        "source_pad_prior_sha256": prior_sha,
        "selection": {
            "method": "uniform_without_replacement_numpy_pcg64_then_sorted",
            "pilot_selection_seed": args.pilot_selection_seed,
            "safety_selection_seed": args.safety_selection_seed,
            "population_size": valid.examples,
            "pilot_size": args.pilot_size,
            "safety_size": args.safety_size,
        },
        "sample_seed": {
            "domain": "bd3lms-music-v2",
            "base_seed": args.sample_base_seed,
            "algorithm": "first_64_bits_of_sha256_domain_base_seed_sample_index",
        },
        "pilot_summary": subset_summary(samples),
        "safety_summary": subset_summary(safety_samples),
        "samples": samples,
        "safety_valid_indices": [int(index) for index in safety_indices],
    }
    pilot_sha = write_hashed_json(scan_dir / "pilot_valid_512.json", pilot)

    result = {
        "scan_id": args.scan_id,
        "identity": str(scan_dir / "data_checkpoint_identity.json"),
        "identity_sha256": identity_sha,
        "pad_prior": str(scan_dir / "pad_prior.json"),
        "pad_prior_sha256": prior_sha,
        "pilot": str(scan_dir / "pilot_valid_512.json"),
        "pilot_sha256": pilot_sha,
        "train_generation_cells": train.generation_cells,
        "train_pad": train.pad,
        "valid_generation_cells": valid.generation_cells,
        "valid_pad": valid.pad,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
