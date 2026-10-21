#!/usr/bin/env python3
"""Extract the frozen 512-sample V2 generation baseline from full-valid output."""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from xmidi_getmusic.constants import NUM_COLS, NUM_ROWS  # noqa: E402
from xmidi_getmusic.eval_protocol import rle_to_mask  # noqa: E402
from xmidi_getmusic.phase0_prior import sha256_file, write_hashed_json  # noqa: E402
from xmidi_getmusic.runtime_requirements import (  # noqa: E402
    EXPECTED_WANDB_PROJECT,
    require_project_path,
    runtime_snapshot,
)

V2_INFERENCE = Path(os.environ.get(
    "CASCA_BASE_JSONL", str(REPO.parent / "CASCA_base" / "outputs" / "CASCA_base" / "inference_valid_best_ema" / "valid_inference.jsonl")))
INSTRUMENT_PAIRS = (
    ("piano", 4, 5),
    ("guitar", 6, 7),
    ("bass", 8, 9),
    ("string", 10, 11),
    ("wind", 12, 13),
    ("drum", 14, 15),
)


def ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def main() -> None:
    os.environ.setdefault("WANDB_PROJECT", EXPECTED_WANDB_PROJECT)
    runtime = runtime_snapshot(require_tmux=True, require_wandb=True)
    pilot_path = REPO / "artifacts" / "phase0" / "scan_a" / "pilot_valid_512.json"
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    expected = {int(item["valid_index"]): item for item in pilot["samples"]}
    manifest_path = V2_INFERENCE.with_suffix(V2_INFERENCE.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    focal_checkpoint_hash = json.loads(
        (REPO / "artifacts" / "phase0" / "scan_a" / "data_checkpoint_identity.json").read_text(
            encoding="utf-8"
        )
    )["checkpoint"]["sha256"]
    if manifest["checkpoint"]["sha256"] != focal_checkpoint_hash:
        raise RuntimeError("historical V2 inference used a different checkpoint")
    identity = manifest["identity"]
    if identity["weight_source"] != "ema" or identity["base_seed"] != 1:
        raise RuntimeError("historical inference is not the canonical EMA/seed=1 run")

    counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"cells": 0, "reference_pad": 0, "generated_pad": 0}
    )
    exact_matches = 0
    structural_pass = 0
    found: dict[int, dict[str, Any]] = {}
    with V2_INFERENCE.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            index = int(record["valid_index"])
            if index not in expected:
                continue
            if int(record["sample_seed"]) != int(expected[index]["sample_seed"]):
                raise RuntimeError(f"sample seed mismatch at valid_index={index}")
            mask = rle_to_mask(record["generation_mask_rle"]).numpy().reshape(
                NUM_ROWS, NUM_COLS
            )
            reference = np.asarray(record["ground_truth"], dtype=object)
            generated = np.asarray(record["gen"], dtype=object)
            if reference.shape != (NUM_ROWS, NUM_COLS) or generated.shape != reference.shape:
                raise RuntimeError(f"invalid grid shape at valid_index={index}")
            reference_pad = (reference == "<pad>") & mask
            generated_pad = (generated == "<pad>") & mask
            exact_matches += int(((reference == generated) & mask).sum())
            counts["global"]["cells"] += int(mask.sum())
            counts["global"]["reference_pad"] += int(reference_pad.sum())
            counts["global"]["generated_pad"] += int(generated_pad.sum())
            for name, first, second in INSTRUMENT_PAIRS:
                pair_mask = mask[[first, second]]
                counts[name]["cells"] += int(pair_mask.sum())
                counts[name]["reference_pad"] += int(reference_pad[[first, second]].sum())
                counts[name]["generated_pad"] += int(generated_pad[[first, second]].sum())
            audit = record.get("structural_audit", {})
            structural_pass += int(bool(audit.get("all_pass", False)))
            found[index] = {
                "valid_index": index,
                "sample_seed": int(record["sample_seed"]),
                "generated_positions": int(mask.sum()),
                "structural_audit_pass": bool(audit.get("all_pass", False)),
            }
    missing = sorted(set(expected) - set(found))
    if missing or len(found) != 512:
        raise RuntimeError(f"expected 512 pilot samples; found={len(found)}, missing={missing}")

    def summarize(item: dict[str, int]) -> dict[str, Any]:
        cells = item["cells"]
        reference_event = cells - item["reference_pad"]
        generated_event = cells - item["generated_pad"]
        return {
            **item,
            "reference_event": reference_event,
            "generated_event": generated_event,
            "reference_pad_ratio": ratio(item["reference_pad"], cells),
            "generated_pad_ratio": ratio(item["generated_pad"], cells),
            "pad_gap_percentage_points": 100.0
            * ratio(item["generated_pad"] - item["reference_pad"], cells),
            "reference_event_density": ratio(reference_event, cells),
            "generated_event_density": ratio(generated_event, cells),
            "event_density_ratio": ratio(generated_event, reference_event),
        }

    payload = {
        "schema_version": "bd3lms-music-v2-focal.phase0-v2-pilot-baseline.v1",
        "project": EXPECTED_WANDB_PROJECT,
        "source": {
            "jsonl": str(V2_INFERENCE),
            "jsonl_sha256": sha256_file(V2_INFERENCE),
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "checkpoint_sha256": focal_checkpoint_hash,
            "weight_source": "ema",
            "sampling": identity["sampling"],
            "base_seed": identity["base_seed"],
        },
        "pilot": {
            "path": str(pilot_path),
            "artifact_sha256": pilot["artifact_sha256"],
            "samples": len(found),
        },
        "global": summarize(counts["global"]),
        "per_instrument_pair": {
            name: summarize(counts[name]) for name, _, _ in INSTRUMENT_PAIRS
        },
        "exact_match_ratio": ratio(exact_matches, counts["global"]["cells"]),
        "structural_audit": {
            "passed": structural_pass,
            "total": len(found),
            "joint_pass_rate": ratio(structural_pass, len(found)),
        },
        "runtime": runtime,
        "sample_records": [found[index] for index in sorted(found)],
    }
    if payload["structural_audit"]["joint_pass_rate"] != 1.0:
        raise RuntimeError("historical V2 pilot contains a structural audit failure")
    out = require_project_path(
        REPO / "artifacts" / "phase0" / "v2_pilot_generation_baseline.json",
        label="V2 pilot generation baseline",
    )
    write_hashed_json(out, payload)
    print(json.dumps({"out": str(out), "global": payload["global"]}, indent=2))


if __name__ == "__main__":
    main()
