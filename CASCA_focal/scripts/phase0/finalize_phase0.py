#!/usr/bin/env python3
"""Validate all Phase 0 evidence and publish canonical frozen artifacts."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from xmidi_getmusic.phase0_prior import (  # noqa: E402
    canonical_json_sha256,
    sha256_file,
    write_hashed_json,
)
from xmidi_getmusic.runtime_requirements import (  # noqa: E402
    EXPECTED_PYTHON,
    EXPECTED_CONDA_ENV,
    EXPECTED_TMUX,
    EXPECTED_WANDB_PROJECT,
    require_project_path,
    runtime_snapshot,
)

PHASE0 = REPO / "artifacts" / "phase0"


def load_hashed(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.get("artifact_sha256")
    body = dict(payload)
    body.pop("artifact_sha256", None)
    observed = canonical_json_sha256(body)
    if expected != observed:
        raise RuntimeError(f"artifact self-hash mismatch: {path}")
    return payload


def strip_fields(payload: dict[str, Any], fields: set[str]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key not in fields}


def require_exit_zero(path: Path) -> bool:
    return path.read_text(encoding="utf-8").strip() == "0"


def read_jsonl_first(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.loads(next(line for line in handle if line.strip()))


def main() -> None:
    os.environ.setdefault("WANDB_PROJECT", EXPECTED_WANDB_PROJECT)
    runtime = runtime_snapshot(require_tmux=True, require_wandb=True)
    scan_a_identity = load_hashed(PHASE0 / "scan_a" / "data_checkpoint_identity.json")
    scan_b_identity = load_hashed(PHASE0 / "scan_b" / "data_checkpoint_identity.json")
    scan_a_prior = load_hashed(PHASE0 / "scan_a" / "pad_prior.json")
    scan_b_prior = load_hashed(PHASE0 / "scan_b" / "pad_prior.json")
    scan_a_pilot = load_hashed(PHASE0 / "scan_a" / "pilot_valid_512.json")
    scan_b_pilot = load_hashed(PHASE0 / "scan_b" / "pilot_valid_512.json")
    boundary = load_hashed(PHASE0 / "project_boundary_audit.json")
    one_step = load_hashed(PHASE0 / "one_step_regression_report.json")
    v2_pilot = load_hashed(PHASE0 / "v2_pilot_generation_baseline.json")

    teacher_v2_path = PHASE0 / "teacher_forced" / "v2.json"
    teacher_focal_path = PHASE0 / "teacher_forced" / "focal.json"
    teacher_v2 = json.loads(teacher_v2_path.read_text(encoding="utf-8"))
    teacher_focal = json.loads(teacher_focal_path.read_text(encoding="utf-8"))
    teacher_semantic_v2 = strip_fields(teacher_v2, {"evaluator_sha256"})
    teacher_semantic_focal = strip_fields(teacher_focal, {"evaluator_sha256"})

    sampling_v2_path = PHASE0 / "sampling" / "v2.jsonl"
    sampling_focal_path = PHASE0 / "sampling" / "focal.jsonl"
    sampling_v2 = read_jsonl_first(sampling_v2_path)
    sampling_focal = read_jsonl_first(sampling_focal_path)
    sampling_keys = (
        "valid_index",
        "sample_seed",
        "ground_truth",
        "gen",
        "row_unk_mask",
        "generation_mask_rle",
        "generation_mask_sha256",
        "generated_positions",
        "structural_audit",
        "match_stats",
        "diffusion_nfes",
        "backbone_forward_calls",
        "nfes",
    )
    sampling_checks = {
        key: sampling_v2.get(key) == sampling_focal.get(key) for key in sampling_keys
    }

    files_exact = scan_a_identity["files"] == scan_b_identity["files"]
    source_copies_exact = all(
        record["source_copy"]["content_identical"]
        for record in scan_a_identity["files"].values()
    )
    v2_dataset_evidence_a = scan_a_identity["v2_historical_dataset_evidence"]
    v2_dataset_evidence_b = scan_b_identity["v2_historical_dataset_evidence"]
    v2_dataset_evidence_exact = v2_dataset_evidence_a == v2_dataset_evidence_b
    prior_exact = scan_a_prior["splits"] == scan_b_prior["splits"]
    pilot_exact = (
        scan_a_pilot["samples"] == scan_b_pilot["samples"]
        and scan_a_pilot["safety_valid_indices"]
        == scan_b_pilot["safety_valid_indices"]
    )
    checkpoint_a = scan_a_identity["checkpoint"]
    checkpoint_b = scan_b_identity["checkpoint"]
    train = scan_a_prior["splits"]["train"]
    valid = scan_a_prior["splits"]["valid"]
    combined = scan_a_prior["splits"]["combined"]

    runtime_checks = {
        "conda_environment": not EXPECTED_CONDA_ENV or runtime["conda_environment"] == EXPECTED_CONDA_ENV,
        "python_executable": Path(runtime["python_executable"]).resolve()
        == EXPECTED_PYTHON.resolve(),
        "tmux_identity": not EXPECTED_TMUX or EXPECTED_TMUX
        in {runtime["tmux"].get("session"), runtime["tmux"].get("window")},
        "wandb_project": runtime["wandb_project"] == EXPECTED_WANDB_PROJECT,
    }
    checks = {
        "scan_a_exit_zero": require_exit_zero(PHASE0 / "scan_a.exit"),
        "scan_b_exit_zero": require_exit_zero(PHASE0 / "scan_b.exit"),
        "independent_file_records_exact": files_exact,
        "local_data_matches_original_source_copy": source_copies_exact,
        "v2_historical_dataset_evidence_passed": v2_dataset_evidence_a["passed"],
        "v2_historical_dataset_evidence_exact_across_scans": v2_dataset_evidence_exact,
        "v2_valid_data_index_length_byte_exact": all(
            v2_dataset_evidence_a["checks"][key]
            for key in (
                "valid_data_byte_exact",
                "valid_idx_byte_exact",
                "valid_length_byte_exact",
            )
        ),
        "v2_dictionary_token_sequence_exact": v2_dataset_evidence_a["checks"][
            "dictionary_token_sequence_exact"
        ],
        "independent_pad_prior_integer_exact": prior_exact,
        "independent_pilot_exact": pilot_exact,
        "train_examples_exact": train["examples"] == 133_646,
        "train_generation_cells_exact": train["generation_cells"] == 323_238_700,
        "train_pad_exact": train["pad"] == 217_999_140,
        "valid_examples_exact": valid["examples"] == 6_817,
        "valid_generation_cells_exact": valid["generation_cells"] == 14_310_994,
        "valid_pad_exact": valid["pad"] == 8_634_610,
        "combined_counts_exact": (
            combined["generation_cells"] == 337_549_694
            and combined["pad"] == 226_633_750
            and combined["non_pad"] == 110_915_944
        ),
        "dictionary_contract_exact": (
            scan_a_identity["dictionary"]["entries"] == 142_419
            and scan_a_identity["dictionary"]["pad_id"] == 605
            and scan_a_identity["dictionary"]["unk_id"] == 142_419
            and scan_a_identity["dictionary"]["mask_id"] == 142_420
        ),
        "checkpoint_scans_exact": checkpoint_a["sha256"] == checkpoint_b["sha256"],
        "checkpoint_matches_v2_reference": scan_a_identity["checkpoint_content_identical"],
        "checkpoint_vocabulary_exact": checkpoint_a["embedding_shape"] == [142_421, 768],
        "checkpoint_has_ema": checkpoint_a["has_ema"],
        "project_boundary_audit_passed": boundary["passed"],
        "one_step_loss_gradient_regression_passed": one_step["passed"],
        "teacher_forced_exit_zero": require_exit_zero(PHASE0 / "teacher_forced.exit"),
        "teacher_forced_semantic_output_exact": teacher_semantic_v2 == teacher_semantic_focal,
        "teacher_forced_selected_count_exact": (
            teacher_v2["aggregate"]["num_loss_positions"]
            == teacher_focal["aggregate"]["num_loss_positions"]
            == 4_950
        ),
        "teacher_forced_nll_exact": teacher_v2["aggregate"]["nll"]
        == teacher_focal["aggregate"]["nll"],
        "sampling_exit_zero": require_exit_zero(PHASE0 / "sampling.exit"),
        "sampling_token_mask_seed_audit_exact": all(sampling_checks.values()),
        "v2_pilot_baseline_exit_zero": require_exit_zero(
            PHASE0 / "v2_pilot_baseline.exit"
        ),
        "v2_pilot_uses_current_frozen_pilot": v2_pilot["pilot"][
            "artifact_sha256"
        ]
        == scan_a_pilot["artifact_sha256"],
        "v2_pilot_complete": v2_pilot["pilot"]["samples"] == 512,
        "v2_pilot_structural_audit_100pct": v2_pilot["structural_audit"][
            "joint_pass_rate"
        ]
        == 1.0,
        "v2_pytest_exit_zero": require_exit_zero(
            PHASE0 / "tests" / "v2_regression.exit"
        ),
        **{f"runtime_{name}": value for name, value in runtime_checks.items()},
    }

    authoritative_identity = strip_fields(
        scan_a_identity, {"artifact_sha256", "scan_id", "created_at_utc"}
    )
    authoritative_identity.update({
        "schema_version": "bd3lms-music-v2-focal.phase0.identity.authoritative.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "independent_scans": {
            "scan_a_sha256": scan_a_identity["artifact_sha256"],
            "scan_b_sha256": scan_b_identity["artifact_sha256"],
            "semantic_file_records_exact": files_exact,
        },
    })
    identity_path = PHASE0 / "data_checkpoint_identity.json"
    identity_sha = write_hashed_json(identity_path, authoritative_identity)

    authoritative_prior = strip_fields(
        scan_a_prior,
        {"artifact_sha256", "scan_id", "created_at_utc", "source_identity_sha256"},
    )
    authoritative_prior.update({
        "schema_version": "bd3lms-music-v2-focal.phase0.pad-prior.authoritative.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_identity_sha256": identity_sha,
        "independent_scans": {
            "scan_a_sha256": scan_a_prior["artifact_sha256"],
            "scan_b_sha256": scan_b_prior["artifact_sha256"],
            "all_integer_statistics_exact": prior_exact,
        },
    })
    prior_path = PHASE0 / "pad_prior.json"
    prior_sha = write_hashed_json(prior_path, authoritative_prior)

    authoritative_pilot = strip_fields(
        scan_a_pilot,
        {
            "artifact_sha256",
            "scan_id",
            "created_at_utc",
            "source_identity_sha256",
            "source_pad_prior_sha256",
        },
    )
    authoritative_pilot.update({
        "schema_version": "bd3lms-music-v2-focal.phase0.pilot.authoritative.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_identity_sha256": identity_sha,
        "source_pad_prior_sha256": prior_sha,
        "independent_scans": {
            "scan_a_sha256": scan_a_pilot["artifact_sha256"],
            "scan_b_sha256": scan_b_pilot["artifact_sha256"],
            "samples_and_seeds_exact": pilot_exact,
        },
    })
    pilot_path = PHASE0 / "pilot_valid_512.json"
    pilot_sha = write_hashed_json(pilot_path, authoritative_pilot)

    safety_indices = set(authoritative_pilot["safety_valid_indices"])
    safety_samples = [
        item
        for item in authoritative_pilot["samples"]
        if item["valid_index"] in safety_indices
    ]
    safety_payload = {
        "schema_version": "bd3lms-music-v2-focal.phase0.safety-subset.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": EXPECTED_WANDB_PROJECT,
        "source_pilot_sha256": pilot_sha,
        "selection_seed": authoritative_pilot["selection"]["safety_selection_seed"],
        "summary": authoritative_pilot["safety_summary"],
        "samples": safety_samples,
    }
    safety_path = PHASE0 / "safety_valid_64.json"
    safety_sha = write_hashed_json(safety_path, safety_payload)

    teacher_report = {
        "schema_version": "bd3lms-music-v2-focal.phase0-teacher-regression.v1",
        "project": EXPECTED_WANDB_PROJECT,
        "v2_summary": {"path": str(teacher_v2_path), "sha256": sha256_file(teacher_v2_path)},
        "focal_summary": {"path": str(teacher_focal_path), "sha256": sha256_file(teacher_focal_path)},
        "aggregate": {
            "v2_nll": teacher_v2["aggregate"]["nll"],
            "focal_nll": teacher_focal["aggregate"]["nll"],
            "absolute_nll_error": abs(
                teacher_v2["aggregate"]["nll"] - teacher_focal["aggregate"]["nll"]
            ),
            "selected_loss_positions": teacher_v2["aggregate"]["num_loss_positions"],
        },
        "semantic_output_exact_excluding_evaluator_hash": teacher_semantic_v2
        == teacher_semantic_focal,
        "passed": (
            teacher_semantic_v2 == teacher_semantic_focal
            and teacher_v2["aggregate"]["nll"] == teacher_focal["aggregate"]["nll"]
        ),
    }
    teacher_report_path = PHASE0 / "teacher_forced_regression_report.json"
    teacher_report_sha = write_hashed_json(teacher_report_path, teacher_report)

    sampling_report = {
        "schema_version": "bd3lms-music-v2-focal.phase0-sampling-regression.v1",
        "project": EXPECTED_WANDB_PROJECT,
        "protocol": {
            "valid_index": 10,
            "base_seed": 1,
            "T": 16,
            "nucleus_p": 0.95,
            "weight_source": "ema",
            "model": "xmidi_getmusic_small",
        },
        "v2_jsonl": {"path": str(sampling_v2_path), "sha256": sha256_file(sampling_v2_path)},
        "focal_jsonl": {
            "path": str(sampling_focal_path),
            "sha256": sha256_file(sampling_focal_path),
        },
        "field_checks": sampling_checks,
        "generated_grid_elementwise_exact": sampling_checks["gen"],
        "passed": all(sampling_checks.values()),
    }
    sampling_report_path = PHASE0 / "sampling_regression_report.json"
    sampling_report_sha = write_hashed_json(sampling_report_path, sampling_report)

    passed = all(checks.values())
    deliverables = {
        "data_checkpoint_identity": {"path": str(identity_path), "sha256": identity_sha},
        "pad_prior": {"path": str(prior_path), "sha256": prior_sha},
        "pilot_valid_512": {"path": str(pilot_path), "sha256": pilot_sha},
        "safety_valid_64": {"path": str(safety_path), "sha256": safety_sha},
        "project_boundary_audit": {
            "path": str(PHASE0 / "project_boundary_audit.json"),
            "sha256": boundary["artifact_sha256"],
        },
        "one_step_regression": {
            "path": str(PHASE0 / "one_step_regression_report.json"),
            "sha256": one_step["artifact_sha256"],
        },
        "teacher_forced_regression": {
            "path": str(teacher_report_path),
            "sha256": teacher_report_sha,
        },
        "sampling_regression": {
            "path": str(sampling_report_path),
            "sha256": sampling_report_sha,
        },
        "v2_pilot_generation_baseline": {
            "path": str(PHASE0 / "v2_pilot_generation_baseline.json"),
            "sha256": v2_pilot["artifact_sha256"],
        },
    }
    status = {
        "schema_version": "bd3lms-music-v2-focal.phase-status.v1",
        "phase": 0,
        "name": "Project isolation, data identity, PAD prior, pilot, and V2 regression",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": EXPECTED_WANDB_PROJECT,
        "status": "PASSED" if passed else "FAILED",
        "runtime": runtime,
        "checks": checks,
        "deliverables": deliverables,
        "reference_statistics": {
            "train_pad_ratio": train["pad_ratio"],
            "valid_pad_ratio": valid["pad_ratio"],
            "combined_pad_ratio": combined["pad_ratio"],
            "v2_pilot_generation": v2_pilot["global"],
        },
        "limitations": [
            "Phase 0 implements no focal loss and makes no focal-quality claim.",
            "The T=16 sampling run is an equivalence regression, not a quality evaluation.",
            "Canonical V2 pilot quality uses the historical T=1000 full-valid EMA output.",
        ],
        "next_phase": "Phase 1 may start only if status is PASSED.",
    }
    status_path = require_project_path(PHASE0 / "phase0_status.json", label="Phase 0 status")
    status_sha = write_hashed_json(status_path, status)
    print(json.dumps({
        "status": status["status"],
        "status_path": str(status_path),
        "status_sha256": status_sha,
        "checks": checks,
    }, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
