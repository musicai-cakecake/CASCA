#!/usr/bin/env python3
"""Validate Phase 2 integration/smoke evidence and publish status."""

from __future__ import annotations

import ast
import json
import math
import os
import statistics
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from xmidi_getmusic.phase0_prior import (  # noqa: E402
    canonical_json_sha256,
    sha256_file,
    write_hashed_json,
)
from xmidi_getmusic.runtime_requirements import (  # noqa: E402
    EXPECTED_WANDB_PROJECT,
    EXPECTED_CONDA_ENV,
    EXPECTED_TMUX,
    require_project_path,
    runtime_snapshot,
)

PHASE1 = REPO / "artifacts" / "phase1"
PHASE2 = REPO / "artifacts" / "phase2"


def load_hashed(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.get("artifact_sha256")
    body = dict(payload)
    body.pop("artifact_sha256", None)
    observed = canonical_json_sha256(body)
    if expected != observed:
        raise RuntimeError(f"artifact self-hash mismatch: {path}")
    return payload


def exit_zero(path: Path) -> bool:
    return path.read_text(encoding="utf-8").strip() == "0"


def junit_summary(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "tests": sum(int(item.attrib.get("tests", 0)) for item in suites),
        "failures": sum(int(item.attrib.get("failures", 0)) for item in suites),
        "errors": sum(int(item.attrib.get("errors", 0)) for item in suites),
        "skipped": sum(int(item.attrib.get("skipped", 0)) for item in suites),
    }


def checkpoint_summary(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    ema = checkpoint.get("ema")
    optimizer_steps = []
    for optimizer in checkpoint.get("optimizer_states", []):
        for item in optimizer.get("state", {}).values():
            if "step" in item:
                value = item["step"]
                optimizer_steps.append(float(value.item() if torch.is_tensor(value) else value))
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "global_step": int(checkpoint["global_step"]),
        "focal_loss_config": checkpoint.get("focal_loss_config"),
        "optimizer_step_min": min(optimizer_steps) if optimizer_steps else None,
        "optimizer_step_max": max(optimizer_steps) if optimizer_steps else None,
        "scheduler_last_epoch": int(checkpoint["lr_schedulers"][0]["last_epoch"]),
        "ema_updates": int(ema["num_updates"]) if ema else None,
        "ema_present": ema is not None,
    }


def compare_sampling() -> dict[str, Any]:
    legacy_path = PHASE2 / "sampling" / "legacy.jsonl"
    focal_path = PHASE2 / "sampling" / "focal.jsonl"
    legacy = json.loads(next(line for line in legacy_path.read_text().splitlines() if line))
    focal = json.loads(next(line for line in focal_path.read_text().splitlines() if line))
    keys = (
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
    field_checks = {key: legacy.get(key) == focal.get(key) for key in keys}
    return {
        "legacy": {"path": str(legacy_path), "sha256": sha256_file(legacy_path)},
        "focal": {"path": str(focal_path), "sha256": sha256_file(focal_path)},
        "field_checks": field_checks,
        "legacy_audit_pass": bool(legacy["structural_audit"]["all_pass"]),
        "focal_audit_pass": bool(focal["structural_audit"]["all_pass"]),
        "passed": all(field_checks.values())
        and legacy["structural_audit"]["all_pass"]
        and focal["structural_audit"]["all_pass"],
    }


def telemetry_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    times = payload["batch_wall_seconds"]
    warm_times = times[1:] if len(times) > 1 else times
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "global_step": payload["global_step"],
        "loss_config": payload["loss_config"],
        "measured_batches": len(times),
        "warm_mean_seconds": statistics.fmean(warm_times),
        "warm_median_seconds": statistics.median(warm_times),
        "peak_allocated_bytes": payload["peak_allocated_bytes"],
        "peak_reserved_bytes": payload["peak_reserved_bytes"],
    }


def main() -> None:
    os.environ.setdefault("WANDB_PROJECT", EXPECTED_WANDB_PROJECT)
    runtime = runtime_snapshot(require_tmux=True, require_wandb=True)
    output_root = require_project_path(PHASE2, label="Phase 2 output")
    phase1 = load_hashed(PHASE1 / "phase1_status.json")
    empty_ddp = load_hashed(PHASE2 / "empty_selected_ddp.json")
    sampling = compare_sampling()

    legacy_telemetry = telemetry_summary(
        REPO / "outputs" / "phase2-1gpu-legacy-100" / "phase2_telemetry" / "rank_0.json"
    )
    focal_telemetry = telemetry_summary(
        REPO / "outputs" / "phase2-1gpu-focal-100" / "phase2_telemetry" / "rank_0.json"
    )
    eight_telemetry_paths = sorted(
        (REPO / "outputs" / "phase2-8gpu-focal-100" / "phase2_telemetry").glob("rank_*.json")
    )
    eight_telemetry = [telemetry_summary(path) for path in eight_telemetry_paths]
    step_overhead = (
        focal_telemetry["warm_median_seconds"]
        / legacy_telemetry["warm_median_seconds"]
        - 1.0
    )
    memory_overhead = (
        focal_telemetry["peak_allocated_bytes"]
        / legacy_telemetry["peak_allocated_bytes"]
        - 1.0
    )

    before = load_hashed(PHASE2 / "resume_before.json")
    after = load_hashed(PHASE2 / "resume_after.json")
    resume_report = {
        "before": before,
        "after": after,
        "checks": {
            "global_step_2_to_5": before["global_step"] == 2 and after["global_step"] == 5,
            "optimizer_step_2_to_5": before["optimizer"]["step_values"] == ["2.0"]
            and after["optimizer"]["step_values"] == ["5.0"],
            "scheduler_2_to_5": before["lr_schedulers"][0]["last_epoch"] == 2
            and after["lr_schedulers"][0]["last_epoch"] == 5,
            "ema_2_to_5": before["ema"]["num_updates"] == 2
            and after["ema"]["num_updates"] == 5,
            "loss_metadata_continuous": before["focal_loss_config"]
            == after["focal_loss_config"]
            == after["hyperparameter_loss_config"],
        },
    }
    resume_report["passed"] = all(resume_report["checks"].values())
    resume_path = output_root / "checkpoint_resume_report.json"
    resume_sha = write_hashed_json(resume_path, resume_report)

    legacy_checkpoint = checkpoint_summary(
        REPO / "outputs" / "phase2-1gpu-legacy-100" / "checkpoints" / "last.ckpt"
    )
    focal_checkpoint = checkpoint_summary(
        REPO / "outputs" / "phase2-1gpu-focal-100" / "checkpoints" / "last.ckpt"
    )
    eight_checkpoint = checkpoint_summary(
        REPO / "outputs" / "phase2-8gpu-focal-100" / "checkpoints" / "last.ckpt"
    )

    diffusion_source = (REPO / "diffusion.py").read_text(encoding="utf-8")
    tree = ast.parse(diffusion_source)
    router_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "focal_subs_token_loss"
    ]
    phase1_junit = junit_summary(PHASE1 / "tests" / "phase1_junit.xml")
    phase2_integration_junit = junit_summary(PHASE2 / "tests" / "integration_junit.xml")
    v2_junit = junit_summary(PHASE2 / "tests" / "v2_regression_junit.xml")

    log_paths = {
        "legacy_100": REPO / "logs" / "phase2" / "one_gpu_legacy_100_driver.log",
        "focal_100": REPO / "logs" / "phase2" / "one_gpu_focal_100_driver.log",
        "eight_gpu_100": REPO / "logs" / "phase2" / "eight_gpu_focal_100_driver.log",
        "empty_ddp": REPO / "logs" / "phase2" / "empty_selected_ddp.log",
        "resume_stage1": REPO / "logs" / "phase2" / "resume_stage1_driver.log",
        "resume_stage2": REPO / "logs" / "phase2" / "resume_stage2_driver.log",
    }
    logs = {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in log_paths.items()
    }

    checks = {
        "phase1_prerequisite_passed": phase1["status"] == "PASSED"
        and all(phase1["checks"].values()),
        "router_has_one_integration_point": len(router_calls) == 1,
        "ng_reduction_unchanged": "nlls.sum(dim=-1) / region_sizes.clamp_min(1)"
        in diffusion_source,
        "one_gpu_legacy_exit_zero": exit_zero(PHASE2 / "one_gpu_legacy_100.exit"),
        "one_gpu_focal_exit_zero": exit_zero(PHASE2 / "one_gpu_focal_100.exit"),
        "one_gpu_legacy_100_steps": legacy_checkpoint["global_step"] == 100,
        "one_gpu_focal_100_steps": focal_checkpoint["global_step"] == 100,
        "one_gpu_checkpoints_have_exact_metadata": (
            legacy_checkpoint["focal_loss_config"]["use_focal"] is False
            and focal_checkpoint["focal_loss_config"]["use_focal"] is True
        ),
        "eight_gpu_exit_zero": exit_zero(PHASE2 / "eight_gpu_focal_100.exit"),
        "eight_gpu_100_steps": eight_checkpoint["global_step"] == 100,
        "eight_gpu_telemetry_has_eight_ranks": len(eight_telemetry) == 8
        and {item["global_step"] for item in eight_telemetry} == {100},
        "empty_selected_ddp_exit_zero": exit_zero(PHASE2 / "empty_selected_ddp.exit"),
        "empty_selected_ddp_passed": empty_ddp["passed"]
        and all(empty_ddp["checks"].values()),
        "checkpoint_resume_exit_zero": exit_zero(PHASE2 / "resume_stage1.exit")
        and exit_zero(PHASE2 / "resume_stage2.exit"),
        "checkpoint_resume_continuity": resume_report["passed"],
        "sampling_both_exit_zero": exit_zero(PHASE2 / "sampling_legacy.exit")
        and exit_zero(PHASE2 / "sampling_focal.exit"),
        "sampling_loss_mode_invariant": sampling["passed"],
        "phase2_integration_tests_passed": phase2_integration_junit["failures"] == 0
        and phase2_integration_junit["errors"] == 0,
        "phase1_tests_still_pass": phase1_junit["tests"] == 54
        and phase1_junit["failures"] == 0
        and phase1_junit["errors"] == 0,
        "v2_tests_still_pass": v2_junit["tests"] == 13
        and v2_junit["failures"] == 0
        and v2_junit["errors"] == 0,
        "step_time_overhead_within_5pct": step_overhead <= 0.05,
        "peak_allocated_memory_overhead_within_5pct": memory_overhead <= 0.05,
        "runtime_conda_exact": not EXPECTED_CONDA_ENV or runtime["conda_environment"] == EXPECTED_CONDA_ENV,
        "runtime_tmux_exact": not EXPECTED_TMUX or EXPECTED_TMUX
        in {runtime["tmux"].get("session"), runtime["tmux"].get("window")},
        "runtime_wandb_project_exact": runtime["wandb_project"]
        == EXPECTED_WANDB_PROJECT,
    }

    performance = {
        "schema_version": "bd3lms-music-v2-focal.phase2-performance-summary.v1",
        "legacy": legacy_telemetry,
        "focal": focal_telemetry,
        "eight_gpu_focal": eight_telemetry,
        "comparison": {
            "warm_median_step_overhead_fraction": step_overhead,
            "peak_allocated_memory_overhead_fraction": memory_overhead,
            "step_budget_fraction": 0.05,
            "memory_budget_fraction": 0.05,
        },
    }
    performance_path = output_root / "performance_summary.json"
    performance_sha = write_hashed_json(performance_path, performance)
    sampling_path = output_root / "sampling_regression_report.json"
    sampling_sha = write_hashed_json(sampling_path, sampling)

    regression = {
        "schema_version": "bd3lms-music-v2-focal.phase2-regression.v1",
        "phase1_junit": phase1_junit,
        "phase2_integration_junit": phase2_integration_junit,
        "v2_junit": v2_junit,
        "legacy_checkpoint": legacy_checkpoint,
        "focal_checkpoint": focal_checkpoint,
        "eight_gpu_checkpoint": eight_checkpoint,
        "source_hashes": {
            relative: sha256_file(REPO / relative)
            for relative in (
                "diffusion.py",
                "main.py",
                "configs/config.yaml",
                "scripts/train/smoke_train_xmidi_getmusic_bd3lm.sh",
                "scripts/train/train_xmidi_getmusic_bd3lm.sh",
                "scripts/eval/batch_infer_xmidi_getmusic.py",
                "xmidi_getmusic/focal_subs_loss.py",
            )
        },
    }
    regression_path = output_root / "regression_report.json"
    regression_sha = write_hashed_json(regression_path, regression)

    deliverables = {
        "performance_summary": {"path": str(performance_path), "sha256": performance_sha},
        "checkpoint_resume_report": {"path": str(resume_path), "sha256": resume_sha},
        "sampling_regression_report": {"path": str(sampling_path), "sha256": sampling_sha},
        "regression_report": {"path": str(regression_path), "sha256": regression_sha},
        "empty_selected_ddp": {
            "path": str(PHASE2 / "empty_selected_ddp.json"),
            "sha256": empty_ddp["artifact_sha256"],
        },
        "logs": logs,
    }
    passed = all(checks.values())
    status = {
        "schema_version": "bd3lms-music-v2-focal.phase-status.v1",
        "phase": 2,
        "name": "Training integration and CPU/1-GPU/8-GPU smoke",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": EXPECTED_WANDB_PROJECT,
        "status": "PASSED" if passed else "FAILED",
        "source_phase1_status_sha256": phase1["artifact_sha256"],
        "runtime": runtime,
        "checks": checks,
        "deliverables": deliverables,
        "incidents": [
            {
                "stage": "resume checkpoint inspection",
                "observed": "The first inspection failed because checkpoint hyperparameter loss was an OmegaConf DictConfig and not JSON serializable.",
                "verification": "Re-read the inspection script and checkpoint type; confirmed the training/resume design was unaffected.",
                "resolution": "Convert DictConfig with OmegaConf.to_container(resolve=True), compile-probe, and rerun inspection.",
                "design_changed": False,
            },
            {
                "stage": "resume output selection",
                "observed": "Lightning versioned the resumed last checkpoint as last-v1.ckpt instead of overwriting last.ckpt.",
                "verification": "Listed the checkpoint directory and inspected both files; last.ckpt remained step 2 while last-v1.ckpt was step 5.",
                "resolution": "Use the actual versioned resume output for the after-summary; no file deletion or overwrite assumption.",
                "design_changed": False,
            },
        ],
        "limitations": [
            "The 100-step smoke uses the tiny backbone and is a stability/performance gate, not a quality result.",
            "The short T=16 sampling run proves sampler invariance and structural safety, not music quality.",
            "Pilot hyperparameter comparison starts only in Phase 3.",
        ],
        "next_phase": "Phase 3 may start only if status is PASSED.",
    }
    status_path = output_root / "phase2_status.json"
    status_sha = write_hashed_json(status_path, status)
    print(json.dumps({"status": status["status"], "status_path": str(status_path), "status_sha256": status_sha, "checks": checks}, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
