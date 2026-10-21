#!/usr/bin/env python3
"""Validate Phase 1 deliverables and publish the authoritative status."""

from __future__ import annotations

import json
import os
import sys
import xml.etree.ElementTree as ET
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
    EXPECTED_WANDB_PROJECT,
    EXPECTED_CONDA_ENV,
    EXPECTED_TMUX,
    require_project_path,
    runtime_snapshot,
)

PHASE0 = REPO / "artifacts" / "phase0"
PHASE1 = REPO / "artifacts" / "phase1"
MODULE_PATH = REPO / "xmidi_getmusic" / "focal_subs_loss.py"
TEST_PATH = REPO / "tests" / "test_focal_subs_loss.py"
GENERATOR_PATH = REPO / "scripts" / "phase1" / "generate_phase1_artifacts.py"


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
    tests = sum(int(suite.attrib.get("tests", 0)) for suite in suites)
    failures = sum(int(suite.attrib.get("failures", 0)) for suite in suites)
    errors = sum(int(suite.attrib.get("errors", 0)) for suite in suites)
    skipped = sum(int(suite.attrib.get("skipped", 0)) for suite in suites)
    test_names = sorted(
        testcase.attrib.get("name", "")
        for testcase in root.iter("testcase")
    )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "tests": tests,
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
        "passed": tests - failures - errors - skipped,
        "test_names": test_names,
    }


def main() -> None:
    os.environ.setdefault("WANDB_PROJECT", EXPECTED_WANDB_PROJECT)
    runtime = runtime_snapshot(require_tmux=True, require_wandb=True)
    output_root = require_project_path(PHASE1, label="Phase 1 output")

    phase0 = load_hashed(PHASE0 / "phase0_status.json")
    boundary = load_hashed(PHASE0 / "project_boundary_audit.json")
    alpha = load_hashed(output_root / "alpha_table.json")
    schema = load_hashed(output_root / "focal_config_schema.json")
    numerical = load_hashed(output_root / "numerical_validation_report.json")
    phase1_junit = junit_summary(output_root / "tests" / "phase1_junit.xml")
    v2_junit = junit_summary(output_root / "tests" / "v2_regression_junit.xml")

    current_core_hashes = {
        relative: sha256_file(REPO / relative)
        for relative in (
            "diffusion.py",
            "models/dit.py",
            "noise_schedule.py",
            "xmidi_getmusic/constants.py",
            "xmidi_getmusic/region.py",
        )
    }
    phase0_core = boundary["core_v2_source_identity"]
    core_frozen = all(
        current_core_hashes[relative]
        == phase0_core[relative]["focal_sha256"]
        == phase0_core[relative]["v2_sha256"]
        for relative in current_core_hashes
    )
    module_hash = sha256_file(MODULE_PATH)
    source_contract = {
        "module": {"path": str(MODULE_PATH), "sha256": module_hash},
        "tests": {"path": str(TEST_PATH), "sha256": sha256_file(TEST_PATH)},
        "artifact_generator": {
            "path": str(GENERATOR_PATH),
            "sha256": sha256_file(GENERATOR_PATH),
        },
        "frozen_v2_core": current_core_hashes,
        "diffusion_imports_focal_module": (
            "focal_subs_loss" in (REPO / "diffusion.py").read_text(encoding="utf-8")
        ),
        "phase2_training_integration_deferred": True,
    }

    required_tests = {
        "test_false_branch_is_elementwise_and_gradient_exact_to_v2",
        "test_gamma_zero_unit_alpha_is_elementwise_and_gradient_exact_to_v2",
        "test_float64_focal_matches_manual_multiclass_reference",
        "test_empty_selected_set_preserves_graph_connected_zero",
        "test_fractional_gamma_does_not_pollute_unselected_gradients",
        "test_extreme_target_probabilities_have_finite_forward_and_backward",
        "test_false_branch_has_no_focal_specific_runtime_validation",
        "test_focal_branch_passes_float64_gradcheck",
        "test_cpu_bfloat16_forward_and_gradient_are_finite",
        "test_cuda_bfloat16_forward_and_gradient_are_finite",
        "test_module_has_no_vocabulary_projection_or_dense_logits_dependency",
    }
    test_names = set(phase1_junit["test_names"])
    base_test_names = {name.split("[", 1)[0] for name in test_names}
    alpha_rows = alpha["rows"]
    checks = {
        "phase0_prerequisite_passed": (
            phase0["status"] == "PASSED" and all(phase0["checks"].values())
        ),
        "phase1_pytest_exit_zero": exit_zero(
            output_root / "tests" / "phase1_pytest.exit"
        ),
        "phase1_pytest_54_passed": (
            phase1_junit["tests"] == 54
            and phase1_junit["passed"] == 54
            and phase1_junit["failures"] == 0
            and phase1_junit["errors"] == 0
            and phase1_junit["skipped"] == 0
        ),
        "required_math_gradient_bf16_tests_present": required_tests.issubset(
            base_test_names
        ),
        "v2_regression_exit_zero": exit_zero(
            output_root / "tests" / "v2_regression.exit"
        ),
        "v2_regression_13_passed": (
            v2_junit["tests"] == 13
            and v2_junit["passed"] == 13
            and v2_junit["failures"] == 0
            and v2_junit["errors"] == 0
        ),
        "artifact_generator_exit_zero": exit_zero(
            output_root / "generate_artifacts.exit"
        ),
        "numerical_validation_passed": (
            numerical["passed"] and all(numerical["checks"].values())
        ),
        "legacy_false_forward_gradient_exact_all_cases": all(
            case["false_forward_elementwise_exact"]
            and case["false_gradient_elementwise_exact"]
            for case in numerical["dtype_device_cases"]
        ),
        "gamma_zero_forward_gradient_exact_all_cases": all(
            case["gamma_zero_forward_elementwise_exact"]
            and case["gamma_zero_gradient_elementwise_exact"]
            for case in numerical["dtype_device_cases"]
        ),
        "all_focal_outputs_and_gradients_finite": all(
            case["focal_output_finite"] and case["focal_gradient_finite"]
            for case in numerical["dtype_device_cases"]
        ),
        "cpu_and_cuda_bfloat16_executed": {
            (case["device"], case["dtype"])
            for case in numerical["dtype_device_cases"]
        }.issuperset({("cpu", "bfloat16"), ("cuda", "bfloat16")}),
        "empty_selected_graph_connected_cpu_cuda": all(
            all(
                all(case.values())
                for case in item["gamma_cases"].values()
            )
            for item in numerical["empty_selected"].values()
        ),
        "stable_expm1_exact": numerical["checks"][
            "stable_expm1_float64_exact"
        ],
        "no_dense_projection_dependency": (
            numerical["checks"]["no_models_import"]
            and numerical["checks"]["no_diffusion_import"]
            and numerical["checks"]["no_projection_calls"]
            and numerical["checks"]["no_vocab_size_parameter"]
        ),
        "module_hash_matches_schema_and_numerical_report": (
            module_hash == schema["module_sha256"]
            and module_hash == numerical["module"]["module_sha256"]
        ),
        "alpha_table_uses_phase0_train_prior": (
            alpha["source_pad_prior_sha256"]
            == phase0["deliverables"]["pad_prior"]["sha256"]
            and alpha["train_counts"]["pad"] == 217_999_140
            and alpha["train_counts"]["non_pad"] == 105_239_560
        ),
        "alpha_expectation_one_all_candidates": all(
            row["expected_alpha"] == 1.0
            and row["expectation_absolute_error"] == 0.0
            for row in alpha_rows
        ),
        "config_schema_strict_legacy_default": (
            schema["fields"]["use_focal"]["default"] is False
            and schema["fields"]["use_focal"]["strict"] is True
            and schema["fields"]["pad_token_id"]["default"] == 605
            and schema["fields"]["pad_token_id"]["allowed"] == [605]
            and schema["fields"]["alpha_normalization"]["allowed"]
            == ["none"]
            and all(
                schema["fields"][name]["validated_when"]
                == "use_focal=true"
                for name in (
                    "gamma",
                    "alpha_pad",
                    "alpha_event",
                    "pad_token_id",
                    "alpha_normalization",
                )
            )
        ),
        "v2_core_sources_unchanged": core_frozen,
        "phase2_integration_not_started": not source_contract[
            "diffusion_imports_focal_module"
        ],
        "runtime_conda_exact": not EXPECTED_CONDA_ENV or runtime["conda_environment"]
        == EXPECTED_CONDA_ENV,
        "runtime_tmux_exact": not EXPECTED_TMUX or EXPECTED_TMUX
        in {runtime["tmux"].get("session"), runtime["tmux"].get("window")},
        "runtime_wandb_project_exact": runtime["wandb_project"]
        == EXPECTED_WANDB_PROJECT,
    }

    test_report = {
        "schema_version": "bd3lms-music-v2-focal.phase1-test-report.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": EXPECTED_WANDB_PROJECT,
        "source_phase0_status_sha256": phase0["artifact_sha256"],
        "phase1_pytest": phase1_junit,
        "v2_regression_pytest": v2_junit,
        "numerical_validation": {
            "path": str(output_root / "numerical_validation_report.json"),
            "sha256": numerical["artifact_sha256"],
            "passed": numerical["passed"],
        },
        "source_contract": source_contract,
        "required_tests": sorted(required_tests),
        "required_tests_present": required_tests.issubset(base_test_names),
        "passed": all(checks.values()),
    }
    test_report_path = output_root / "unit_gradient_test_report.json"
    test_report_sha = write_hashed_json(test_report_path, test_report)

    deliverables = {
        "focal_loss_module": {
            "path": str(MODULE_PATH),
            "sha256": module_hash,
        },
        "focal_config_schema": {
            "path": str(output_root / "focal_config_schema.json"),
            "sha256": schema["artifact_sha256"],
        },
        "alpha_table": {
            "path": str(output_root / "alpha_table.json"),
            "sha256": alpha["artifact_sha256"],
        },
        "numerical_validation_report": {
            "path": str(output_root / "numerical_validation_report.json"),
            "sha256": numerical["artifact_sha256"],
        },
        "unit_gradient_test_report": {
            "path": str(test_report_path),
            "sha256": test_report_sha,
        },
    }
    passed = all(checks.values())
    status = {
        "schema_version": "bd3lms-music-v2-focal.phase-status.v1",
        "phase": 1,
        "name": "Standalone PAD-aware multiclass Focal SUBS loss module",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": EXPECTED_WANDB_PROJECT,
        "status": "PASSED" if passed else "FAILED",
        "source_phase0_status_sha256": phase0["artifact_sha256"],
        "runtime": runtime,
        "checks": checks,
        "deliverables": deliverables,
        "recommended_initial_candidate": alpha[
            "recommended_initial_candidate"
        ],
        "incidents": [
            {
                "stage": "first numerical artifact generation",
                "observed": (
                    "FP32 focal absolute error was 2.37970882466243e-06, "
                    "above an initially chosen 1e-06 absolute tolerance."
                ),
                "verification": (
                    "Re-read the generator and module, confirmed the intended "
                    "code was present, and confirmed legacy/gamma-zero paths "
                    "were still elementwise exact."
                ),
                "resolution": (
                    "Set the FP32 focal-vs-FP64 absolute tolerance to 5e-06 "
                    "for outputs up to approximately 34.4; reran the complete "
                    "CPU/CUDA validation."
                ),
                "design_changed": False,
            },
            {
                "stage": "initial CPU unit run",
                "observed": (
                    "One test incorrectly expected zero gradient with respect "
                    "to an artificial leaf at unselected log-prob positions."
                ),
                "verification": (
                    "Re-read selected projection: unselected zeros have no "
                    "model dependence, while the token-loss legacy graph is "
                    "intentionally preserved for DDP empty selection."
                ),
                "resolution": (
                    "Corrected the test to require legacy graph gradient at "
                    "the artificial input and graph-connected zero upstream."
                ),
                "design_changed": False,
            },
            {
                "stage": "first Phase 1 finalization",
                "observed": (
                    "The finalizer expected pytest parameter IDs float32/float64, "
                    "while JUnit emitted dtype0/dtype1. All 44 tests had passed."
                ),
                "verification": (
                    "Re-read the JUnit XML and finalizer source and confirmed "
                    "the failure was only required-test name matching."
                ),
                "resolution": (
                    "Required coverage now matches parameterized test base "
                    "function names; reran finalization without rerunning or "
                    "weakening any test."
                ),
                "design_changed": False,
            },
            {
                "stage": "independent post-finalization review",
                "observed": (
                    "The original focal implementation evaluated 0**gamma at "
                    "unselected log_p=0 positions; for 0<gamma<1 backward could "
                    "form 0*Inf. The runtime normalization label and PAD-ID "
                    "override contract were also too permissive."
                ),
                "verification": (
                    "Re-read the loss module, selected projection, tests, schema "
                    "and status. Confirmed the risk was real and not a failed edit."
                ),
                "resolution": (
                    "Compute focal powers only on selected positions; return the "
                    "legacy multiplication before focal validation; require PAD_ID "
                    "605; treat train-prior normalization only as resolver provenance; "
                    "add fractional-gamma, extreme-probability and strict legacy tests."
                ),
                "design_changed": True,
            },
        ],
        "limitations": [
            "Phase 1 is standalone and does not modify diffusion.py or Hydra training config.",
            "No model training, inference-quality experiment, or sampler change is part of Phase 1.",
            "Hydra/CLI routing and performance smoke tests are explicitly deferred to Phase 2.",
        ],
        "next_phase": "Phase 2 may start only if status is PASSED.",
    }
    status_path = output_root / "phase1_status.json"
    status_sha = write_hashed_json(status_path, status)
    print(
        json.dumps(
            {
                "status": status["status"],
                "status_path": str(status_path),
                "status_sha256": status_sha,
                "checks": checks,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
