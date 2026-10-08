#!/usr/bin/env python3
"""Generate Phase 1 alpha, config, and numerical-validation artifacts."""

from __future__ import annotations

import ast
import inspect
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from xmidi_getmusic.constants import PAD_ID  # noqa: E402
from xmidi_getmusic.focal_subs_loss import (  # noqa: E402
    ALPHA_NORMALIZATION_NONE,
    ALPHA_NORMALIZATION_TRAIN_PRIOR,
    FocalSubsLossConfig,
    focal_subs_token_loss,
    resolve_group_alpha,
    stable_one_minus_probability,
)
from xmidi_getmusic.phase0_prior import (  # noqa: E402
    canonical_json_sha256,
    sha256_file,
    write_hashed_json,
)
from xmidi_getmusic.runtime_requirements import (  # noqa: E402
    EXPECTED_WANDB_PROJECT,
    require_project_path,
    runtime_snapshot,
)

PHASE0 = REPO / "artifacts" / "phase0"
PHASE1 = REPO / "artifacts" / "phase1"
MODULE_PATH = REPO / "xmidi_getmusic" / "focal_subs_loss.py"


def load_hashed(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.get("artifact_sha256")
    body = dict(payload)
    body.pop("artifact_sha256", None)
    observed = canonical_json_sha256(body)
    if expected != observed:
        raise RuntimeError(f"artifact self-hash mismatch: {path}")
    return payload


def tensor_case(dtype: torch.dtype, device: torch.device) -> dict[str, Any]:
    selected = torch.tensor(
        [[True, True, False], [True, False, True]],
        dtype=torch.bool,
        device=device,
    )
    targets = torch.tensor(
        [[PAD_ID, 17, 123], [42, PAD_ID, PAD_ID]],
        dtype=torch.long,
        device=device,
    )
    source_values = torch.tensor(
        [[-1e-4, -0.3, 0.0], [-2.0, 0.0, -8.0]],
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    scale = torch.tensor(
        [[-2.0], [-5.0]], dtype=torch.float32, device=device
    )

    false_input = source_values.detach().clone().requires_grad_(True)
    false_expected_input = source_values.detach().clone().requires_grad_(True)
    false_expected = scale * false_expected_input
    false_actual = focal_subs_token_loss(
        false_input,
        targets,
        scale,
        selected,
        FocalSubsLossConfig(use_focal=False),
    )
    false_gradient = torch.autograd.grad(false_actual.sum(), false_input)[0]
    false_expected_gradient = torch.autograd.grad(
        false_expected.sum(), false_expected_input
    )[0]

    gamma_zero_input = source_values.detach().clone().requires_grad_(True)
    gamma_zero_expected_input = source_values.detach().clone().requires_grad_(True)
    gamma_zero_expected = scale * gamma_zero_expected_input
    gamma_zero_actual = focal_subs_token_loss(
        gamma_zero_input,
        targets,
        scale,
        selected,
        FocalSubsLossConfig(
            use_focal=True,
            gamma=0.0,
            alpha_pad=1.0,
            alpha_event=1.0,
        ),
    )
    gamma_zero_gradient = torch.autograd.grad(
        gamma_zero_actual.sum(), gamma_zero_input
    )[0]
    gamma_zero_expected_gradient = torch.autograd.grad(
        gamma_zero_expected.sum(), gamma_zero_expected_input
    )[0]

    focal_input = source_values.detach().clone().requires_grad_(True)
    focal_config = FocalSubsLossConfig(
        use_focal=True,
        gamma=2.0,
        alpha_pad=0.86,
        alpha_event=1.29,
        alpha_normalization=ALPHA_NORMALIZATION_NONE,
    )
    focal_output = focal_subs_token_loss(
        focal_input, targets, scale, selected, focal_config
    )
    focal_gradient = torch.autograd.grad(focal_output.sum(), focal_input)[0]

    quantized_fp64 = focal_input.detach().to(torch.float64)
    scale_fp64 = scale.to(torch.float64)
    reference = scale_fp64 * quantized_fp64
    one_minus = -torch.expm1(quantized_fp64)
    alpha = torch.where(
        targets.eq(PAD_ID),
        torch.tensor(0.86, dtype=torch.float64, device=device),
        torch.tensor(1.29, dtype=torch.float64, device=device),
    )
    reference_focal = reference * alpha * one_minus.pow(2.0)
    reference = torch.where(selected, reference_focal, reference)
    absolute_error = (focal_output.to(torch.float64) - reference).abs()

    return {
        "dtype": str(dtype).replace("torch.", ""),
        "device": str(device),
        "false_forward_elementwise_exact": torch.equal(
            false_actual, false_expected
        ),
        "false_gradient_elementwise_exact": torch.equal(
            false_gradient, false_expected_gradient
        ),
        "gamma_zero_forward_elementwise_exact": torch.equal(
            gamma_zero_actual, gamma_zero_expected
        ),
        "gamma_zero_gradient_elementwise_exact": torch.equal(
            gamma_zero_gradient, gamma_zero_expected_gradient
        ),
        "focal_output_finite": bool(torch.isfinite(focal_output).all()),
        "focal_gradient_finite": bool(torch.isfinite(focal_gradient).all()),
        "focal_max_absolute_error_vs_quantized_fp64_reference": float(
            absolute_error.max().cpu()
        ),
        "output_dtype": str(focal_output.dtype).replace("torch.", ""),
        "gradient_dtype": str(focal_gradient.dtype).replace("torch.", ""),
    }


def empty_selected_check(device: torch.device) -> dict[str, Any]:
    gamma_cases: dict[str, Any] = {}
    for gamma in (0.0, 0.5, 1.0, 2.0):
        parameter = torch.nn.Parameter(
            torch.tensor([2.0], dtype=torch.float64, device=device)
        )
        log_p = parameter.sum() * 0.0 + torch.zeros(
            (2, 3), dtype=torch.float64, device=device
        )
        targets = torch.zeros((2, 3), dtype=torch.long, device=device)
        selected = torch.zeros((2, 3), dtype=torch.bool, device=device)
        scale = torch.full((2, 1), -2.0, dtype=torch.float64, device=device)
        output = focal_subs_token_loss(
            log_p,
            targets,
            scale,
            selected,
            FocalSubsLossConfig(use_focal=True, gamma=gamma),
        )
        output.sum().backward()
        gamma_cases[str(gamma)] = {
            "output_exact_zero": bool(
                torch.equal(output, torch.zeros_like(output))
            ),
            "output_requires_grad": output.requires_grad,
            "parameter_gradient_present": parameter.grad is not None,
            "parameter_gradient_exact_zero": bool(
                parameter.grad is not None
                and torch.equal(parameter.grad, torch.zeros_like(parameter.grad))
            ),
            "parameter_gradient_finite": bool(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
            ),
        }
    return {"gamma_cases": gamma_cases}


def static_module_check() -> dict[str, Any]:
    import xmidi_getmusic.focal_subs_loss as module

    source = inspect.getsource(module)
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    forbidden_calls = {"softmax", "log_softmax", "one_hot", "linear", "embedding"}
    return {
        "module_path": str(MODULE_PATH),
        "module_sha256": sha256_file(MODULE_PATH),
        "imports_models": "models" in imported,
        "imports_diffusion": "diffusion" in imported,
        "forbidden_projection_calls": sorted(forbidden_calls & called),
        "token_loss_parameters": list(
            inspect.signature(focal_subs_token_loss).parameters
        ),
        "contains_vocab_size_parameter": (
            "vocab_size" in inspect.signature(focal_subs_token_loss).parameters
        ),
    }


def main() -> None:
    os.environ.setdefault("WANDB_PROJECT", EXPECTED_WANDB_PROJECT)
    runtime = runtime_snapshot(require_tmux=True, require_wandb=True)
    output_root = require_project_path(PHASE1, label="Phase 1 output")
    output_root.mkdir(parents=True, exist_ok=True)

    phase0_status = load_hashed(PHASE0 / "phase0_status.json")
    if phase0_status["status"] != "PASSED" or not all(
        phase0_status["checks"].values()
    ):
        raise RuntimeError("Phase 0 is not a fully passed prerequisite")
    phase0_prior = load_hashed(PHASE0 / "pad_prior.json")
    train = phase0_prior["splits"]["train"]
    train_pad_probability = train["pad"] / train["generation_cells"]
    train_event_probability = train["non_pad"] / train["generation_cells"]
    prior_balanced_ratio = train["pad"] / train["non_pad"]

    ratios = (
        ("alpha_neutral", 1.0),
        ("mild_balanced", 1.5),
        ("medium_balanced", 2.0),
        ("prior_balanced", prior_balanced_ratio),
    )
    rows = []
    for name, ratio in ratios:
        resolved = resolve_group_alpha(train_pad_probability, ratio)
        row = {"name": name, **resolved.to_dict()}
        row["expectation_absolute_error"] = abs(resolved.expected_alpha - 1.0)
        rows.append(row)
    alpha_table = {
        "schema_version": "bd3lms-music-v2-focal.phase1-alpha-table.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": EXPECTED_WANDB_PROJECT,
        "source_phase0_status_sha256": phase0_status["artifact_sha256"],
        "source_pad_prior_sha256": phase0_prior["artifact_sha256"],
        "train_counts": {
            "generation_cells": train["generation_cells"],
            "pad": train["pad"],
            "non_pad": train["non_pad"],
            "pad_probability": train_pad_probability,
            "event_probability": train_event_probability,
        },
        "normalization_formula": {
            "alpha_pad": "1 / (q_pad + r * q_event)",
            "alpha_event": "r / (q_pad + r * q_event)",
            "constraint": "q_pad * alpha_pad + q_event * alpha_event = 1",
        },
        "rows": rows,
        "recommended_initial_candidate": {
            "name": "mild_balanced",
            "gamma": 2.0,
            "event_to_pad_ratio": 1.5,
            "alpha_pad": rows[1]["alpha_pad"],
            "alpha_event": rows[1]["alpha_event"],
        },
    }
    alpha_path = output_root / "alpha_table.json"
    alpha_sha = write_hashed_json(alpha_path, alpha_table)

    config_schema = {
        "schema_version": "bd3lms-music-v2-focal.phase1-config-schema.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": EXPECTED_WANDB_PROJECT,
        "module": str(MODULE_PATH),
        "module_sha256": sha256_file(MODULE_PATH),
        "python_type": "FocalSubsLossConfig",
        "phase2_hydra_namespace": "loss",
        "fields": {
            "use_focal": {
                "type": "bool",
                "default": False,
                "strict": True,
            },
            "gamma": {
                "type": "finite non-negative float",
                "default": 2.0,
                "validated_when": "use_focal=true",
            },
            "alpha_pad": {
                "type": "finite non-negative float",
                "default": 1.0,
                "validated_when": "use_focal=true",
            },
            "alpha_event": {
                "type": "finite non-negative float",
                "default": 1.0,
                "validated_when": "use_focal=true",
            },
            "pad_token_id": {
                "type": "int equal to V2 PAD_ID",
                "default": PAD_ID,
                "allowed": [PAD_ID],
                "validated_when": "use_focal=true",
            },
            "alpha_normalization": {
                "type": "enum",
                "default": ALPHA_NORMALIZATION_NONE,
                "allowed": [ALPHA_NORMALIZATION_NONE],
                "validated_when": "use_focal=true",
                "resolver_provenance_label": ALPHA_NORMALIZATION_TRAIN_PRIOR,
                "note": (
                    "train_prior_expected_one is emitted by resolve_group_alpha "
                    "as provenance and is not accepted as a runtime transform"
                ),
            },
        },
        "legacy_contract": (
            "use_focal=false returns the direct V2 expression "
            "loss_scale * log_p_target"
        ),
        "focal_contract": (
            "use_focal=true applies alpha(target) * (1-p_target)^gamma "
            "to the direct V2 token loss only at selected positions"
        ),
        "reduction_contract": "No reduction; Phase 2 retains V2 N_G reduction",
        "projection_contract": "Consumes selected log probabilities; no logits or vocabulary projection",
    }
    schema_path = output_root / "focal_config_schema.json"
    schema_sha = write_hashed_json(schema_path, config_schema)

    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    cases = []
    for device in devices:
        cases.extend(
            [
                tensor_case(torch.float64, device),
                tensor_case(torch.float32, device),
                tensor_case(torch.bfloat16, device),
            ]
        )
    empty_checks = {
        str(device): empty_selected_check(device) for device in devices
    }
    near_one = torch.tensor([-1e-12, -1e-8, -1e-4], dtype=torch.float64)
    stable = stable_one_minus_probability(near_one)
    stable_reference = -torch.expm1(near_one)
    static = static_module_check()
    tolerances = {
        "float64": 1e-14,
        "float32": 5e-6,
        "bfloat16": 2e-3,
    }
    case_checks = []
    for case in cases:
        case_checks.append(
            case["false_forward_elementwise_exact"]
            and case["false_gradient_elementwise_exact"]
            and case["gamma_zero_forward_elementwise_exact"]
            and case["gamma_zero_gradient_elementwise_exact"]
            and case["focal_output_finite"]
            and case["focal_gradient_finite"]
            and case["focal_max_absolute_error_vs_quantized_fp64_reference"]
            <= tolerances[case["dtype"]]
        )
    checks = {
        "phase0_prerequisite_passed": True,
        "all_dtype_device_cases_passed": all(case_checks),
        "cuda_bfloat16_executed": any(
            case["device"].startswith("cuda") and case["dtype"] == "bfloat16"
            for case in cases
        ),
        "all_empty_selected_cases_passed": all(
            all(
                all(case.values())
                for case in item["gamma_cases"].values()
            )
            for item in empty_checks.values()
        ),
        "stable_expm1_float64_exact": torch.equal(stable, stable_reference),
        "no_models_import": not static["imports_models"],
        "no_diffusion_import": not static["imports_diffusion"],
        "no_projection_calls": not static["forbidden_projection_calls"],
        "no_vocab_size_parameter": not static["contains_vocab_size_parameter"],
        "alpha_expectations_unit": all(
            row["expectation_absolute_error"] <= 2e-16 for row in rows
        ),
    }
    numerical = {
        "schema_version": "bd3lms-music-v2-focal.phase1-numerical-validation.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": EXPECTED_WANDB_PROJECT,
        "runtime": runtime,
        "source_phase0_status_sha256": phase0_status["artifact_sha256"],
        "module": static,
        "tolerances": tolerances,
        "dtype_device_cases": cases,
        "empty_selected": empty_checks,
        "stable_one_minus_probability": {
            "inputs": near_one.tolist(),
            "actual": stable.tolist(),
            "reference": stable_reference.tolist(),
            "elementwise_exact": torch.equal(stable, stable_reference),
        },
        "checks": checks,
        "passed": all(checks.values()),
        "artifacts": {
            "alpha_table": {"path": str(alpha_path), "sha256": alpha_sha},
            "config_schema": {"path": str(schema_path), "sha256": schema_sha},
        },
    }
    numerical_path = output_root / "numerical_validation_report.json"
    numerical_sha = write_hashed_json(numerical_path, numerical)
    print(
        json.dumps(
            {
                "alpha_table": {"path": str(alpha_path), "sha256": alpha_sha},
                "config_schema": {"path": str(schema_path), "sha256": schema_sha},
                "numerical_validation": {
                    "path": str(numerical_path),
                    "sha256": numerical_sha,
                },
                "checks": checks,
                "passed": numerical["passed"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not numerical["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
