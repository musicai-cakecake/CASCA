#!/usr/bin/env python3
"""Compare V2 and focal-project one-step checkpoints and critical source."""

from __future__ import annotations

import ast
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterator

import torch

REPO = Path(__file__).resolve().parents[2]
V2_ROOT = REPO.parent / "CASCA_base"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from xmidi_getmusic.phase0_prior import sha256_file, write_hashed_json  # noqa: E402
from xmidi_getmusic.runtime_requirements import require_project_path  # noqa: E402

CRITICAL_FUNCTIONS = {
    "diffusion.py": (
        "Diffusion.q_xt",
        "Diffusion._selected_subs_log_probs",
        "Diffusion._forward_pass_diffusion",
        "Diffusion._loss",
    ),
    "models/dit.py": (
        "DDiTFinalLayer.selected_log_probs",
        "DIT.forward",
    ),
    "noise_schedule.py": (
        "LogLinearNoise.compute_loss_scaling_and_move_chance",
    ),
    "xmidi_getmusic/region.py": (
        "build_region_masks",
        "attention_active_mask",
        "expected_loss_positions",
    ),
}


def walk_functions(tree: ast.AST) -> dict[str, ast.AST]:
    functions: dict[str, ast.AST] = {}

    def visit(node: ast.AST, prefix: str = "") -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, f"{prefix}{child.name}.")
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions[f"{prefix}{child.name}"] = child
            else:
                visit(child, prefix)

    visit(tree)
    return functions


def critical_ast(root: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for relative, names in CRITICAL_FUNCTIONS.items():
        path = root / relative
        functions = walk_functions(ast.parse(path.read_text(encoding="utf-8")))
        result[relative] = {}
        for name in names:
            if name not in functions:
                raise KeyError(f"{path}: missing {name}")
            result[relative][name] = ast.dump(
                functions[name], annotate_fields=True, include_attributes=False
            )
    return result


def tensor_items(value: Any, prefix: str = "") -> Iterator[tuple[str, torch.Tensor]]:
    if torch.is_tensor(value):
        yield prefix, value
    elif isinstance(value, dict):
        for key in sorted(value, key=lambda item: str(item)):
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from tensor_items(value[key], child)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from tensor_items(item, f"{prefix}[{index}]")


def compare_tensor_maps(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
) -> dict[str, Any]:
    if set(left) != set(right):
        return {
            "keys_exact": False,
            "missing_in_focal": sorted(set(left) - set(right)),
            "extra_in_focal": sorted(set(right) - set(left)),
        }
    exact = 0
    max_abs = 0.0
    diff_sq = 0.0
    reference_sq = 0.0
    finite_exact = True
    shape_dtype_exact = True
    first_mismatch = None
    for key in sorted(left):
        first = left[key].detach().cpu()
        second = right[key].detach().cpu()
        if first.shape != second.shape or first.dtype != second.dtype:
            shape_dtype_exact = False
            first_mismatch = key
            break
        finite_exact &= torch.equal(torch.isfinite(first), torch.isfinite(second))
        exact += int(torch.equal(first, second))
        difference = first.to(torch.float64) - second.to(torch.float64)
        if difference.numel():
            max_abs = max(max_abs, float(difference.abs().max()))
            diff_sq += float(difference.square().sum())
            reference_sq += float(first.to(torch.float64).square().sum())
    return {
        "keys_exact": True,
        "shape_dtype_exact": shape_dtype_exact,
        "first_shape_dtype_mismatch": first_mismatch,
        "tensor_count": len(left),
        "exact_tensor_count": exact,
        "all_tensors_elementwise_exact": exact == len(left),
        "finite_masks_exact": bool(finite_exact),
        "max_absolute_difference": max_abs,
        "relative_l2_error": math.sqrt(diff_sq) / max(math.sqrt(reference_sq), 1e-300),
    }


def main() -> None:
    base = REPO / "artifacts" / "phase0" / "one_step_regression"
    v2_path = base / "v2" / "checkpoints" / "last.ckpt"
    focal_path = base / "focal" / "checkpoints" / "last.ckpt"
    out = require_project_path(
        REPO / "artifacts" / "phase0" / "one_step_regression_report.json",
        label="one-step report",
    )
    v2 = torch.load(v2_path, map_location="cpu", weights_only=False, mmap=True)
    focal = torch.load(focal_path, map_location="cpu", weights_only=False, mmap=True)
    states = compare_tensor_maps(dict(v2["state_dict"]), dict(focal["state_dict"]))
    optimizers = compare_tensor_maps(
        dict(tensor_items(v2.get("optimizer_states", []))),
        dict(tensor_items(focal.get("optimizer_states", []))),
    )
    schedulers = compare_tensor_maps(
        dict(tensor_items(v2.get("lr_schedulers", []))),
        dict(tensor_items(focal.get("lr_schedulers", []))),
    )
    v2_ast = critical_ast(V2_ROOT)
    focal_ast = critical_ast(REPO)
    ast_checks = {
        path: {name: v2_ast[path][name] == focal_ast[path][name] for name in names}
        for path, names in CRITICAL_FUNCTIONS.items()
    }
    source_file_checks = {
        path: {
            "v2_sha256": sha256_file(V2_ROOT / path),
            "focal_sha256": sha256_file(REPO / path),
            "byte_identical": sha256_file(V2_ROOT / path) == sha256_file(REPO / path),
        }
        for path in CRITICAL_FUNCTIONS
    }
    checks = {
        "both_global_step_one": v2.get("global_step") == focal.get("global_step") == 1,
        "updated_parameters_elementwise_exact": states.get("all_tensors_elementwise_exact", False),
        "optimizer_moments_elementwise_exact": optimizers.get("all_tensors_elementwise_exact", False),
        "scheduler_tensors_elementwise_exact": schedulers.get("all_tensors_elementwise_exact", False),
        "critical_ast_exact": all(value for item in ast_checks.values() for value in item.values()),
        "critical_source_files_byte_identical": all(
            item["byte_identical"] for item in source_file_checks.values()
        ),
    }
    payload = {
        "schema_version": "bd3lms-music-v2-focal.phase0-one-step-regression.v1",
        "project": "BD3LMs_Music_V2_focal_loss",
        "protocol": {
            "seed": 1,
            "model": "xmidi_getmusic_tiny",
            "batch_size": 1,
            "optimizer_steps": 1,
            "precision": "bf16-mixed",
            "learning_rate": 3e-5,
            "attention_backend": "flex",
            "ema": 0.0,
            "gpu": 3,
        },
        "inputs": {
            "v2_checkpoint": {"path": str(v2_path), "sha256": sha256_file(v2_path)},
            "focal_checkpoint": {"path": str(focal_path), "sha256": sha256_file(focal_path)},
        },
        "updated_parameter_state": states,
        "optimizer_state": optimizers,
        "scheduler_state": schedulers,
        "critical_ast_exact": ast_checks,
        "critical_source_files": source_file_checks,
        "gradient_evidence": (
            "The one-step AdamW moments and updated model parameters are elementwise "
            "exact under identical initialization, data order, RNG, precision, and LR."
        ),
        "checks": checks,
        "passed": all(checks.values()),
    }
    write_hashed_json(out, payload)
    print(json.dumps({"out": str(out), "checks": checks, "passed": payload["passed"]}, indent=2))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
