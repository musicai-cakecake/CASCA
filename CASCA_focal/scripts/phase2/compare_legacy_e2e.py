#!/usr/bin/env python3
"""Compare original V2 and integrated-disabled one-step checkpoints."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Iterator

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from xmidi_getmusic.phase0_prior import sha256_file, write_hashed_json  # noqa: E402
from xmidi_getmusic.runtime_requirements import require_project_path  # noqa: E402


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


def compare(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> dict[str, Any]:
    keys_exact = set(left) == set(right)
    if not keys_exact:
        return {
            "keys_exact": False,
            "missing": sorted(set(left) - set(right)),
            "extra": sorted(set(right) - set(left)),
        }
    exact = 0
    max_abs = 0.0
    difference_sq = 0.0
    reference_sq = 0.0
    for key in sorted(left):
        first = left[key].detach().cpu()
        second = right[key].detach().cpu()
        if first.shape != second.shape or first.dtype != second.dtype:
            return {"keys_exact": True, "shape_dtype_exact": False, "mismatch": key}
        exact += int(torch.equal(first, second))
        difference = first.to(torch.float64) - second.to(torch.float64)
        if difference.numel():
            max_abs = max(max_abs, float(difference.abs().max()))
            difference_sq += float(difference.square().sum())
            reference_sq += float(first.to(torch.float64).square().sum())
    return {
        "keys_exact": True,
        "shape_dtype_exact": True,
        "tensor_count": len(left),
        "exact_tensor_count": exact,
        "all_elementwise_exact": exact == len(left),
        "max_absolute_difference": max_abs,
        "relative_l2_error": math.sqrt(difference_sq) / max(math.sqrt(reference_sq), 1e-300),
    }


def main() -> None:
    base = REPO / "outputs" / "phase2-legacy-e2e-regression-v2"
    v2_path = base / "v2" / "checkpoints" / "last.ckpt"
    integrated_path = base / "focal" / "checkpoints" / "last.ckpt"
    output = require_project_path(
        REPO / "artifacts" / "phase2" / "legacy_e2e_regression.json",
        label="legacy E2E regression",
    )
    v2 = torch.load(v2_path, map_location="cpu", weights_only=False, mmap=True)
    integrated = torch.load(
        integrated_path, map_location="cpu", weights_only=False, mmap=True
    )
    state = compare(dict(v2["state_dict"]), dict(integrated["state_dict"]))
    optimizer = compare(
        dict(tensor_items(v2["optimizer_states"])),
        dict(tensor_items(integrated["optimizer_states"])),
    )
    scheduler = compare(
        dict(tensor_items(v2["lr_schedulers"])),
        dict(tensor_items(integrated["lr_schedulers"])),
    )
    ema = compare(
        dict(tensor_items(v2["ema"])),
        dict(tensor_items(integrated["ema"])),
    )
    checks = {
        "both_global_step_one": v2["global_step"] == integrated["global_step"] == 1,
        "model_state_exact": state.get("all_elementwise_exact", False),
        "optimizer_state_exact": optimizer.get("all_elementwise_exact", False),
        "scheduler_tensor_state_exact": scheduler.get("all_elementwise_exact", False),
        "ema_state_exact": ema.get("all_elementwise_exact", False),
        "integrated_metadata_is_legacy": integrated["focal_loss_config"]["use_focal"] is False,
    }
    payload = {
        "schema_version": "bd3lms-music-v2-focal.phase2-legacy-e2e.v1",
        "protocol": {
            "model": "xmidi_getmusic_tiny",
            "seed": 1,
            "precision": "bf16-mixed",
            "optimizer_steps": 1,
            "learning_rate": 3e-5,
            "ema": 0.9999,
            "batch_size": 1,
        },
        "v2_checkpoint": {"path": str(v2_path), "sha256": sha256_file(v2_path)},
        "integrated_checkpoint": {
            "path": str(integrated_path),
            "sha256": sha256_file(integrated_path),
        },
        "model_state": state,
        "optimizer_state": optimizer,
        "scheduler_state": scheduler,
        "ema_state": ema,
        "checks": checks,
        "passed": all(checks.values()),
    }
    write_hashed_json(output, payload)
    print(json.dumps({"output": str(output), "checks": checks}, indent=2))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
