#!/usr/bin/env python3
"""Write a compact, hashed checkpoint continuity summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from xmidi_getmusic.phase0_prior import sha256_file, write_hashed_json  # noqa: E402
from xmidi_getmusic.runtime_requirements import require_project_path  # noqa: E402


def scalar(value: Any) -> int | float | str | None:
    if torch.is_tensor(value):
        if value.numel() != 1:
            return f"tensor{tuple(value.shape)}"
        return value.item()
    if isinstance(value, (int, float, str)) or value is None:
        return value
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint_path = args.checkpoint.resolve()
    output = require_project_path(args.output, label="checkpoint summary")
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    optimizer_states = checkpoint.get("optimizer_states", [])
    optimizer_steps: list[int | float | str | None] = []
    optimizer_param_groups = []
    optimizer_state_entries = 0
    for optimizer in optimizer_states:
        state = optimizer.get("state", {})
        optimizer_state_entries += len(state)
        for item in state.values():
            if "step" in item:
                optimizer_steps.append(scalar(item["step"]))
        for group in optimizer.get("param_groups", []):
            optimizer_param_groups.append(
                {
                    key: scalar(value)
                    for key, value in group.items()
                    if key != "params"
                }
            )
    ema = checkpoint.get("ema")
    hyperparameter_loss_config = checkpoint.get("hyper_parameters", {}).get(
        "config", {}
    ).get("loss")
    if OmegaConf.is_config(hyperparameter_loss_config):
        hyperparameter_loss_config = OmegaConf.to_container(
            hyperparameter_loss_config, resolve=True
        )
    payload = {
        "schema_version": "bd3lms-music-v2-focal.phase2-checkpoint-summary.v1",
        "checkpoint": {
            "path": str(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
            "sha256": sha256_file(checkpoint_path),
        },
        "global_step": int(checkpoint["global_step"]),
        "epoch": int(checkpoint["epoch"]),
        "focal_loss_config": checkpoint.get("focal_loss_config"),
        "hyperparameter_loss_config": hyperparameter_loss_config,
        "optimizer": {
            "count": len(optimizer_states),
            "state_entries": optimizer_state_entries,
            "step_values": sorted({str(value) for value in optimizer_steps}),
            "param_groups": optimizer_param_groups,
        },
        "lr_schedulers": checkpoint.get("lr_schedulers", []),
        "ema": (
            {
                "present": True,
                "decay": float(ema["decay"]),
                "num_updates": int(ema["num_updates"]),
                "shadow_parameter_count": len(ema["shadow_params"]),
            }
            if ema is not None
            else {"present": False}
        ),
    }
    write_hashed_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
