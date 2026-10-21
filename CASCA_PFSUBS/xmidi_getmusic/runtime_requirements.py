"""Fail-fast runtime identity and project-boundary checks for V2 focal loss."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CONDA_ENV = os.environ.get("BD3LM_EXPECTED_CONDA_ENV", "")
EXPECTED_PYTHON = Path(os.environ.get("BD3LM_EXPECTED_PYTHON", sys.executable))
EXPECTED_TMUX = os.environ.get("BD3LM_EXPECTED_TMUX", "")
EXPECTED_WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "CASCA_focal")


def _resolved(path: str | os.PathLike[str]) -> Path:
    return Path(path).expanduser().resolve()


def require_conda_environment() -> dict[str, str]:
    env_name = os.environ.get("CONDA_DEFAULT_ENV")
    executable = _resolved(sys.executable)
    expected_executable = _resolved(EXPECTED_PYTHON)
    errors: list[str] = []
    if EXPECTED_CONDA_ENV and env_name != EXPECTED_CONDA_ENV:
        errors.append(
            f"CONDA_DEFAULT_ENV={env_name!r}, expected {EXPECTED_CONDA_ENV!r}"
        )
    if executable != expected_executable:
        errors.append(f"sys.executable={executable}, expected {expected_executable}")
    if errors:
        raise RuntimeError("Invalid Conda runtime: " + "; ".join(errors))
    return {"conda_environment": env_name, "python_executable": str(executable)}


def tmux_identity() -> dict[str, str | None]:
    pane = os.environ.get("TMUX_PANE")
    if not os.environ.get("TMUX") and not pane:
        return {"session": None, "window": None, "pane": pane}
    target_args = ["-t", pane] if pane else []

    def query(fmt: str) -> str | None:
        result = subprocess.run(
            ["tmux", "display-message", "-p", *target_args, fmt],
            check=False,
            capture_output=True,
            text=True,
        )
        value = result.stdout.strip()
        return value or None

    return {
        "session": query("#S"),
        "window": query("#W"),
        "pane": pane or query("#P"),
    }


def require_tmux_identity(
    expected: str = EXPECTED_TMUX,
) -> dict[str, str | None]:
    identity = tmux_identity()
    if expected and expected not in {identity["session"], identity["window"]}:
        raise RuntimeError(
            "Task must run inside tmux session/window "
            f"{expected!r}; current={identity}"
        )
    return identity


def require_wandb_project(project: str | None = None) -> str:
    actual = project if project is not None else os.environ.get("WANDB_PROJECT", EXPECTED_WANDB_PROJECT)
    if actual != EXPECTED_WANDB_PROJECT:
        raise RuntimeError(
            f"W&B project must be {EXPECTED_WANDB_PROJECT!r}; got {actual!r}"
        )
    return actual


def require_project_path(
    path: str | os.PathLike[str],
    *,
    label: str,
    allow_project_root: bool = False,
) -> Path:
    resolved = _resolved(path)
    if resolved == PROJECT_ROOT and allow_project_root:
        return resolved
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise RuntimeError(
            f"{label} must remain under {PROJECT_ROOT}; got {resolved}"
        ) from error
    if resolved == PROJECT_ROOT:
        raise RuntimeError(f"{label} must not overwrite the project root")
    return resolved


def require_project_paths(paths: Iterable[tuple[str, str | os.PathLike[str]]]) -> None:
    for label, path in paths:
        require_project_path(path, label=label)


def runtime_snapshot(*, require_tmux: bool, require_wandb: bool) -> dict[str, object]:
    conda = require_conda_environment()
    tmux = require_tmux_identity() if require_tmux else tmux_identity()
    wandb_project = (
        require_wandb_project() if require_wandb else os.environ.get("WANDB_PROJECT")
    )
    try:
        import torch

        torch_info: dict[str, object] = {
            "torch_version": torch.__version__,
            "cuda_runtime_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cudnn_version": torch.backends.cudnn.version(),
        }
        if torch.cuda.is_available():
            torch_info["gpu_names"] = [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ]
    except ImportError:
        torch_info = {"torch_importable": False}
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        **conda,
        "tmux": tmux,
        "wandb_project": wandb_project,
        "cwd": str(Path.cwd().resolve()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        **torch_info,
    }


def write_runtime_snapshot(
    path: str | os.PathLike[str], *, require_tmux: bool, require_wandb: bool
) -> dict[str, object]:
    output = require_project_path(path, label="runtime snapshot")
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshot = runtime_snapshot(require_tmux=require_tmux, require_wandb=require_wandb)
    output.write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return snapshot
