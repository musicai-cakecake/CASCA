"""Phase 2-only Lightning callback for comparable step/memory telemetry."""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import lightning as L
import torch
from omegaconf import OmegaConf

from .phase0_prior import canonical_json_sha256, sha256_file, write_hashed_json
from .runtime_requirements import require_project_path, runtime_snapshot


class Phase2PerformanceCallback(L.Callback):
    """Measure synchronized train-batch wall time and peak allocated memory."""

    def __init__(self, output_dir: str):
        super().__init__()
        self.output_dir = str(
            require_project_path(output_dir, label="Phase 2 telemetry directory")
        )
        self._batch_start: float | None = None
        self._batch_times: list[float] = []
        self._loss_values: list[float] = []
        self._gradient_norms: list[float] = []
        self._nonfinite_loss_count = 0
        self._nonfinite_gradient_count = 0
        self._started_at_utc: str | None = None

    def on_train_start(self, trainer, pl_module) -> None:
        del trainer, pl_module
        self._batch_times = []
        self._loss_values = []
        self._gradient_norms = []
        self._nonfinite_loss_count = 0
        self._nonfinite_gradient_count = 0
        self._started_at_utc = datetime.now(timezone.utc).isoformat()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

    def on_train_batch_start(
        self, trainer, pl_module, batch, batch_idx
    ) -> None:
        del trainer, pl_module, batch, batch_idx
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._batch_start = time.perf_counter()

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ) -> None:
        del trainer, pl_module, batch, batch_idx
        if self._batch_start is None:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._batch_times.append(time.perf_counter() - self._batch_start)
        self._batch_start = None
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
        if torch.is_tensor(loss) and loss.numel() == 1:
            value = float(loss.detach().float().cpu())
            self._loss_values.append(value)
            self._nonfinite_loss_count += int(not torch.isfinite(loss).item())

    def on_after_backward(self, trainer, pl_module) -> None:
        del trainer
        squared_norm = torch.zeros((), device=pl_module.device, dtype=torch.float32)
        finite = True
        for parameter in pl_module.parameters():
            if parameter.grad is None:
                continue
            gradient = parameter.grad.detach().float()
            finite = finite and bool(torch.isfinite(gradient).all().item())
            squared_norm = squared_norm + gradient.square().sum()
        norm = squared_norm.sqrt()
        self._gradient_norms.append(float(norm.cpu()))
        self._nonfinite_gradient_count += int(not finite or not torch.isfinite(norm).item())

    def on_fit_end(self, trainer, pl_module) -> None:
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        rank = int(getattr(trainer, "global_rank", 0))
        times = list(self._batch_times)
        payload = {
            "schema_version": "bd3lms-music-v2-focal.phase2-performance.v1",
            "pid": os.getpid(),
            "global_rank": rank,
            "world_size": int(getattr(trainer, "world_size", 1)),
            "global_step": int(getattr(trainer, "global_step", 0)),
            "loss_config": pl_module.focal_loss_config.to_dict(),
            "measured_train_batches": len(times),
            "batch_wall_seconds": times,
            "mean_batch_wall_seconds": (
                sum(times) / len(times) if times else None
            ),
            "loss_values": self._loss_values,
            "gradient_norms": self._gradient_norms,
            "nonfinite_loss_count": self._nonfinite_loss_count,
            "nonfinite_gradient_count": self._nonfinite_gradient_count,
            "all_losses_finite": self._nonfinite_loss_count == 0,
            "all_gradients_finite": self._nonfinite_gradient_count == 0,
            "peak_allocated_bytes": (
                int(torch.cuda.max_memory_allocated())
                if torch.cuda.is_available()
                else 0
            ),
            "peak_reserved_bytes": (
                int(torch.cuda.max_memory_reserved())
                if torch.cuda.is_available()
                else 0
            ),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
        path = output_dir / f"rank_{rank}.json"
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if rank != 0:
            return

        root = Path(__file__).resolve().parents[1]
        source_paths = (
            "diffusion.py",
            "main.py",
            "models/dit.py",
            "configs/config.yaml",
            "configs/callbacks/checkpoint_monitor.yaml",
            "scripts/train/smoke_train_xmidi_getmusic_bd3lm.sh",
            "scripts/train/train_xmidi_getmusic_bd3lm.sh",
            "xmidi_getmusic/focal_subs_loss.py",
            "xmidi_getmusic/phase2_performance.py",
        )
        source_hashes = {
            relative: sha256_file(root / relative) for relative in source_paths
        }
        resolved_config = OmegaConf.to_container(
            pl_module.config, resolve=True, enum_to_str=True
        )
        config_hash = canonical_json_sha256(resolved_config)
        evidence = {}
        for name, evidence_path in (
            ("phase0_identity", root / "artifacts/phase0/data_checkpoint_identity.json"),
            ("phase0_prior", root / "artifacts/phase0/pad_prior.json"),
            ("phase1_status", root / "artifacts/phase1/phase1_status.json"),
        ):
            evidence[name] = {
                "path": str(evidence_path),
                "sha256": sha256_file(evidence_path),
            }
        resume_path = pl_module.config.checkpointing.resume_ckpt_path
        checkpoint_input = None
        if (pl_module.config.checkpointing.resume_from_ckpt
                and resume_path):
            resume_path = Path(str(resume_path)).resolve()
            if resume_path.is_file():
                checkpoint_input = {
                    "path": str(resume_path),
                    "bytes": resume_path.stat().st_size,
                    "sha256": sha256_file(resume_path),
                }
        manifest = {
            "schema_version": "bd3lms-music-v2-focal.phase2-run-manifest.v1",
            "status": "COMPLETED",
            "started_at_utc": self._started_at_utc,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(),
            "command": sys.argv,
            "cwd": os.getcwd(),
            "runtime": runtime_snapshot(require_tmux=True, require_wandb=True),
            "run_identity": {
                "wandb_project": os.environ.get("WANDB_PROJECT"),
                "wandb_run_name": (
                    os.environ.get("WANDB_RUN_NAME")
                    or (resolved_config.get("wandb") or {}).get("name")
                ),
                "wandb_run_id": (
                    os.environ.get("WANDB_RUN_ID")
                    or (resolved_config.get("wandb") or {}).get("id")
                ),
                "wandb_run_tag": os.environ.get("WANDB_RUN_TAG"),
            },
            "resolved_config": resolved_config,
            "resolved_config_sha256": config_hash,
            "source_hashes": source_hashes,
            "source_tree_sha256": canonical_json_sha256(source_hashes),
            "evidence": evidence,
            "checkpoint_input": checkpoint_input,
            "telemetry": {
                "rank_files": [f"rank_{index}.json" for index in range(int(getattr(trainer, "world_size", 1)))],
                "all_rank0_losses_finite": self._nonfinite_loss_count == 0,
                "all_rank0_gradients_finite": self._nonfinite_gradient_count == 0,
            },
        }
        write_hashed_json(output_dir / "run_manifest.json", manifest)
