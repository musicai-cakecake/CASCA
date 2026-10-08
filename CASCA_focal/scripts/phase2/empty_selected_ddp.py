#!/usr/bin/env python3
"""8-rank DDP smoke through the real tied selected projection.

Rank 0 intentionally has no selected target positions. Other ranks execute the
real DDiTFinalLayer.selected_log_probs path before the integrated focal router.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from models.dit import DDiTFinalLayer  # noqa: E402
from xmidi_getmusic.constants import PAD_ID  # noqa: E402
from xmidi_getmusic.focal_subs_loss import (  # noqa: E402
    FocalSubsLossConfig,
    focal_subs_token_loss,
)
from xmidi_getmusic.phase0_prior import write_hashed_json  # noqa: E402
from xmidi_getmusic.runtime_requirements import (  # noqa: E402
    EXPECTED_WANDB_PROJECT,
    require_project_path,
    runtime_snapshot,
)

VOCAB_SIZE = PAD_ID + 2
EXCLUDED_TOKEN_ID = PAD_ID + 1
HIDDEN_SIZE = 8
SEQUENCE_LENGTH = 2


class RealSelectedProjectionModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = torch.nn.Parameter(
            torch.randn(1, SEQUENCE_LENGTH, HIDDEN_SIZE) * 0.01
        )
        self.embedding = torch.nn.Parameter(
            torch.randn(VOCAB_SIZE, HIDDEN_SIZE) * 0.01
        )
        self.output_layer = DDiTFinalLayer(
            HIDDEN_SIZE,
            VOCAB_SIZE,
            cond_dim=4,
            adaLN=False,
            tie_word_embeddings=True,
        )
        self.config = FocalSubsLossConfig(
            use_focal=True,
            gamma=0.5,
            alpha_pad=0.86,
            alpha_event=1.29,
        )

    def forward(self, empty_selected: bool) -> torch.Tensor:
        selected = torch.tensor(
            [[not empty_selected, False]],
            dtype=torch.bool,
            device=self.hidden.device,
        )
        targets = torch.tensor(
            [[PAD_ID, 17]], dtype=torch.long, device=self.hidden.device
        )
        log_p = self.output_layer.selected_log_probs(
            self.hidden,
            None,
            targets,
            selected,
            self.embedding,
            excluded_token_id=EXCLUDED_TOKEN_ID,
            chunk_size=1,
        )
        loss_scale = torch.tensor(
            [[-2.0]], dtype=torch.float32, device=self.hidden.device
        )
        return focal_subs_token_loss(
            log_p, targets, loss_scale, selected, self.config
        ).sum()


def tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = require_project_path(args.output, label="empty-selected DDP report")
    os.environ.setdefault("WANDB_PROJECT", EXPECTED_WANDB_PROJECT)

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(20260730)

    module = RealSelectedProjectionModule().to(device)
    ddp = torch.nn.parallel.DistributedDataParallel(
        module, device_ids=[local_rank], output_device=local_rank
    )
    optimizer = torch.optim.AdamW(ddp.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    empty_selected = rank == 0
    loss = ddp(empty_selected)
    loss.backward()

    gradients = [
        parameter.grad.detach().reshape(-1)
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    gradient_vector = torch.cat(gradients)
    optimizer.step()
    parameter_vector = torch.cat(
        [parameter.detach().reshape(-1) for parameter in module.parameters()]
    )

    local = {
        "rank": rank,
        "local_rank": local_rank,
        "empty_selected": empty_selected,
        "loss": float(loss.detach().cpu()),
        "gradient_norm": float(gradient_vector.norm().cpu()),
        "gradient_finite": bool(torch.isfinite(gradient_vector).all()),
        "gradient_sha256": tensor_sha256(gradient_vector),
        "updated_parameters_finite": bool(torch.isfinite(parameter_vector).all()),
        "updated_parameters_sha256": tensor_sha256(parameter_vector),
        "projection_class": type(module.output_layer).__name__,
        "tied_projection": module.output_layer.tie_word_embeddings,
        "output_layer_linear_is_none": module.output_layer.linear is None,
    }
    gathered: list[dict[str, object] | None] = [None] * world_size
    dist.all_gather_object(gathered, local)
    dist.barrier()
    if rank == 0:
        records = [item for item in gathered if item is not None]
        checks = {
            "world_size_is_eight": world_size == 8,
            "rank_zero_empty_selected": bool(records[0]["empty_selected"]),
            "other_ranks_nonempty": all(
                not bool(item["empty_selected"]) for item in records[1:]
            ),
            "uses_real_ddit_final_layer": all(
                item["projection_class"] == "DDiTFinalLayer"
                for item in records
            ),
            "uses_tied_projection": all(
                bool(item["tied_projection"])
                and bool(item["output_layer_linear_is_none"])
                for item in records
            ),
            "all_gradients_finite": all(
                bool(item["gradient_finite"]) for item in records
            ),
            "all_updated_parameters_finite": all(
                bool(item["updated_parameters_finite"]) for item in records
            ),
            "ddp_gradients_identical": len(
                {str(item["gradient_sha256"]) for item in records}
            )
            == 1,
            "ddp_updated_parameters_identical": len(
                {str(item["updated_parameters_sha256"]) for item in records}
            )
            == 1,
        }
        payload = {
            "schema_version": "bd3lms-music-v2-focal.phase2-empty-ddp.v2",
            "runtime": runtime_snapshot(require_tmux=True, require_wandb=True),
            "world_size": world_size,
            "loss_config": module.config.to_dict(),
            "projection": {
                "class": "models.dit.DDiTFinalLayer",
                "tie_word_embeddings": True,
                "vocab_size": VOCAB_SIZE,
                "hidden_size": HIDDEN_SIZE,
                "excluded_token_id": EXCLUDED_TOKEN_ID,
            },
            "records": records,
            "checks": checks,
            "passed": all(checks.values()),
        }
        write_hashed_json(output, payload)
        print(json.dumps({"output": str(output), "checks": checks}, indent=2))
        if not payload["passed"]:
            raise SystemExit(1)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
