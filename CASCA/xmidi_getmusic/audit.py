"""Structural invariant audit for BD3LMs-Music-V2 samples."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

import torch

from .constants import (
    DICT_LEN,
    MASK_ID,
    NUM_COLS,
    NUM_PROMPT_COLS,
    NUM_ROWS,
    NUM_SOURCE_ROWS,
    UNK_ID,
)


def audit_xmidi_batch(
    output_ids: torch.Tensor,
    initial_ids: torch.Tensor,
    generation_mask: torch.Tensor,
    row_unk_mask: Optional[torch.Tensor],
    runtime_trace: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Audit V2 structural/runtime invariants and return one record per item.

    V2.1 extends the original seven checks with fixed-condition invariance so
    arbitrary DAW partial-generation masks cannot overwrite user-authored
    ordinary target tokens.
    """
    expected_length = NUM_ROWS * NUM_COLS
    if output_ids.dim() != 2 or output_ids.shape[1] != expected_length:
        raise ValueError(
            f"output_ids must be (B, {expected_length}), got {tuple(output_ids.shape)}"
        )
    if initial_ids.shape != output_ids.shape or generation_mask.shape != output_ids.shape:
        raise ValueError("initial_ids and generation_mask must match output_ids")
    if row_unk_mask is None:
        row_unk_mask = torch.zeros(
            output_ids.shape[0], NUM_ROWS, dtype=torch.bool, device=output_ids.device
        )
    if row_unk_mask.shape != (output_ids.shape[0], NUM_ROWS):
        raise ValueError("row_unk_mask has invalid shape")

    output = output_ids.reshape(-1, NUM_ROWS, NUM_COLS)
    initial = initial_ids.reshape_as(output)
    generate = generation_mask.reshape_as(output).bool()
    runtime_trace = dict(runtime_trace or {})
    runtime_ok = all(
        bool(runtime_trace.get(name, False))
        for name in (
            "attention_mask_valid",
            "current_query_contract",
            "cache_consistent",
            "absolute_rope",
        )
    )

    records: List[Dict[str, Any]] = []
    for index in range(output.shape[0]):
        sample = output[index]
        prior = initial[index]
        selected = generate[index]
        inactive = row_unk_mask[index].bool()
        fixed_unk = (prior == UNK_ID) & (~selected)

        checks = {
            "shape_and_vocabulary": bool(
                sample.shape == (NUM_ROWS, NUM_COLS)
                and not sample.dtype.is_floating_point
                and torch.all((sample >= 0) & (sample <= UNK_ID)).item()
            ),
            "source_invariance": bool(
                torch.equal(
                    sample[:NUM_SOURCE_ROWS], prior[:NUM_SOURCE_ROWS]
                )
            ),
            "prompt_invariance": bool(
                torch.equal(
                    sample[NUM_SOURCE_ROWS:, :NUM_PROMPT_COLS],
                    prior[NUM_SOURCE_ROWS:, :NUM_PROMPT_COLS],
                )
            ),
            "cell_unk_invariance": bool(
                torch.all(sample[fixed_unk] == UNK_ID).item()
            ),
            "fixed_condition_invariance": bool(
                torch.equal(sample[~selected], prior[~selected])
            ),
            "inactive_row_invariance": bool(
                torch.all(sample[inactive] == UNK_ID).item()
            ),
            "diffusion_completion": bool(
                not torch.any(sample == MASK_ID).item()
                and torch.all(
                    (sample[selected] >= 0) & (sample[selected] < DICT_LEN)
                ).item()
            ),
            "attention_cache_consistency": runtime_ok,
        }
        failures = [name for name, passed in checks.items() if not passed]
        records.append(
            {
                "sample_index": index,
                "checks": checks,
                "all_pass": not failures,
                "failures": failures,
                "generated_positions": int(selected.sum().item()),
                "inactive_rows": int(inactive.sum().item()),
                "runtime": runtime_trace,
            }
        )
    return records
