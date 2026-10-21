"""Region and active-track masks for the V2 16x512 music grid."""

from __future__ import annotations

from typing import Tuple

import torch

from .constants import (
    NUM_COLS,
    NUM_PROMPT_COLS,
    NUM_ROWS,
    NUM_SOURCE_ROWS,
    UNK_ID,
)


def build_region_masks(
    x: torch.Tensor,
    unk_id: int = UNK_ID,
    num_rows: int = NUM_ROWS,
    num_cols: int = NUM_COLS,
    num_source_rows: int = NUM_SOURCE_ROWS,
    num_prompt_cols: int = NUM_PROMPT_COLS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the generative-cell mask and all-``<UNK>`` row mask.

    ``x`` must be row-major with shape ``(B, num_rows * num_cols)``.
    The generative region contains non-``<UNK>`` cells in target rows after
    the prompt columns. Returned masks are boolean.
    """
    expected_length = num_rows * num_cols
    if x.dim() != 2 or x.shape[1] != expected_length:
        raise ValueError(
            f"Expected x of shape (B, {expected_length}), got {tuple(x.shape)}"
        )
    if not 0 <= num_source_rows <= num_rows:
        raise ValueError("num_source_rows must lie in [0, num_rows]")
    if not 0 <= num_prompt_cols <= num_cols:
        raise ValueError("num_prompt_cols must lie in [0, num_cols]")

    batch_size = x.shape[0]
    grid = x.reshape(batch_size, num_rows, num_cols)
    rows = torch.arange(num_rows, device=x.device).view(1, num_rows, 1)
    cols = torch.arange(num_cols, device=x.device).view(1, 1, num_cols)
    region = (
        (rows >= num_source_rows)
        & (cols >= num_prompt_cols)
        & (grid != unk_id)
    )
    row_unk = (grid == unk_id).all(dim=-1)
    return region.reshape(batch_size, expected_length), row_unk


def attention_active_mask(
    row_unk_mask: torch.Tensor,
    num_cols: int = NUM_COLS,
) -> torch.Tensor:
    """Expand ``(B,R)`` inactive-row flags to ``(B,R*C)`` active flags."""
    if row_unk_mask.dim() != 2:
        raise ValueError(
            f"row_unk_mask must have shape (B, R), got {tuple(row_unk_mask.shape)}"
        )
    batch_size, num_rows = row_unk_mask.shape
    return (
        (~row_unk_mask.bool())
        .view(batch_size, num_rows, 1)
        .expand(batch_size, num_rows, num_cols)
        .reshape(batch_size, num_rows * num_cols)
    )


def expected_loss_positions(
    num_rows: int = NUM_ROWS,
    num_cols: int = NUM_COLS,
    num_source_rows: int = NUM_SOURCE_ROWS,
    num_prompt_cols: int = NUM_PROMPT_COLS,
) -> int:
    """Maximum number of V2 target-content cells (12 * 496 = 5952)."""
    return (num_rows - num_source_rows) * (num_cols - num_prompt_cols)
