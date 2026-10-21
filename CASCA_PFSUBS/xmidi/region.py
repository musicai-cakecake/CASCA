"""Region-aware masks for XMIDI training.

Given a flat sequence ``x`` of shape ``(B, 5120)`` produced by row-major
flattening of the original ``(B, 10, 512)`` grid, we build:

1. ``region_mask: (B, 5120)``  — 1 where the position is *eligible to be
   diffusion-masked and contributes to the loss*, 0 where it must stay clean
   throughout the diffusion process. Positions that are 0:

      a) entire source rows (rows 0..3)                           §R&D
      b) the prompt prefix of every row (cols 0..15)              §R&D
      c) any row that is *entirely* <UNK>                         §R&D §<UNK>(1)
      d) any individual <UNK> position inside an otherwise valid row §R&D §<UNK>(2)

   Note (c) and (d) collapse together once we mark every <UNK> token as 0:
   if all 512 cols of a row are <UNK>, the whole row is 0 automatically.

2. ``row_unk_mask: (B, 10)`` — 1 if a row is entirely <UNK> (so that
   attention can drop that row's tokens completely from key/value pool).

The two masks are produced together because they share the same scan over
the input.
"""

from __future__ import annotations

from typing import Tuple

import torch

from .tokenizer import (
    UNK_ID,
    NUM_ROWS,
    NUM_COLS,
    NUM_SOURCE_ROWS,
    NUM_PROMPT_COLS,
    SEQ_LEN,
)


def build_region_masks(
    x: torch.Tensor,
    unk_id: int = UNK_ID,
    num_rows: int = NUM_ROWS,
    num_cols: int = NUM_COLS,
    num_source_rows: int = NUM_SOURCE_ROWS,
    num_prompt_cols: int = NUM_PROMPT_COLS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build region masks for a row-major flattened XMIDI batch.

    Parameters
    ----------
    x: (B, num_rows * num_cols) int tensor  (row-major flatten)

    Returns
    -------
    region_mask: (B, num_rows * num_cols) float tensor  (0 or 1)
        1 means "loss / mask candidate", 0 means "fixed context".
    row_unk_mask: (B, num_rows) bool tensor
        True if the entire row consists of <UNK>.
    """
    if x.dim() != 2 or x.shape[1] != num_rows * num_cols:
        raise ValueError(
            f"Expected x of shape (B, {num_rows * num_cols}), got {tuple(x.shape)}"
        )

    B = x.shape[0]
    grid = x.view(B, num_rows, num_cols)             # (B, 10, 512)

    # ----- (a) source-row gate: rows in [0, num_source_rows) are fixed -----
    is_target_row = torch.zeros(num_rows, dtype=torch.bool, device=x.device)
    is_target_row[num_source_rows:] = True            # rows 4..9 -> True
    target_row_mask = is_target_row.view(1, num_rows, 1).expand(B, num_rows, num_cols)

    # ----- (b) prompt-column gate: cols in [0, num_prompt_cols) are fixed --
    is_content_col = torch.zeros(num_cols, dtype=torch.bool, device=x.device)
    is_content_col[num_prompt_cols:] = True           # cols 16..511 -> True
    content_col_mask = is_content_col.view(1, 1, num_cols).expand(B, num_rows, num_cols)

    # ----- (c)+(d) <UNK> gate: any <UNK> position is fixed ----------------
    not_unk_mask = (grid != unk_id)                   # (B, 10, 512), True on real tokens

    # final region mask: must satisfy ALL three conditions
    region_mask_grid = target_row_mask & content_col_mask & not_unk_mask  # (B, 10, 512)

    # entirely-UNK row indicator
    row_unk_mask = (grid == unk_id).all(dim=-1)       # (B, 10) bool

    region_mask = region_mask_grid.view(B, num_rows * num_cols).to(x.dtype)
    return region_mask, row_unk_mask


def attention_active_mask(
    row_unk_mask: torch.Tensor,
    num_cols: int = NUM_COLS,
) -> torch.Tensor:
    """Per-token "is this token active for attention" indicator.

    A token is *inactive* iff it lies on a fully-<UNK> row, in which case it
    must be skipped both as query and as key/value (R&D §"局部 attention").

    Parameters
    ----------
    row_unk_mask: (B, num_rows) bool

    Returns
    -------
    active: (B, num_rows * num_cols) bool
    """
    B, num_rows = row_unk_mask.shape
    active = (~row_unk_mask).view(B, num_rows, 1).expand(B, num_rows, num_cols)
    return active.contiguous().view(B, num_rows * num_cols)


def expected_loss_positions(
    num_rows: int = NUM_ROWS,
    num_cols: int = NUM_COLS,
    num_source_rows: int = NUM_SOURCE_ROWS,
    num_prompt_cols: int = NUM_PROMPT_COLS,
) -> int:
    """How many positions are *potentially* learnable, ignoring <UNK>.

    Used for sanity checks. With defaults: (10-4) * (512-16) = 6 * 496 = 2976.
    """
    return (num_rows - num_source_rows) * (num_cols - num_prompt_cols)
