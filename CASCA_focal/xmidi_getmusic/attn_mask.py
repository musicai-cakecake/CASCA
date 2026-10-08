"""Per-sample row-adaptive BD3 attention masks for V2."""

from __future__ import annotations

from typing import Optional

import torch

from .constants import NUM_COLS


def _base_block_diff_mask(
    seqlen: int,
    block_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.bool,
) -> torch.Tensor:
    """Build the static noisy/clean BD3 mask with shape ``(2L, 2L)``."""
    length2 = 2 * seqlen
    index = torch.arange(length2, device=device)
    q_idx = index.view(length2, 1)
    kv_idx = index.view(1, length2)
    q_clean = q_idx >= seqlen
    kv_clean = kv_idx >= seqlen
    q_block = torch.where(
        q_clean, (q_idx - seqlen) // block_size, q_idx // block_size
    )
    kv_block = torch.where(
        kv_clean, (kv_idx - seqlen) // block_size, kv_idx // block_size
    )
    block_diagonal = (q_block == kv_block) & (q_clean == kv_clean)
    offset_causal = (q_block > kv_block) & kv_clean & (~q_clean)
    clean_causal = (q_block >= kv_block) & kv_clean & q_clean
    return (block_diagonal | offset_causal | clean_causal).to(dtype)


def _row_for_position(
    seqlen: int,
    num_cols: int,
    device: torch.device,
) -> torch.Tensor:
    position = torch.arange(2 * seqlen, device=device)
    local_position = position.remainder(seqlen)
    return torch.div(local_position, num_cols, rounding_mode="floor")


def build_xmidi_attn_mask(
    seqlen: int,
    block_size: int,
    row_unk_mask: torch.Tensor,
    num_cols: int = NUM_COLS,
    base_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Combine the BD3 mask with per-sample inactive-row semantics.

    Active queries see only active keys/values. An inactive query receives one
    self edge solely to keep SDPA numerically defined; the model multiplies its
    attention update by the matching query-activity mask.
    """
    if row_unk_mask.dim() != 2:
        raise ValueError(
            f"row_unk_mask must be (B, num_rows), got {tuple(row_unk_mask.shape)}"
        )
    batch_size, num_rows = row_unk_mask.shape
    if seqlen != num_rows * num_cols:
        raise ValueError(
            f"seqlen={seqlen} does not equal num_rows*num_cols="
            f"{num_rows * num_cols}"
        )
    length2 = 2 * seqlen
    device = row_unk_mask.device
    if base_mask is None:
        base_mask = _base_block_diff_mask(seqlen, block_size, device)
    if base_mask.shape != (length2, length2):
        raise ValueError(
            f"base_mask shape {tuple(base_mask.shape)} != {(length2, length2)}"
        )
    base_mask = base_mask.to(device=device, dtype=torch.bool)

    rows = _row_for_position(seqlen, num_cols, device)
    inactive = row_unk_mask.bool().gather(
        1, rows.unsqueeze(0).expand(batch_size, -1)
    )
    active = ~inactive
    q_active = active[:, None, :, None]
    kv_active = active[:, None, None, :]
    mask = base_mask[None, None] & q_active & kv_active

    # Safe self-only fallback for inactive queries. It has no semantic effect
    # because DIT zeros their attention updates after SDPA.
    diagonal = torch.eye(length2, dtype=torch.bool, device=device)[None, None]
    mask = mask | ((~q_active) & diagonal)
    return mask


def query_active_from_rows(
    row_unk_mask: torch.Tensor,
    positions: torch.Tensor,
    num_cols: int = NUM_COLS,
) -> torch.Tensor:
    """Return ``(B,Q)`` activity for absolute flattened positions."""
    rows = torch.div(positions, num_cols, rounding_mode="floor")
    if rows.numel() and int(rows.max()) >= row_unk_mask.shape[1]:
        raise ValueError("Position exceeds row_unk_mask geometry")
    return ~row_unk_mask.bool().gather(
        1, rows.unsqueeze(0).expand(row_unk_mask.shape[0], -1)
    )
