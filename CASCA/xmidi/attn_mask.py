"""Adaptive attention mask for the XMIDI variant of BD3-LM.

The base BD3-LM mask (paper Suppl. B.6, ``models/dit.py::block_diff_mask``)
defines, on a 1D flat sequence of length ``2*seqlen`` (noisy ⊕ clean):

    M_BD : block-diagonal     (within-block self-attention)
    M_OBC: offset-block-causal (noisy block sees clean blocks before it)
    M_BC : block-causal       (clean block sees clean blocks at or before it)

For XMIDI we layer one more rule on top, per R&D §"局部 attention 算法":

    Any token that lives on a "fully-<UNK>" row must NOT participate in
    attention, neither as query nor as key/value. The row composition is
    sample-specific, so the mask must be built per-batch.

This module provides:

    build_xmidi_attn_mask(...)
        Returns a (B, 1, 2L, 2L) bool tensor suitable for
        ``F.scaled_dot_product_attention(attn_mask=...)``.

The returned mask broadcasts over the head dimension. Memory cost:
    B * (2L)^2  bool bytes  = B * 4L^2 / 8e9  GB
For B=8, L=5120 -> 8 * (10240)^2 / 8e9 ≈ 105 MB.  Well within budget.
"""

from __future__ import annotations

from typing import Optional

import torch

from .tokenizer import NUM_COLS


# ---------------------------------------------------------------------------
def _base_block_diff_mask(seqlen: int,
                          block_size: int,
                          device: torch.device,
                          dtype: torch.dtype = torch.bool) -> torch.Tensor:
    """Re-implementation of the original BD3-LM mask, computed once on
    GPU as a dense (2*seqlen, 2*seqlen) bool tensor.

    This is byte-equivalent to ``models.dit.block_diff_mask`` evaluated with
    ``q_idx = arange(2L)[:, None]`` and ``kv_idx = arange(2L)[None, :]``.
    """
    n = seqlen
    L2 = 2 * n
    idx = torch.arange(L2, device=device)
    q_idx = idx.view(L2, 1)
    kv_idx = idx.view(1, L2)

    # x0 vs xt flag (xt -> 0, x0 -> 1)
    x0_flag_q = (q_idx >= n).long()
    x0_flag_kv = (kv_idx >= n).long()
    block_q = torch.where(x0_flag_q == 1, (q_idx - n) // block_size, q_idx // block_size)
    block_kv = torch.where(x0_flag_kv == 1, (kv_idx - n) // block_size, kv_idx // block_size)

    block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)
    offset_block_causal = (
        (block_q > block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 0)
    )
    block_causal = (block_q >= block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 1)

    return (block_diagonal | offset_block_causal | block_causal).to(dtype)


# ---------------------------------------------------------------------------
def _row_for_position(seqlen: int,
                      num_cols: int,
                      device: torch.device) -> torch.Tensor:
    """Map every position p in [0, 2*seqlen) to its row index in [0, num_rows).

    Layout:
        p in [0, seqlen)        -> noisy half;  row = p // num_cols
        p in [seqlen, 2*seqlen) -> clean half;  row = (p - seqlen) // num_cols
    """
    L2 = 2 * seqlen
    pos = torch.arange(L2, device=device)
    in_clean = pos >= seqlen
    rows = torch.where(
        in_clean,
        (pos - seqlen) // num_cols,
        pos // num_cols,
    )
    return rows  # (2L,) long


# ---------------------------------------------------------------------------
def build_xmidi_attn_mask(
    seqlen: int,
    block_size: int,
    row_unk_mask: torch.Tensor,
    num_cols: int = NUM_COLS,
    base_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-sample attention mask combining BD3-LM block_diff_mask and the
    "skip <UNK> rows" rule.

    Parameters
    ----------
    seqlen: int
        The XMIDI flat-sequence length, e.g. 5120.
    block_size: int
        BD3-LM block size (must divide num_cols if rows are atomic blocks;
        with default block_size=16 the math works on any layout because the
        UNK gate is applied row-wise after the block mask).
    row_unk_mask: (B, num_rows) bool tensor
        True if row r of sample b is fully <UNK>.
    num_cols: int
        Width of one row (default 512).
    base_mask: optional (2L, 2L) bool tensor, pre-computed
        ``_base_block_diff_mask``. Pass it in to avoid the O(L^2) recompute
        cost on every forward; the caller (DIT) caches one per ``(seqlen,
        block_size, device)`` key.

    Returns
    -------
    mask: (B, 1, 2L, 2L) bool
        True = the (q, kv) pair is allowed to attend.
    """
    if row_unk_mask.dim() != 2:
        raise ValueError(
            f"row_unk_mask must be (B, num_rows); got {tuple(row_unk_mask.shape)}"
        )
    B, _ = row_unk_mask.shape
    L = seqlen
    L2 = 2 * L
    device = row_unk_mask.device

    if base_mask is None:
        base_mask = _base_block_diff_mask(L, block_size, device=device)
    else:
        # caller may pass a (2L, 2L) tensor stored on a possibly-different
        # device; align it
        if base_mask.shape != (L2, L2):
            raise ValueError(
                f"base_mask shape {tuple(base_mask.shape)} != ({L2}, {L2})"
            )
        if base_mask.device != device:
            base_mask = base_mask.to(device)
        if base_mask.dtype != torch.bool:
            base_mask = base_mask.to(torch.bool)

    # row index of each of the 2L positions
    rows_for_pos = _row_for_position(L, num_cols, device=device)  # (2L,)

    # is_unk_for_pos: (B, 2L)
    #   == row_unk_mask[b, rows_for_pos[p]]
    is_unk = row_unk_mask.gather(
        dim=1, index=rows_for_pos.unsqueeze(0).expand(B, -1)
    )                                                # (B, 2L)
    active = (~is_unk).unsqueeze(1)                  # (B, 1, 2L)

    q_active = active.unsqueeze(-1)                  # (B, 1, 2L, 1)
    kv_active = active.unsqueeze(-2)                 # (B, 1, 1, 2L)

    base_4d = base_mask.view(1, 1, L2, L2)           # (1, 1, 2L, 2L)
    mask = base_4d & q_active & kv_active            # (B, 1, 2L, 2L) bool
    return mask
