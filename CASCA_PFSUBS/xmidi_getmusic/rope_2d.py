"""Axis-local factorised 2-D RoPE for the V2 music grid."""

from __future__ import annotations

import torch

from .constants import NUM_COLS, NUM_ROWS


class Rotary2D(torch.nn.Module):
    """Return row/time rotary phases without mixing the two axis subspaces.

    The returned third tuple element describes the two subspace widths. The
    attention implementation uses it to rotate each half independently.
    """

    def __init__(
        self,
        dim: int,
        num_rows: int = NUM_ROWS,
        num_cols: int = NUM_COLS,
        base: float = 10_000.0,
    ):
        super().__init__()
        if dim % 4 != 0:
            raise ValueError(f"Rotary2D requires head_dim divisible by 4, got {dim}")
        self.dim = int(dim)
        self.num_rows = int(num_rows)
        self.num_cols = int(num_cols)
        self.dim_per_axis = self.dim // 2
        self.axis_dims = (self.dim_per_axis, self.dim_per_axis)

        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, self.dim_per_axis, 2).float()
                / self.dim_per_axis
            )
        )
        self.register_buffer("inv_freq_row", inv_freq.clone(), persistent=True)
        self.register_buffer("inv_freq_col", inv_freq.clone(), persistent=True)
        self._cache_key = None
        self._cos_cached = None
        self._sin_cached = None

    def forward(self, x: torch.Tensor, seq_dim: int = 1):
        seq_len = int(x.shape[seq_dim])
        if seq_len > self.num_rows * self.num_cols:
            raise ValueError(
                f"2-D RoPE length {seq_len} exceeds grid capacity "
                f"{self.num_rows * self.num_cols}"
            )
        cache_key = (seq_len, x.device.type, x.device.index)
        if cache_key == self._cache_key and self._cos_cached is not None:
            return self._cos_cached, self._sin_cached, self.axis_dims

        position = torch.arange(
            seq_len, device=x.device, dtype=self.inv_freq_row.dtype
        )
        row = torch.div(position, self.num_cols, rounding_mode="floor")
        col = position.remainder(self.num_cols)
        row_freq = torch.outer(row, self.inv_freq_row)
        col_freq = torch.outer(col, self.inv_freq_col)

        # Each axis uses the standard duplicated-half layout. Rotation itself
        # is performed independently inside these two contiguous subspaces.
        phase = torch.cat(
            (row_freq, row_freq, col_freq, col_freq), dim=-1
        )
        cos = phase.cos()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
        sin = phase.sin()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
        cos[:, :, 2].fill_(1.0)
        sin[:, :, 2].zero_()

        self._cache_key = cache_key
        self._cos_cached = cos
        self._sin_cached = sin
        return cos, sin, self.axis_dims
