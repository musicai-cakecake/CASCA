"""2D rotary position embeddings for the XMIDI grid.

The original BD3-LM uses 1D RoPE: every token at flat position ``i`` is
rotated by a single set of frequencies derived from ``i``. For a token
sequence that is really a [num_rows, num_cols] grid (R&D §"相对位置编码"),
we instead split the head dimension into two equal halves:

    head_dim[ : head_dim/2]   ->  rotated by row index  (i // num_cols)
    head_dim[head_dim/2 : ]   ->  rotated by col index  (i %  num_cols)

This is the standard "axial" 2D-RoPE construction (used in ViT-RoPE,
PaLI / PaliGemma, RoFormer-2D). Two tokens in the same column but
different rows get different rotations on the first half of the head;
two tokens in the same row but different columns get different
rotations on the second half. The model can therefore distinguish "this
is row 4 col 32" from "this is row 7 col 32" purely through positional
geometry, which matches our claim that *each row carries fixed semantic
content* (e.g. row 0 = melody source, row 5 = guitar target, ...).

The module's forward signature is identical to ``models.dit.Rotary`` so
``apply_rotary_pos_emb_torchscript(qkv, cos, sin)`` from BD3-LM works
unchanged.

Notes
-----
- ``head_dim`` must be divisible by 4 (each half must hold an even number
  of frequency components).
- For positions ``i >= num_rows * num_cols`` (e.g. KV-cache that grew
  past one full XMIDI sample), the row index is taken modulo
  ``num_rows`` so the embedding stays well-defined. Concrete sampling
  semantics are revisited in milestone 6.
"""

from __future__ import annotations

import torch

from .tokenizer import NUM_ROWS, NUM_COLS


class Rotary2D(torch.nn.Module):
    """Drop-in replacement for ``models.dit.Rotary`` on a [num_rows, num_cols]
    grid.

    Parameters
    ----------
    dim: int
        Per-head dimension (i.e. ``hidden_size // n_heads``).
    num_rows: int
        Number of rows in the underlying 2D grid (default 10 for XMIDI).
    num_cols: int
        Number of columns per row (default 512 for XMIDI).
    base: float
        RoPE base frequency. Same default as 1D RoPE in this repo.
    """

    def __init__(self,
                 dim: int,
                 num_rows: int = NUM_ROWS,
                 num_cols: int = NUM_COLS,
                 base: float = 10_000.0):
        super().__init__()
        if dim % 4 != 0:
            raise ValueError(
                f"Rotary2D needs head_dim divisible by 4, got {dim}. "
                "Each axis splits the head into a (cos, sin) pair, so the "
                "axis dimension must be even, and there are two axes."
            )
        self.dim = int(dim)
        self.num_rows = int(num_rows)
        self.num_cols = int(num_cols)
        # half of the head goes to row freqs, the other half to col freqs.
        self.dim_per_axis = self.dim // 2

        # one set of inverse frequencies per axis (independent buffers in
        # case we ever want different bases per axis).
        inv_freq = 1.0 / (
            base ** (
                torch.arange(0, self.dim_per_axis, 2).float()
                / self.dim_per_axis
            )
        )
        self.register_buffer("inv_freq_row", inv_freq.clone())
        self.register_buffer("inv_freq_col", inv_freq.clone())

        # caches keyed by seq_len + device (recomputed on length change)
        self._seq_len_cached: int = -1
        self._cos_cached: torch.Tensor | None = None
        self._sin_cached: torch.Tensor | None = None

    # NOTE: signature matches models.dit.Rotary.forward(x, seq_dim=1)
    def forward(self, x: torch.Tensor, seq_dim: int = 1):
        seq_len = x.shape[seq_dim]
        if seq_len == self._seq_len_cached and self._cos_cached is not None:
            return self._cos_cached, self._sin_cached

        device = x.device
        dtype = self.inv_freq_row.dtype  # float32 for accuracy

        pos = torch.arange(seq_len, device=device, dtype=dtype)
        # row index within the grid; positions past one full grid wrap.
        # During training we only ever pass seq_len <= num_rows*num_cols.
        # During sampling the KV-cache may exceed one grid; modulo keeps
        # the embedding well-defined (revisited in milestone 6).
        row = (pos // self.num_cols) % self.num_rows
        col = pos % self.num_cols

        row_freqs = torch.einsum("i,j->ij", row, self.inv_freq_row.clone())
        col_freqs = torch.einsum("i,j->ij", col, self.inv_freq_col.clone())
        # duplicate freqs to match the (cos, sin) interleave used by
        # apply_rotary_pos_emb_torchscript (rotate_half).
        emb_row = torch.cat((row_freqs, row_freqs), dim=-1)  # (L, dim_per_axis)
        emb_col = torch.cat((col_freqs, col_freqs), dim=-1)  # (L, dim_per_axis)
        emb = torch.cat((emb_row, emb_col), dim=-1).to(device)  # (L, dim)

        # Layout matches Rotary: dims are (batch, seq, qkv, head, head_dim)
        cos = emb.cos()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
        sin = emb.sin()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
        # identity transform for v: cos=1, sin=0 along the qkv axis at idx 2
        cos[:, :, 2, :, :] = 1.0
        sin[:, :, 2, :, :] = 0.0

        self._seq_len_cached = seq_len
        self._cos_cached = cos
        self._sin_cached = sin
        return cos, sin
