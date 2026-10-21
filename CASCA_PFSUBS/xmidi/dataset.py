"""XMIDI dataset for BD3-LM training.

Supports two storage formats described in the R&D doc:

(1) ``idx data`` (recommended, default)
    - ``<split>.data``  binary blob (pickle per item)
    - ``<split>.idx``   numpy dict {'offsets': [byte offset, ...]}
    - ``<split>_length.npy`` total count
    Produced by :class:`read_representation_data.IndexedDatasetReader`.

(2) ``jsonl`` (fallback / debugging)
    - one json per line, ``{"tokens": [[10][512] of token strings]}``

Each sample yields a dict matching BD3-LM's expected batch interface:

    {
        'input_ids':     LongTensor[5120],
        'attention_mask':FloatTensor[5120],   # = region_mask (loss + diffusion mask)
        'row_unk_mask':  BoolTensor[10],      # 1 if the row is fully <UNK>
    }

The flatten convention is **row-major**:
    pos i  -> (row=i // 512, col=i % 512)

That is, the first 512 ids correspond to row 0 (full source row), the next
512 to row 1, ..., and the last 512 to row 9.
"""

from __future__ import annotations

import functools
import json
import os
import pickle
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .region import build_region_masks
from .tokenizer import (
    NUM_ROWS,
    NUM_COLS,
    SEQ_LEN,
    UNK_ID,
    XMIDITokenizer,
)


# ---------------------------------------------------------------------------
#                              raw readers
# ---------------------------------------------------------------------------
class _IdxDataReader:
    """Lightweight reader for the on-disk binary format.

    Modeled after ``read_representation_data.IndexedDatasetReader`` but adds
    lazy-open semantics so it is safe inside DataLoader workers (every worker
    opens its own file handle on first access).
    """

    def __init__(self, base_path: str):
        self.base_path = base_path
        self.idx_path = f"{base_path}.idx"
        self.data_path = f"{base_path}.data"
        self.length_path = f"{base_path}_length.npy"

        if not os.path.exists(self.idx_path):
            raise FileNotFoundError(self.idx_path)
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(self.data_path)

        idx_data = np.load(self.idx_path, allow_pickle=True).item()
        self.offsets: List[int] = list(idx_data["offsets"])
        self.total_num = len(self.offsets) - 1

        if os.path.exists(self.length_path):
            cached = int(np.load(self.length_path).item())
            if cached != self.total_num:
                # offsets is the source of truth
                pass

        self._fh: Optional[Any] = None

    def __len__(self) -> int:
        return self.total_num

    def _ensure_open(self) -> None:
        if self._fh is None:
            self._fh = open(self.data_path, "rb")

    def get(self, idx: int) -> np.ndarray:
        if idx < 0 or idx >= self.total_num:
            raise IndexError(idx)
        self._ensure_open()
        start = self.offsets[idx]
        end = self.offsets[idx + 1]
        self._fh.seek(start)
        raw = self._fh.read(end - start)
        item = pickle.loads(raw)
        arr = np.asarray(item)
        return arr

    def __del__(self):
        try:
            if self._fh is not None:
                self._fh.close()
        except Exception:
            pass


class _JsonlReader:
    """Reader for ``train.jsonl`` / ``valid.jsonl``.

    Each line: ``{"tokens": [[s00, s01, ...], [s10, ...], ...]}`` of shape
    [10, 512] with token strings.

    We pre-scan the file once to record line offsets for O(1) random access.
    """

    def __init__(self, path: str, tokenizer: XMIDITokenizer):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        self.path = path
        self.tokenizer = tokenizer
        self.offsets: List[int] = []
        with open(self.path, "rb") as f:
            ofs = 0
            for line in f:
                self.offsets.append(ofs)
                ofs += len(line)
        self.total_num = len(self.offsets)
        self._fh: Optional[Any] = None

    def __len__(self) -> int:
        return self.total_num

    def _ensure_open(self) -> None:
        if self._fh is None:
            self._fh = open(self.path, "rb")

    def get(self, idx: int) -> np.ndarray:
        self._ensure_open()
        self._fh.seek(self.offsets[idx])
        line = self._fh.readline().decode("utf-8")
        obj = json.loads(line)
        toks = obj["tokens"]  # nested list [10][512] of strings
        # token string -> id
        ids = np.empty((NUM_ROWS, NUM_COLS), dtype=np.int64)
        for r in range(NUM_ROWS):
            for c in range(NUM_COLS):
                ids[r, c] = self.tokenizer._convert_token_to_id(toks[r][c])
        return ids

    def __del__(self):
        try:
            if self._fh is not None:
                self._fh.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
#                              Torch Dataset
# ---------------------------------------------------------------------------
class XMIDIDataset(Dataset):
    """Map-style dataset that yields one (10, 512) sample per index.

    Parameters
    ----------
    base_path:
        For ``idx`` format: e.g. ``XMIDI_Dataset_full_output/train``
        For ``jsonl`` format: e.g. ``XMIDI_Dataset_full_output/train.jsonl``
    fmt: one of ``{"idx", "jsonl"}``
    tokenizer:
        Required for jsonl (token string -> id). Optional for idx.
    """

    def __init__(
        self,
        base_path: str,
        fmt: str = "idx",
        tokenizer: Optional[XMIDITokenizer] = None,
    ):
        self.base_path = base_path
        self.fmt = fmt
        if fmt == "idx":
            self._reader = _IdxDataReader(base_path)
        elif fmt == "jsonl":
            if tokenizer is None:
                raise ValueError("jsonl format requires a tokenizer")
            self._reader = _JsonlReader(base_path, tokenizer)
        else:
            raise ValueError(f"Unknown fmt={fmt!r}")

    def __len__(self) -> int:
        return len(self._reader)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        arr = self._reader.get(idx)              # (10, 512) int64 ndarray
        if arr.shape != (NUM_ROWS, NUM_COLS):
            raise ValueError(
                f"Sample {idx} has unexpected shape {arr.shape}, "
                f"expected {(NUM_ROWS, NUM_COLS)}"
            )

        flat = torch.from_numpy(arr).long().reshape(SEQ_LEN)            # (5120,)
        # build masks (single-batch helper)
        region_mask, row_unk_mask = build_region_masks(flat.unsqueeze(0))

        return {
            "input_ids": flat,                                          # (5120,)
            "attention_mask": region_mask.squeeze(0).float(),           # (5120,)
            "row_unk_mask": row_unk_mask.squeeze(0).bool(),             # (10,)
        }


# ---------------------------------------------------------------------------
#                       integration with BD3-LM dataloader
# ---------------------------------------------------------------------------
def get_xmidi_dataset(
    config,
    tokenizer: XMIDITokenizer,
    split: str,
) -> XMIDIDataset:
    """Returns the train or valid XMIDIDataset.

    Reads the following keys from ``config.data``:
        - cache_dir          (root folder containing the data files)
        - data_format        ("idx" or "jsonl", default "idx")

    The expected file layout under ``cache_dir`` is exactly the official
    XMIDI release:

        <cache_dir>/<split>.data       (idx format)
        <cache_dir>/<split>.idx
        <cache_dir>/<split>_length.npy
        <cache_dir>/<split>.jsonl      (jsonl format)
    """
    fmt = getattr(config.data, "data_format", "idx")
    cache_dir = config.data.cache_dir
    if not os.path.isdir(cache_dir):
        raise FileNotFoundError(
            f"XMIDI cache_dir not found: {cache_dir}\n"
            "Set configs/data/xmidi.yaml `cache_dir` to the absolute path of "
            "the XMIDI_Dataset_full_output folder."
        )

    if fmt == "idx":
        base = os.path.join(cache_dir, split)
        return XMIDIDataset(base, fmt="idx", tokenizer=tokenizer)
    elif fmt == "jsonl":
        path = os.path.join(cache_dir, f"{split}.jsonl")
        return XMIDIDataset(path, fmt="jsonl", tokenizer=tokenizer)
    else:
        raise ValueError(f"Unknown data_format={fmt!r}")
