"""Indexed V2 XMIDI-GetMusic dataset.

Each item is an integer array of shape ``(16, 512)`` and is returned as a
row-major sequence of length 8192 together with generation and inactive-row
masks.
"""

from __future__ import annotations

import json
import os
import pickle
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .constants import MASK_ID, NUM_COLS, NUM_ROWS, SEQ_LEN, UNK_ID
from .region import build_region_masks
from .tokenizer import XMIDIGetmusicTokenizer


class _IdxDataReader:
    """Lazy-open binary reader that is safe to construct before worker fork."""

    def __init__(self, base_path: str):
        self.base_path = base_path
        self.idx_path = f"{base_path}.idx"
        self.data_path = f"{base_path}.data"
        self.length_path = f"{base_path}_length.npy"
        for path in (self.idx_path, self.data_path, self.length_path):
            if not os.path.isfile(path):
                raise FileNotFoundError(path)

        idx_data = np.load(self.idx_path, allow_pickle=True).item()
        if "offsets" not in idx_data:
            raise RuntimeError(f"Missing 'offsets' in {self.idx_path}")
        self.offsets: List[int] = [int(value) for value in idx_data["offsets"]]
        if len(self.offsets) < 2 or self.offsets[0] != 0:
            raise RuntimeError(f"Invalid offsets in {self.idx_path}")
        if any(left > right for left, right in zip(self.offsets, self.offsets[1:])):
            raise RuntimeError(f"Offsets are not monotonic in {self.idx_path}")
        if self.offsets[-1] > os.path.getsize(self.data_path):
            raise RuntimeError(f"Last offset exceeds {self.data_path} size")

        self.total_num = len(self.offsets) - 1
        recorded_length = int(np.load(self.length_path, allow_pickle=True).item())
        if recorded_length != self.total_num:
            raise RuntimeError(
                f"Index has {self.total_num} items but {self.length_path} "
                f"records {recorded_length}"
            )
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
        start, end = self.offsets[idx], self.offsets[idx + 1]
        self._fh.seek(start)
        raw = self._fh.read(end - start)
        if len(raw) != end - start:
            raise EOFError(f"Short read for item {idx} from {self.data_path}")
        return np.asarray(pickle.loads(raw))

    def __del__(self):
        try:
            if self._fh is not None:
                self._fh.close()
        except Exception:
            pass


class _JsonlReader:
    """Debug-only reader for nested token or integer grids."""

    def __init__(self, path: str, tokenizer: XMIDIGetmusicTokenizer):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        self.path = path
        self.tokenizer = tokenizer
        self.offsets: List[int] = []
        with open(path, "rb") as handle:
            offset = 0
            for line in handle:
                self.offsets.append(offset)
                offset += len(line)
        self.total_num = len(self.offsets)
        self._fh: Optional[Any] = None

    def __len__(self) -> int:
        return self.total_num

    def _ensure_open(self) -> None:
        if self._fh is None:
            self._fh = open(self.path, "rb")

    def get(self, idx: int) -> np.ndarray:
        if idx < 0 or idx >= self.total_num:
            raise IndexError(idx)
        self._ensure_open()
        self._fh.seek(self.offsets[idx])
        obj = json.loads(self._fh.readline().decode("utf-8"))
        tokens = obj["tokens"]
        if len(tokens) != NUM_ROWS or any(len(row) != NUM_COLS for row in tokens):
            raise ValueError(
                f"JSONL item {idx} must contain a {NUM_ROWS}x{NUM_COLS} grid"
            )
        if all(isinstance(value, int) for row in tokens for value in row):
            return np.asarray(tokens, dtype=np.int64)
        vocab = self.tokenizer.get_vocab()
        unknown = sorted({
            value for row in tokens for value in row if value not in vocab
        })
        if unknown:
            preview = unknown[:5]
            raise ValueError(
                f"JSONL item {idx} contains tokens absent from the V2 "
                f"dictionary: {preview}"
            )
        return np.asarray(
            [[vocab[value] for value in row] for row in tokens],
            dtype=np.int64,
        )

    def __del__(self):
        try:
            if self._fh is not None:
                self._fh.close()
        except Exception:
            pass


class XMIDIGetmusicDataset(Dataset):
    """Map-style V2 dataset returning one 16x512 grid per index."""

    def __init__(
        self,
        base_path: str,
        fmt: str = "idx",
        tokenizer: Optional[XMIDIGetmusicTokenizer] = None,
    ):
        self.base_path = base_path
        self.fmt = fmt
        self.tokenizer = tokenizer
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
        arr = self._reader.get(idx)
        if arr.shape != (NUM_ROWS, NUM_COLS):
            raise ValueError(
                f"Sample {idx} has shape {arr.shape}; expected "
                f"{(NUM_ROWS, NUM_COLS)}"
            )
        if not np.issubdtype(arr.dtype, np.integer):
            raise TypeError(f"Sample {idx} must have integer dtype, got {arr.dtype}")
        min_id, max_id = int(arr.min()), int(arr.max())
        if min_id < 0 or max_id > UNK_ID:
            raise ValueError(
                f"Sample {idx} ids [{min_id}, {max_id}] exceed disk range "
                f"[0, {UNK_ID}]"
            )
        if np.any(arr == MASK_ID):
            raise ValueError(f"Sample {idx} contains diffusion-only <MASK>")

        flat = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.int64)).reshape(SEQ_LEN)
        region_mask, row_unk_mask = build_region_masks(flat.unsqueeze(0))
        return {
            "input_ids": flat,
            "attention_mask": region_mask.squeeze(0),
            "row_unk_mask": row_unk_mask.squeeze(0),
        }


def get_xmidi_getmusic_dataset(
    config,
    tokenizer: XMIDIGetmusicTokenizer,
    split: str,
) -> XMIDIGetmusicDataset:
    """Load ``train`` or ``valid`` from the configured V2 dataset directory."""
    fmt = getattr(config.data, "data_format", "idx")
    cache_dir = config.data.cache_dir
    if not os.path.isdir(cache_dir):
        raise FileNotFoundError(
            f"V2 XMIDI cache_dir not found: {cache_dir}\n"
            "Set configs/data/xmidi_getmusic.yaml `cache_dir` to "
            "XMIDI_Dataset_output_v2."
        )
    if split not in {"train", "valid"}:
        raise ValueError(f"Unsupported split={split!r}; expected 'train' or 'valid'")
    if fmt == "idx":
        return XMIDIGetmusicDataset(
            os.path.join(cache_dir, split), fmt="idx", tokenizer=tokenizer
        )
    if fmt == "jsonl":
        return XMIDIGetmusicDataset(
            os.path.join(cache_dir, f"{split}.jsonl"),
            fmt="jsonl",
            tokenizer=tokenizer,
        )
    raise ValueError(f"Unknown data_format={fmt!r}")
