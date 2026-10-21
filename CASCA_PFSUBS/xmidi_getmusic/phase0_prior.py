"""Read-only Phase 0 data identity and PAD-prior statistics."""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .constants import (
    BLOCK_SIZE,
    MASK_ID,
    NUM_COLS,
    NUM_PROMPT_COLS,
    NUM_ROWS,
    NUM_SOURCE_ROWS,
    PAD_ID,
    ROW_NAMES,
    UNK_ID,
    VOCAB_SIZE,
)

TARGET_ROWS = tuple(range(NUM_SOURCE_ROWS, NUM_ROWS))
CONTENT_BLOCKS = (NUM_COLS - NUM_PROMPT_COLS) // BLOCK_SIZE
INSTRUMENT_PAIRS = (
    ("piano", 4, 5),
    ("guitar", 6, 7),
    ("bass", 8, 9),
    ("string", 10, 11),
    ("wind", 12, 13),
    ("drum", 14, 15),
)
EXPECTED_V2_COUNTS = {
    "train": {"examples": 133_646, "generation_cells": 323_238_700, "pad": 217_999_140},
    "valid": {"examples": 6_817, "generation_cells": 14_310_994, "pad": 8_634_610},
}


def sha256_file(path: str | Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_hashed_json(path: str | Path, payload: Mapping[str, Any]) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    body = dict(payload)
    body.pop("artifact_sha256", None)
    digest = canonical_json_sha256(body)
    body["artifact_sha256"] = digest
    output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return digest


@dataclass
class RawCounts:
    examples: int
    generation_cells: int
    pad: int
    token_counts: np.ndarray
    row_cells: np.ndarray
    row_pad: np.ndarray
    block_cells: np.ndarray
    block_pad: np.ndarray
    row_block_cells: np.ndarray
    row_block_pad: np.ndarray


def _new_counts() -> RawCounts:
    rows = len(TARGET_ROWS)
    return RawCounts(
        examples=0,
        generation_cells=0,
        pad=0,
        token_counts=np.zeros(VOCAB_SIZE, dtype=np.int64),
        row_cells=np.zeros(rows, dtype=np.int64),
        row_pad=np.zeros(rows, dtype=np.int64),
        block_cells=np.zeros(CONTENT_BLOCKS, dtype=np.int64),
        block_pad=np.zeros(CONTENT_BLOCKS, dtype=np.int64),
        row_block_cells=np.zeros((rows, CONTENT_BLOCKS), dtype=np.int64),
        row_block_pad=np.zeros((rows, CONTENT_BLOCKS), dtype=np.int64),
    )


def load_offsets(data_path: str | Path, idx_path: str | Path) -> list[int]:
    data_path = Path(data_path)
    idx_path = Path(idx_path)
    payload = np.load(idx_path, allow_pickle=True).item()
    offsets = [int(value) for value in payload.get("offsets", ())]
    if len(offsets) < 2 or offsets[0] != 0:
        raise ValueError(f"invalid offsets in {idx_path}")
    if any(left > right for left, right in zip(offsets, offsets[1:])):
        raise ValueError(f"non-monotonic offsets in {idx_path}")
    if offsets[-1] > data_path.stat().st_size:
        raise ValueError(f"last offset exceeds {data_path} size")
    length_path = data_path.with_name(f"{data_path.stem}_length.npy")
    recorded = int(np.load(length_path, allow_pickle=True).item())
    if recorded != len(offsets) - 1:
        raise ValueError(f"{length_path} records {recorded}, index has {len(offsets) - 1}")
    return offsets


def load_sample(handle: Any, start: int, end: int, sample_index: int) -> np.ndarray:
    handle.seek(start)
    raw = handle.read(end - start)
    if len(raw) != end - start:
        raise EOFError(f"short read for sample {sample_index}")
    sample = np.asarray(pickle.loads(raw))
    if sample.shape not in ((NUM_ROWS, NUM_COLS), (NUM_ROWS * NUM_COLS,)):
        raise ValueError(f"sample {sample_index}: invalid shape {sample.shape}")
    if not np.issubdtype(sample.dtype, np.integer):
        raise TypeError(f"sample {sample_index}: expected integer dtype")
    sample = sample.reshape(NUM_ROWS, NUM_COLS).astype(np.int64, copy=False)
    if np.any(sample < 0) or np.any(sample >= VOCAB_SIZE):
        raise ValueError(f"sample {sample_index}: token outside vocabulary")
    if np.any(sample == MASK_ID):
        raise ValueError(f"sample {sample_index}: on-disk <MASK> is forbidden")
    return sample


def scan_split(data_path: str | Path, idx_path: str | Path) -> RawCounts:
    data_path = Path(data_path)
    offsets = load_offsets(data_path, idx_path)
    counts = _new_counts()
    with data_path.open("rb") as handle:
        for sample_index, (start, end) in enumerate(zip(offsets, offsets[1:])):
            sample = load_sample(handle, start, end, sample_index)
            target = sample[NUM_SOURCE_ROWS:, NUM_PROMPT_COLS:]
            generation = target != UNK_ID
            pad = (target == PAD_ID) & generation
            selected = target[generation]
            counts.examples += 1
            counts.generation_cells += int(generation.sum())
            counts.pad += int(pad.sum())
            counts.token_counts += np.bincount(
                selected, minlength=VOCAB_SIZE
            ).astype(np.int64, copy=False)
            counts.row_cells += generation.sum(axis=1, dtype=np.int64)
            counts.row_pad += pad.sum(axis=1, dtype=np.int64)
            by_row_block = generation.reshape(len(TARGET_ROWS), CONTENT_BLOCKS, BLOCK_SIZE)
            pad_by_row_block = pad.reshape(len(TARGET_ROWS), CONTENT_BLOCKS, BLOCK_SIZE)
            row_block_cells = by_row_block.sum(axis=2, dtype=np.int64)
            row_block_pad = pad_by_row_block.sum(axis=2, dtype=np.int64)
            counts.row_block_cells += row_block_cells
            counts.row_block_pad += row_block_pad
            counts.block_cells += row_block_cells.sum(axis=0, dtype=np.int64)
            counts.block_pad += row_block_pad.sum(axis=0, dtype=np.int64)
    validate_counts(counts)
    return counts


def combine_counts(parts: Iterable[RawCounts]) -> RawCounts:
    combined = _new_counts()
    for part in parts:
        combined.examples += part.examples
        combined.generation_cells += part.generation_cells
        combined.pad += part.pad
        combined.token_counts += part.token_counts
        combined.row_cells += part.row_cells
        combined.row_pad += part.row_pad
        combined.block_cells += part.block_cells
        combined.block_pad += part.block_pad
        combined.row_block_cells += part.row_block_cells
        combined.row_block_pad += part.row_block_pad
    validate_counts(combined)
    return combined


def validate_counts(counts: RawCounts) -> None:
    if int(counts.token_counts.sum()) != counts.generation_cells:
        raise ValueError("token histogram does not sum to generation cells")
    if int(counts.token_counts[PAD_ID]) != counts.pad:
        raise ValueError("PAD histogram count disagrees with PAD aggregate")
    if int(counts.token_counts[UNK_ID]) != 0 or int(counts.token_counts[MASK_ID]) != 0:
        raise ValueError("<UNK>/<MASK> must not occur in canonical generation cells")
    for name, cells, pad in (
        ("row", counts.row_cells, counts.row_pad),
        ("block", counts.block_cells, counts.block_pad),
        ("row-block", counts.row_block_cells, counts.row_block_pad),
    ):
        if int(cells.sum()) != counts.generation_cells or int(pad.sum()) != counts.pad:
            raise ValueError(f"{name} aggregates do not sum to global counts")


def assert_expected_v2_counts(split: str, counts: RawCounts) -> None:
    actual = {
        "examples": counts.examples,
        "generation_cells": counts.generation_cells,
        "pad": counts.pad,
    }
    if actual != EXPECTED_V2_COUNTS[split]:
        raise ValueError(f"{split} V2 count regression: {actual}")


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def counts_to_json(counts: RawCounts) -> dict[str, Any]:
    non_pad = counts.generation_cells - counts.pad
    rows = []
    for offset, row_index in enumerate(TARGET_ROWS):
        cells = int(counts.row_cells[offset])
        pad = int(counts.row_pad[offset])
        rows.append({
            "row_index": row_index,
            "row_name": ROW_NAMES[row_index],
            "generation_cells": cells,
            "pad": pad,
            "non_pad": cells - pad,
            "pad_ratio": _ratio(pad, cells),
        })
    pairs = []
    for name, first_row, second_row in INSTRUMENT_PAIRS:
        offsets = [first_row - NUM_SOURCE_ROWS, second_row - NUM_SOURCE_ROWS]
        cells = int(counts.row_cells[offsets].sum())
        pad = int(counts.row_pad[offsets].sum())
        pairs.append({
            "instrument_pair": name,
            "row_indices": [first_row, second_row],
            "generation_cells": cells,
            "pad": pad,
            "non_pad": cells - pad,
            "pad_ratio": _ratio(pad, cells),
        })
    blocks = []
    for index in range(CONTENT_BLOCKS):
        cells = int(counts.block_cells[index])
        pad = int(counts.block_pad[index])
        start = NUM_PROMPT_COLS + index * BLOCK_SIZE
        blocks.append({
            "content_block_index": index,
            "column_range": [start, start + BLOCK_SIZE - 1],
            "generation_cells": cells,
            "pad": pad,
            "non_pad": cells - pad,
            "pad_ratio": _ratio(pad, cells),
        })
    return {
        "examples": counts.examples,
        "generation_cells": counts.generation_cells,
        "pad": counts.pad,
        "non_pad": non_pad,
        "pad_ratio": _ratio(counts.pad, counts.generation_cells),
        "event_ratio": _ratio(non_pad, counts.generation_cells),
        "event_to_pad_count_ratio": _ratio(non_pad, counts.pad),
        "pad_to_event_count_ratio": _ratio(counts.pad, non_pad),
        "per_target_row": rows,
        "per_instrument_pair": pairs,
        "per_content_block": blocks,
        "per_target_row_content_block": {
            "generation_cells": counts.row_block_cells.tolist(),
            "pad": counts.row_block_pad.tolist(),
        },
        "token_counts_by_id": counts.token_counts.tolist(),
    }
