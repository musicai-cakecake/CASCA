"""Reproducible inference/evaluation protocol helpers for BD3LMs-Music-V2.

This module contains no neural-network layers and therefore does not alter the
trained checkpoint structure.  It centralises provenance, RNG, generation-mask
and checkpoint compatibility checks used by the V2 inference/evaluation tools.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch

from .constants import (
    BLOCK_SIZE,
    DICT_LEN,
    MASK_ID,
    NUM_COLS,
    NUM_PROMPT_COLS,
    NUM_ROWS,
    NUM_SOURCE_ROWS,
    PAD_ID,
    SEQ_LEN,
    UNK_ID,
    VOCAB_SIZE,
)

INFERENCE_SCHEMA_VERSION = "bd3lms-music-v2.inference.v3"
MANIFEST_SCHEMA_VERSION = "bd3lms-music-v2.manifest.v3"


def seed_everything(seed: int) -> None:
    """Seed all RNGs used by the sampler in the current process."""
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def derive_sample_seed(base_seed: int, sample_index: int) -> int:
    """Stable per-sample seed, independent of sharding and resume boundaries."""
    payload = f"bd3lms-music-v2:{base_seed}:{sample_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def fingerprint_file(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Return a content-addressed, JSON-safe file identity."""
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"protocol input not found: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def dataset_fingerprint(
    dataset_root: str | os.PathLike[str],
    split: str,
    dict_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Bind an experiment to the exact dictionary/index/data bytes."""
    base = Path(dataset_root).resolve()
    if split not in {"train", "valid"}:
        raise ValueError(f"unsupported XMIDI split: {split}")
    if base.name != split:
        raise ValueError(
            f"dataset base path must end in {split!r}, got {str(base)!r}")
    files = {
        "dictionary": fingerprint_file(dict_path),
        "index": fingerprint_file(f"{base}.idx"),
        "length": fingerprint_file(base.parent / f"{split}_length.npy"),
        "data": fingerprint_file(f"{base}.data"),
    }
    return {
        "split": split,
        "files": files,
        "content_sha256": canonical_sha256(files),
    }


def source_fingerprint(
    repository_root: str | os.PathLike[str],
    relative_paths: Sequence[str],
) -> dict[str, Any]:
    """Hash inference-critical source files without relying on Git metadata."""
    root = Path(repository_root).resolve()
    files = {
        relative: fingerprint_file(root / relative)
        for relative in sorted(relative_paths)
    }
    return {"files": files, "content_sha256": canonical_sha256(files)}


def validate_manifest_provenance(manifest: Mapping[str, Any]) -> None:
    """Verify that a manifest's identity still hashes to its provenance ID."""
    identity = manifest.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError("manifest has no identity mapping")
    expected = canonical_sha256(identity)
    if manifest.get("provenance_id") != expected:
        raise ValueError("manifest identity/provenance hash mismatch")


def inspect_v2_checkpoint(
    path: str | os.PathLike[str],
    *,
    expected_hidden_size: int | None = None,
    require_ema: bool = False,
    expected_use_focal: bool | None = None,
    compute_sha256: bool = True,
) -> dict[str, Any]:
    """Fail fast on V1/incompatible checkpoints and return JSON-safe metadata."""
    checkpoint_path = Path(path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("state_dict"), dict
    ):
        raise ValueError("expected a Lightning checkpoint with a state_dict")

    state = checkpoint["state_dict"]
    embedding_keys = [
        key for key in state if key.endswith("backbone.vocab_embed.embedding")
    ]
    if len(embedding_keys) != 1:
        raise ValueError(
            "checkpoint must contain exactly one V2 vocab embedding; found "
            f"{embedding_keys}"
        )
    embedding_key = embedding_keys[0]
    embedding_shape = tuple(state[embedding_key].shape)
    if len(embedding_shape) != 2 or embedding_shape[0] != VOCAB_SIZE:
        raise ValueError(
            "incompatible checkpoint vocabulary: embedding shape "
            f"{embedding_shape}, expected first dimension {VOCAB_SIZE}. "
            "This usually indicates a V1 checkpoint."
        )
    if expected_hidden_size is not None and embedding_shape[1] != expected_hidden_size:
        raise ValueError(
            f"checkpoint hidden size {embedding_shape[1]} does not match model "
            f"configuration {expected_hidden_size}"
        )

    ema = checkpoint.get("ema")
    has_ema = isinstance(ema, dict) and isinstance(ema.get("shadow_params"), list)
    ema_embedding_index = None
    if has_ema:
        matching = [
            index
            for index, tensor in enumerate(ema["shadow_params"])
            if tuple(tensor.shape) == embedding_shape
        ]
        if len(matching) != 1:
            raise ValueError(
                "cannot uniquely identify the EMA embedding by shape; matching "
                f"shadow indices: {matching}"
            )
        ema_embedding_index = matching[0]
    if require_ema and not has_ema:
        raise ValueError("weight_source=ema requested, but checkpoint has no EMA state")

    focal_config = checkpoint.get("focal_loss_config")
    actual_use_focal = (
        bool(focal_config.get("use_focal", False))
        if isinstance(focal_config, dict)
        else False
    )
    if expected_use_focal is not None and actual_use_focal != expected_use_focal:
        raise ValueError(
            "checkpoint focal mode mismatch: "
            f"checkpoint use_focal={actual_use_focal}, "
            f"expected use_focal={expected_use_focal}"
        )

    metadata = {
        "path": str(checkpoint_path),
        "size_bytes": checkpoint_path.stat().st_size,
        "sha256": sha256_file(checkpoint_path) if compute_sha256 else None,
        "epoch": int(checkpoint.get("epoch", -1)),
        "global_step": int(checkpoint.get("global_step", -1)),
        "embedding_key": embedding_key,
        "embedding_shape": list(embedding_shape),
        "has_ema": has_ema,
        "ema_decay": float(ema.get("decay")) if has_ema else None,
        "ema_num_updates": int(ema.get("num_updates", -1)) if has_ema else None,
        "ema_embedding_index": ema_embedding_index,
        "focal_loss_config": dict(focal_config) if isinstance(focal_config, dict) else None,
    }
    del checkpoint
    return metadata


def materialize_inference_weights(model: Any, weight_source: str) -> None:
    """Select raw/EMA weights in-place without changing model parameters or keys."""
    if weight_source not in {"raw", "ema"}:
        raise ValueError(f"weight_source must be raw or ema, got {weight_source!r}")
    parameters = [parameter for parameter in model._get_parameters() if parameter.requires_grad]
    if weight_source == "ema":
        if model.ema is None:
            raise ValueError("weight_source=ema requested, but loaded model has no EMA")
        shadows = model.ema.shadow_params
        if len(shadows) != len(parameters):
            raise ValueError(
                f"EMA parameter count mismatch: {len(shadows)} != {len(parameters)}"
            )
        for index, (shadow, parameter) in enumerate(zip(shadows, parameters)):
            if tuple(shadow.shape) != tuple(parameter.shape):
                raise ValueError(
                    f"EMA shape mismatch at parameter {index}: "
                    f"{tuple(shadow.shape)} != {tuple(parameter.shape)}"
                )
        model.ema.copy_to(parameters)
    # The selected values are now ordinary model parameters.  Disable hooks
    # that could silently swap weights again during direct inference.
    model.ema = None
    model._inference_weight_source = weight_source


def mask_to_rle(mask: torch.Tensor | np.ndarray | Sequence[bool]) -> list[list[int]]:
    """Encode a flat boolean mask as ``[start, length]`` true runs."""
    flat = torch.as_tensor(mask, dtype=torch.bool).reshape(-1).cpu().tolist()
    runs: list[list[int]] = []
    start: int | None = None
    for index, selected in enumerate(flat + [False]):
        if selected and start is None:
            start = index
        elif not selected and start is not None:
            runs.append([start, index - start])
            start = None
    return runs


def rle_to_mask(runs: Sequence[Sequence[int]], length: int = SEQ_LEN) -> torch.Tensor:
    """Decode and strictly validate a generation-mask RLE."""
    mask = torch.zeros(length, dtype=torch.bool)
    previous_end = 0
    for run_index, run in enumerate(runs):
        if len(run) != 2:
            raise ValueError(f"RLE run {run_index} must be [start, length]")
        start, run_length = int(run[0]), int(run[1])
        end = start + run_length
        if run_length <= 0 or start < previous_end or start < 0 or end > length:
            raise ValueError(f"invalid/overlapping RLE run {run_index}: {run}")
        mask[start:end] = True
        previous_end = end
    return mask


def validate_generation_mask(
    generation_mask: torch.Tensor,
    condition_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Validate the V2 DAW generation region and return ``(B,16,512)`` bool."""
    mask = torch.as_tensor(generation_mask, dtype=torch.bool)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    if mask.ndim == 2 and tuple(mask.shape) == (NUM_ROWS, NUM_COLS):
        mask = mask.unsqueeze(0)
    elif mask.ndim == 2 and mask.shape[-1] == SEQ_LEN:
        mask = mask.reshape(mask.shape[0], NUM_ROWS, NUM_COLS)
    if mask.ndim != 3 or tuple(mask.shape[1:]) != (NUM_ROWS, NUM_COLS):
        raise ValueError(
            f"generation_mask must be (B,{NUM_ROWS},{NUM_COLS}) or (B,{SEQ_LEN}), "
            f"got {tuple(mask.shape)}"
        )
    if mask[:, :NUM_SOURCE_ROWS].any():
        raise ValueError("generation_mask selects source rows")
    if mask[:, NUM_SOURCE_ROWS:, :NUM_PROMPT_COLS].any():
        raise ValueError("generation_mask selects target prompts")
    if condition_ids is not None:
        condition = torch.as_tensor(condition_ids)
        if condition.ndim == 2:
            condition = condition.reshape(condition.shape[0], NUM_ROWS, NUM_COLS)
        if tuple(condition.shape) != tuple(mask.shape):
            raise ValueError("condition_ids and generation_mask shapes differ")
        if (mask & (condition == UNK_ID)).any():
            raise ValueError("generation_mask selects structural <UNK> cells")
    return mask


def generation_mask_from_record(record: Mapping[str, Any]) -> torch.Tensor:
    """Load canonical RLE mask, with an explicit legacy fallback."""
    if "generation_mask_rle" in record:
        mask = rle_to_mask(record["generation_mask_rle"]).reshape(NUM_ROWS, NUM_COLS)
        return validate_generation_mask(mask.unsqueeze(0))[0]

    gt_grid = record.get("ground_truth")
    if not isinstance(gt_grid, list) or len(gt_grid) != NUM_ROWS:
        raise ValueError("legacy record has no valid 16-row ground_truth grid")
    mask = torch.zeros(NUM_ROWS, NUM_COLS, dtype=torch.bool)
    for row in range(NUM_SOURCE_ROWS, NUM_ROWS):
        if len(gt_grid[row]) != NUM_COLS:
            raise ValueError(f"ground_truth row {row} does not have {NUM_COLS} cells")
        for col in range(NUM_PROMPT_COLS, NUM_COLS):
            mask[row, col] = gt_grid[row][col] != "<UNK>"
    return mask


def iter_selected_token_pairs(
    record: Mapping[str, Any],
) -> Iterator[tuple[int, int, str, str]]:
    """Yield aligned ``(row, col, reference, generated)`` pairs on exact G=1."""
    gt_grid = record["ground_truth"]
    gen_grid = record["gen"]
    if len(gt_grid) != NUM_ROWS or len(gen_grid) != NUM_ROWS:
        raise ValueError("ground_truth/gen must each contain 16 rows")
    mask = generation_mask_from_record(record)
    for row in range(NUM_SOURCE_ROWS, NUM_ROWS):
        if len(gt_grid[row]) != NUM_COLS or len(gen_grid[row]) != NUM_COLS:
            raise ValueError(f"grid row {row} does not contain {NUM_COLS} cells")
        for col in range(NUM_PROMPT_COLS, NUM_COLS):
            if bool(mask[row, col]):
                yield row, col, gt_grid[row][col], gen_grid[row][col]


def selected_token_segments(
    record: Mapping[str, Any],
) -> list[tuple[int, int, list[str], list[str]]]:
    """Return contiguous per-row G=1 segments without crossing fixed holes."""
    gt_grid = record["ground_truth"]
    gen_grid = record["gen"]
    mask = generation_mask_from_record(record)
    segments: list[tuple[int, int, list[str], list[str]]] = []
    for row in range(NUM_SOURCE_ROWS, NUM_ROWS):
        col = NUM_PROMPT_COLS
        while col < NUM_COLS:
            if not bool(mask[row, col]):
                col += 1
                continue
            start = col
            while col < NUM_COLS and bool(mask[row, col]):
                col += 1
            segments.append(
                (row, start, gt_grid[row][start:col], gen_grid[row][start:col])
            )
    return segments


def atomic_write_json(path: str | os.PathLike[str], value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=destination.parent, delete=False
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, destination)


def protocol_constants() -> dict[str, int]:
    return {
        "num_rows": NUM_ROWS,
        "num_cols": NUM_COLS,
        "num_source_rows": NUM_SOURCE_ROWS,
        "num_prompt_cols": NUM_PROMPT_COLS,
        "sequence_length": SEQ_LEN,
        "block_size": BLOCK_SIZE,
        "pad_id": PAD_ID,
        "dictionary_length": DICT_LEN,
        "unk_id": UNK_ID,
        "mask_id": MASK_ID,
        "vocab_size": VOCAB_SIZE,
    }
