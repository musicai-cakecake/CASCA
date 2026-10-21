"""Public V2 XMIDI-GetMusic API."""

from .attn_mask import (
    _base_block_diff_mask,
    build_xmidi_attn_mask,
    query_active_from_rows,
)
from .audit import audit_xmidi_batch
from .constants import (
    BLOCK_SIZE,
    DICT_LEN,
    MASK_ID,
    MAX_LOSS_POSITIONS,
    NUM_BLOCKS,
    NUM_COLS,
    NUM_PROMPT_COLS,
    NUM_ROWS,
    NUM_SOURCE_ROWS,
    NUM_TARGET_ROWS,
    PAD_ID,
    ROW_NAMES,
    SEQ_LEN,
    TARGET_ROW_PAIRS,
    UNK_ID,
    VOCAB_SIZE,
)
from .dataset import XMIDIGetmusicDataset, get_xmidi_getmusic_dataset
from .region import attention_active_mask, build_region_masks, expected_loss_positions
from .rope_2d import Rotary2D
from .tokenizer import XMIDIGetmusicTokenizer

__all__ = [
    "XMIDIGetmusicTokenizer",
    "XMIDIGetmusicDataset",
    "get_xmidi_getmusic_dataset",
    "build_region_masks",
    "attention_active_mask",
    "expected_loss_positions",
    "build_xmidi_attn_mask",
    "query_active_from_rows",
    "audit_xmidi_batch",
    "_base_block_diff_mask",
    "Rotary2D",
    "PAD_ID",
    "UNK_ID",
    "MASK_ID",
    "VOCAB_SIZE",
    "DICT_LEN",
    "NUM_ROWS",
    "NUM_COLS",
    "NUM_SOURCE_ROWS",
    "NUM_TARGET_ROWS",
    "NUM_PROMPT_COLS",
    "SEQ_LEN",
    "BLOCK_SIZE",
    "NUM_BLOCKS",
    "MAX_LOSS_POSITIONS",
    "ROW_NAMES",
    "TARGET_ROW_PAIRS",
]
