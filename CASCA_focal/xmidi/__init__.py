"""XMIDI extension for BD3-LM.

Symbolic-music adaptation: a [10, 512] grid where the first 4 rows are
user-input source, the first 16 columns of every row are prompts, and the
remaining 6 x 496 region is the target to be diffusion-generated. Tokens
equal to ``UNK_ID`` indicate "no info / empty" and stay clean throughout
the diffusion process.

Public API:
    XMIDITokenizer:         transformers-compatible tokenizer
    XMIDIDataset:           map-style torch Dataset
    get_xmidi_dataset:      helper that reads ``config.data`` and returns
                            the requested split
    build_region_masks:     produces (region_mask, row_unk_mask)
    PAD_ID, UNK_ID, MASK_ID, VOCAB_SIZE: id constants
    NUM_ROWS, NUM_COLS, NUM_SOURCE_ROWS, NUM_PROMPT_COLS, SEQ_LEN: shape
"""

from .tokenizer import (
    XMIDITokenizer,
    PAD_ID,
    UNK_ID,
    MASK_ID,
    VOCAB_SIZE,
    NUM_ROWS,
    NUM_COLS,
    NUM_SOURCE_ROWS,
    NUM_PROMPT_COLS,
    SEQ_LEN,
)
from .region import build_region_masks, attention_active_mask, expected_loss_positions
from .dataset import XMIDIDataset, get_xmidi_dataset
from .attn_mask import build_xmidi_attn_mask, _base_block_diff_mask
from .rope_2d import Rotary2D

__all__ = [
    "XMIDITokenizer",
    "XMIDIDataset",
    "get_xmidi_dataset",
    "build_region_masks",
    "attention_active_mask",
    "expected_loss_positions",
    "build_xmidi_attn_mask",
    "Rotary2D",
    "PAD_ID",
    "UNK_ID",
    "MASK_ID",
    "VOCAB_SIZE",
    "NUM_ROWS",
    "NUM_COLS",
    "NUM_SOURCE_ROWS",
    "NUM_PROMPT_COLS",
    "SEQ_LEN",
]
