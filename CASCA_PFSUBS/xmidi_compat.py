"""Tiny compatibility layer that lets ``diffusion.py``, ``models/dit.py``,
``main.py`` and ``dataloader.py`` work with both XMIDI variants:

* ``xmidi``                - original (10, 512) grid with vocab=663
* ``xmidi_getmusic``       - V2 (16, 512) grid with vocab=142421
* ``xmidi_getmusic_large`` - (20, 512) grid with vocab=42376

Both packages export the same public names (NUM_ROWS, NUM_COLS,
NUM_SOURCE_ROWS, NUM_PROMPT_COLS, SEQ_LEN, UNK_ID, MASK_ID,
build_region_masks, build_xmidi_attn_mask, _base_block_diff_mask,
Rotary2D, ...). This module lets call-sites fetch the right module
based on ``config.data.train`` (which is set in
``configs/data/xmidi*.yaml``).
"""

from __future__ import annotations

from types import ModuleType
from typing import Optional


# tag set kept here so callers don't have to enumerate
XMIDI_DATASET_TAGS = ("xmidi", "xmidi_getmusic", "xmidi_getmusic_large")


def get_xmidi_dataset_tag(config) -> Optional[str]:
    """Return ``config.data.train`` if it is one of XMIDI_DATASET_TAGS, else None."""
    name = getattr(getattr(config, "data", None), "train", None)
    if name in XMIDI_DATASET_TAGS:
        return name
    return None


def is_xmidi_like(config) -> bool:
    """True iff ``config.data.train`` is one of the XMIDI variants."""
    return get_xmidi_dataset_tag(config) is not None


def get_xmidi_module(config) -> ModuleType:
    """Return the package object (``xmidi`` or ``xmidi_getmusic``) that
    matches ``config.data.train``. Raises ValueError if the config is not
    an XMIDI variant.
    """
    tag = get_xmidi_dataset_tag(config)
    if tag == "xmidi":
        import xmidi as m
        return m
    if tag == "xmidi_getmusic":
        import xmidi_getmusic as m
        return m
    if tag == "xmidi_getmusic_large":
        import xmidi_getmusic_large as m
        return m
    raise ValueError(
        f"config.data.train={tag!r} is not a known XMIDI variant; "
        f"expected one of {XMIDI_DATASET_TAGS}"
    )
