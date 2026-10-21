"""HuggingFace-compatible tokenizer for BD3LMs-Music-V2.

The serialized dictionary contains exactly 142,419 ordinary music tokens
(ids 0..142418), including ``<pad>`` at id 605. The dataset uses id 142419
for structural ``<UNK>`` cells. Diffusion appends ``<MASK>`` at id 142420.
"""

from __future__ import annotations

import os
import typing

import transformers

from .constants import (
    DICT_LEN,
    MASK_ID,
    NUM_COLS,
    NUM_PROMPT_COLS,
    NUM_ROWS,
    NUM_SOURCE_ROWS,
    NUM_TARGET_ROWS,
    PAD_ID,
    SEQ_LEN,
    UNK_ID,
    VOCAB_SIZE,
)


class XMIDIGetmusicTokenizer(transformers.PreTrainedTokenizer):
    """Tokenizer over the pre-encoded V2 GetMusic dictionary."""

    pad_token = "<pad>"
    unk_token = "<UNK>"
    mask_token = "<MASK>"
    bos_token = "<pad>"
    eos_token = "<pad>"

    def __init__(self, dict_path: typing.Optional[str] = None, **kwargs):
        if dict_path is None:
            raise ValueError(
                "XMIDIGetmusicTokenizer requires `dict_path` pointing to "
                "XMIDI_Dataset_output_v2/whole_dict.txt"
            )
        if not os.path.isfile(dict_path):
            raise FileNotFoundError(f"Dictionary file not found: {dict_path}")

        with open(dict_path, "r", encoding="utf-8") as handle:
            dict_tokens = [line.rstrip("\r\n") for line in handle]

        if len(dict_tokens) != DICT_LEN:
            raise RuntimeError(
                f"V2 dictionary has {len(dict_tokens)} lines; expected exactly "
                f"{DICT_LEN}: {dict_path}"
            )
        for special in (self.unk_token, self.mask_token):
            if special in dict_tokens:
                raise RuntimeError(
                    f"{special} must not be a line in the V2 data dictionary"
                )
        if dict_tokens[PAD_ID] != self.pad_token:
            raise RuntimeError(
                f"<pad> must be id {PAD_ID}; found {dict_tokens[PAD_ID]!r}"
            )

        self._dict_tokens = tuple(dict_tokens)
        self._id_to_tok: typing.Dict[int, str] = dict(enumerate(dict_tokens))
        self._id_to_tok[UNK_ID] = self.unk_token
        self._id_to_tok[MASK_ID] = self.mask_token
        self._tok_to_id: typing.Dict[str, int] = {
            token: idx for idx, token in self._id_to_tok.items()
        }
        if len(self._tok_to_id) != VOCAB_SIZE:
            raise RuntimeError(
                "V2 dictionary tokens must be unique for reversible ID/token "
                f"evaluation; got {len(self._tok_to_id)} unique of {VOCAB_SIZE}"
            )

        super().__init__(
            pad_token=self.pad_token,
            unk_token=self.unk_token,
            mask_token=self.mask_token,
            bos_token=self.bos_token,
            eos_token=self.eos_token,
            **kwargs,
        )

        expected = {
            "pad_token_id": PAD_ID,
            "unk_token_id": UNK_ID,
            "mask_token_id": MASK_ID,
            "vocab_size": VOCAB_SIZE,
        }
        for name, value in expected.items():
            actual = getattr(self, name)
            if actual != value:
                raise RuntimeError(f"{name}={actual}, expected {value}")

    @property
    def vocab_size(self) -> int:
        return VOCAB_SIZE

    def get_vocab(self) -> typing.Dict[str, int]:
        return dict(self._tok_to_id)

    def _tokenize(self, text: str, **kwargs) -> typing.List[str]:
        # Training consumes pre-encoded integer arrays. This method exists only
        # for lightweight diagnostics; dictionary entries can themselves contain
        # spaces and therefore text tokenization is not round-trip safe.
        return text.split()

    def _convert_token_to_id(self, token: str) -> int:
        return self._tok_to_id.get(token, UNK_ID)

    def _convert_id_to_token(self, index: int) -> str:
        return self._id_to_tok.get(int(index), self.unk_token)

    def convert_tokens_to_string(self, tokens: typing.List[str]) -> str:
        return " ".join(token.replace(" ", "_") for token in tokens)

    def save_vocabulary(self, save_directory: str, filename_prefix=None):
        """Save only the data dictionary; HF stores special tokens separately."""
        os.makedirs(save_directory, exist_ok=True)
        filename = "whole_dict.txt"
        if filename_prefix:
            filename = f"{filename_prefix}-{filename}"
        output_path = os.path.join(save_directory, filename)
        with open(output_path, "w", encoding="utf-8") as handle:
            for token in self._dict_tokens:
                handle.write(token + "\n")
        return (output_path,)
