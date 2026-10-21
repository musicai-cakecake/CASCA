"""XMIDI tokenizer for symbolic music tokens.

Vocabulary layout (final vocab_size = 663):
    idx [0, 485]  : ordinary music tokens
    idx 486       : <pad>          (already in dictionary)
    idx [487, 660]: ordinary music tokens (continued)
    idx 661       : <UNK>          (NOT in dictionary file; pre-encoded in data)
    idx 662       : <MASK>         (added by this tokenizer for diffusion training)

Notes
-----
- The on-disk dictionary file (``texture_dictionary.txt``) currently contains
  661 lines. The dataset has *already* encoded "no info / empty" positions as
  the integer 661 (= dictionary length), per the R&D spec. We therefore reserve
  661 for ``<UNK>`` even though it is not a line in the file.
- ``<MASK>`` is appended by this tokenizer to support absorbing-state diffusion
  in BD3-LM. Its id (662) is made available via the standard
  ``tokenizer.mask_token_id`` interface.
- This tokenizer subclasses ``transformers.PreTrainedTokenizer`` so it plugs
  into the existing BD3-LM training pipeline (``Diffusion.__init__`` reads
  ``tokenizer.vocab_size`` and ``tokenizer.mask_token_id``).
"""

from __future__ import annotations

import os
import typing

import transformers


# --- canonical special token ids (do not change without migrating data) ---
PAD_ID = 486
UNK_ID = 661
MASK_ID = 662
VOCAB_SIZE = 663

# rows / cols layout of one sample (R&D §"shape [10, 512]")
NUM_ROWS = 10
NUM_COLS = 512
NUM_SOURCE_ROWS = 4          # rows [0, 4) are user-input source, never masked
NUM_PROMPT_COLS = 16         # cols [0, 16) are prompts, never masked
SEQ_LEN = NUM_ROWS * NUM_COLS  # = 5120


class XMIDITokenizer(transformers.PreTrainedTokenizer):
    """A minimal HuggingFace-compatible tokenizer wrapping the texture dict.

    It is designed for the BD3-LM training loop, which only needs:
      * ``vocab_size`` property
      * ``mask_token_id`` / ``pad_token_id`` / ``bos_token_id`` / ``eos_token_id``
      * ``decode`` / ``batch_decode`` (for ``main.py::_print_batch`` debug)
      * ``encode`` (only used if user wants to convert jsonl token strings)

    XMIDI sequences have no natural BOS/EOS, so we alias them to ``<pad>``.
    The diffusion loop should set ``algo.ignore_bos=False`` so this aliasing
    has no side effect on the loss.
    """

    # standard tokens recognised by HF tokenizer base class
    pad_token = "<pad>"
    unk_token = "<UNK>"
    mask_token = "<MASK>"
    bos_token = "<pad>"  # XMIDI has no BOS; alias to <pad>
    eos_token = "<pad>"  # idem

    def __init__(
        self,
        dict_path: typing.Optional[str] = None,
        **kwargs,
    ):
        # 1) load the on-disk dictionary
        if dict_path is None:
            raise ValueError(
                "XMIDITokenizer requires `dict_path` pointing to "
                "texture_dictionary.txt"
            )
        if not os.path.exists(dict_path):
            raise FileNotFoundError(f"Dictionary file not found: {dict_path}")

        with open(dict_path, "r", encoding="utf-8") as f:
            dict_tokens = [line.rstrip("\n") for line in f]

        # 2) append <UNK> and <MASK> at fixed ids
        self._id_to_tok: typing.Dict[int, str] = {
            i: t for i, t in enumerate(dict_tokens)
        }
        # sanity check: <pad> must be at PAD_ID
        if self._id_to_tok.get(PAD_ID) != "<pad>":
            raise RuntimeError(
                f"<pad> not at idx {PAD_ID} in {dict_path}; got "
                f"{self._id_to_tok.get(PAD_ID)!r}"
            )
        # the dataset already pre-encodes "no info" positions as id=UNK_ID
        # (= len(dict_tokens) = 661). We register <UNK> at that id so token<->id
        # mapping is consistent with the data on disk.
        self._id_to_tok[UNK_ID] = self.unk_token
        self._id_to_tok[MASK_ID] = self.mask_token

        self._tok_to_id: typing.Dict[str, int] = {
            t: i for i, t in self._id_to_tok.items()
        }

        super().__init__(
            pad_token=self.pad_token,
            unk_token=self.unk_token,
            mask_token=self.mask_token,
            bos_token=self.bos_token,
            eos_token=self.eos_token,
            **kwargs,
        )

    # ---- HF interface ----
    @property
    def vocab_size(self) -> int:
        return VOCAB_SIZE  # 663

    def get_vocab(self) -> typing.Dict[str, int]:
        return dict(self._tok_to_id)

    def _tokenize(self, text: str, **kwargs) -> typing.List[str]:
        # XMIDI tokens in the jsonl files are space-separated strings.
        # We do not actually use this path during training (the dataset feeds
        # ids directly), but support it for ad-hoc inspection.
        return text.split()

    def _convert_token_to_id(self, token: str) -> int:
        # unknown text token falls back to <UNK>
        return self._tok_to_id.get(token, UNK_ID)

    def _convert_id_to_token(self, index: int) -> str:
        return self._id_to_tok.get(int(index), self.unk_token)

    def convert_tokens_to_string(self, tokens: typing.List[str]) -> str:
        return " ".join(tokens)

    def save_vocabulary(self, save_directory: str, filename_prefix=None):
        # not needed for our use case, but required by HF interface
        os.makedirs(save_directory, exist_ok=True)
        out = os.path.join(save_directory, "texture_dictionary.txt")
        with open(out, "w", encoding="utf-8") as f:
            for i in range(VOCAB_SIZE):
                f.write(self._id_to_tok[i] + "\n")
        return (out,)
