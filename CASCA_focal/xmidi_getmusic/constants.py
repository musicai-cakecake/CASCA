"""Canonical data and vocabulary constants for BD3LMs-Music-V2."""

NUM_ROWS = 16
NUM_COLS = 512
NUM_SOURCE_ROWS = 4
NUM_TARGET_ROWS = 12
NUM_PROMPT_COLS = 16
SEQ_LEN = NUM_ROWS * NUM_COLS

BLOCK_SIZE = 16
NUM_BLOCKS = SEQ_LEN // BLOCK_SIZE
MAX_LOSS_POSITIONS = NUM_TARGET_ROWS * (NUM_COLS - NUM_PROMPT_COLS)

PAD_ID = 605
DICT_LEN = 142_419
UNK_ID = DICT_LEN
MASK_ID = DICT_LEN + 1
VOCAB_SIZE = DICT_LEN + 2

# Canonical row-major semantic layout. Target instruments occupy pitch/duration
# row pairs; keeping this table next to the geometry prevents V1/V2 drift.
ROW_NAMES = (
    "lead_pitch",
    "lead_duration",
    "chord_root",
    "chord_kind",
    "piano_pitch",
    "piano_duration",
    "guitar_pitch",
    "guitar_duration",
    "bass_pitch",
    "bass_duration",
    "string_pitch",
    "string_duration",
    "wind_pitch",
    "wind_duration",
    "drum_pitch",
    "drum_duration",
)
TARGET_ROW_PAIRS = ((4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15))

assert NUM_SOURCE_ROWS + NUM_TARGET_ROWS == NUM_ROWS
assert SEQ_LEN == 8192
assert NUM_COLS % BLOCK_SIZE == 0
assert NUM_PROMPT_COLS % BLOCK_SIZE == 0
assert NUM_BLOCKS == 512
assert MAX_LOSS_POSITIONS == 5952
assert UNK_ID == DICT_LEN
assert MASK_ID == DICT_LEN + 1
assert VOCAB_SIZE == DICT_LEN + 2
assert len(ROW_NAMES) == NUM_ROWS
assert tuple(row for pair in TARGET_ROW_PAIRS for row in pair) == tuple(
    range(NUM_SOURCE_ROWS, NUM_ROWS)
)
