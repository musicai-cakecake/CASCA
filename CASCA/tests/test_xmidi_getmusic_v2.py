from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from xmidi_getmusic import (
    DICT_LEN,
    MASK_ID,
    NUM_COLS,
    NUM_PROMPT_COLS,
    NUM_ROWS,
    NUM_SOURCE_ROWS,
    PAD_ID,
    ROW_NAMES,
    SEQ_LEN,
    TARGET_ROW_PAIRS,
    UNK_ID,
    VOCAB_SIZE,
    XMIDIGetmusicTokenizer,
    audit_xmidi_batch,
    build_region_masks,
    build_xmidi_attn_mask,
)
from xmidi_getmusic.eval_protocol import (
    derive_sample_seed,
    mask_to_rle,
    rle_to_mask,
    validate_generation_mask,
)
from xmidi_getmusic.rope_2d import Rotary2D


def test_v2_constants_and_dictionary(tmp_path):
    assert (NUM_ROWS, NUM_COLS, SEQ_LEN) == (16, 512, 8192)
    assert (PAD_ID, UNK_ID, MASK_ID, VOCAB_SIZE) == (
        605,
        142419,
        142420,
        142421,
    )
    dictionary = tmp_path / "whole_dict.txt"
    tokens = [f"test_token_{index}" for index in range(DICT_LEN)]
    tokens[PAD_ID] = "<pad>"
    tokens[1000] = "piano60 64 67"
    dictionary.write_text("\n".join(tokens) + "\n", encoding="utf-8")
    tokenizer = XMIDIGetmusicTokenizer(dict_path=str(dictionary))
    assert tokenizer.vocab_size == VOCAB_SIZE
    assert tokenizer.pad_token_id == PAD_ID
    assert tokenizer.unk_token_id == UNK_ID
    assert tokenizer.mask_token_id == MASK_ID
    space_token_id = next(
        index for index in range(DICT_LEN)
        if " " in tokenizer._convert_id_to_token(index)
    )
    space_token = tokenizer._convert_id_to_token(space_token_id)
    assert tokenizer._convert_token_to_id(space_token) == space_token_id
    assert " " in space_token
    assert ROW_NAMES == (
        "lead_pitch", "lead_duration", "chord_root", "chord_kind",
        "piano_pitch", "piano_duration", "guitar_pitch", "guitar_duration",
        "bass_pitch", "bass_duration", "string_pitch", "string_duration",
        "wind_pitch", "wind_duration", "drum_pitch", "drum_duration",
    )
    assert TARGET_ROW_PAIRS == (
        (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15)
    )


def test_region_masks_use_four_source_and_twelve_target_rows():
    grid = torch.zeros(1, NUM_ROWS, NUM_COLS, dtype=torch.long)
    grid[:, 5, 100] = UNK_ID
    grid[:, 15] = UNK_ID
    region, row_unk = build_region_masks(grid.reshape(1, -1))
    region = region.reshape(1, NUM_ROWS, NUM_COLS)
    assert not region[:, :NUM_SOURCE_ROWS].any()
    assert not region[:, NUM_SOURCE_ROWS:, :NUM_PROMPT_COLS].any()
    assert region[0, 4, NUM_PROMPT_COLS]
    assert not region[0, 5, 100]
    assert not region[0, 15].any()
    assert row_unk[0, 15]
    assert int(region.sum()) == 11 * (NUM_COLS - NUM_PROMPT_COLS) - 1


def test_dense_attention_mask_excludes_inactive_kv_and_has_safe_queries():
    row_unk = torch.tensor([[False, True]])
    mask = build_xmidi_attn_mask(
        seqlen=8,
        block_size=2,
        row_unk_mask=row_unk,
        num_cols=4,
    )
    assert mask.shape == (1, 1, 16, 16)
    active = torch.tensor([True] * 4 + [False] * 4 + [True] * 4 + [False] * 4)
    assert not mask[0, 0, active][:, ~active].any()
    assert torch.all(mask[0, 0, ~active].sum(dim=-1) == 1)


def test_axis_local_rope_has_no_cross_axis_leakage():
    rope = Rotary2D(64)
    qkv = torch.randn(1, SEQ_LEN, 3, 1, 64)
    cos, sin, axis_dims = rope(qkv)
    assert axis_dims == (32, 32)
    assert cos.shape == (1, SEQ_LEN, 3, 1, 64)

    def rotate_half(value):
        first, second = value.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)

    row, col = qkv[..., :32], qkv[..., 32:]
    rotated = torch.cat(
        (
            row * cos[..., :32] + rotate_half(row) * sin[..., :32],
            col * cos[..., 32:] + rotate_half(col) * sin[..., 32:],
        ),
        dim=-1,
    )
    qkv_without_col = qkv.clone()
    qkv_without_col[..., 32:] = 0
    row2, col2 = qkv_without_col[..., :32], qkv_without_col[..., 32:]
    rotated_without_col = torch.cat(
        (
            row2 * cos[..., :32] + rotate_half(row2) * sin[..., :32],
            col2 * cos[..., 32:] + rotate_half(col2) * sin[..., 32:],
        ),
        dim=-1,
    )
    assert not rotated_without_col[..., 32:].any()
    assert torch.allclose(
        rotated.square().sum(dim=-1),
        qkv.square().sum(dim=-1),
        atol=3e-4,
        rtol=3e-4,
    )
    assert torch.equal(rotated[:, :, 2], qkv[:, :, 2])


def test_seven_point_audit_accepts_valid_output_and_rejects_mask():
    initial = torch.zeros(1, NUM_ROWS, NUM_COLS, dtype=torch.long)
    initial[:, 4:, NUM_PROMPT_COLS:] = MASK_ID
    initial[:, 15] = UNK_ID
    generation = initial == MASK_ID
    output = initial.clone()
    output[generation] = 1
    row_unk = torch.zeros(1, NUM_ROWS, dtype=torch.bool)
    row_unk[:, 15] = True
    runtime = {
        "attention_mask_valid": True,
        "current_query_contract": True,
        "cache_consistent": True,
        "absolute_rope": True,
    }
    record = audit_xmidi_batch(
        output.reshape(1, -1),
        initial.reshape(1, -1),
        generation.reshape(1, -1),
        row_unk,
        runtime,
    )[0]
    assert record["all_pass"]
    fixed_corruption = output.clone()
    fixed_corruption[0, 4, 0] = 7
    fixed_record = audit_xmidi_batch(
        fixed_corruption.reshape(1, -1),
        initial.reshape(1, -1),
        generation.reshape(1, -1),
        row_unk,
        runtime,
    )[0]
    assert not fixed_record["all_pass"]
    assert "fixed_condition_invariance" in fixed_record["failures"]

    output[0, 4, NUM_PROMPT_COLS] = MASK_ID
    record = audit_xmidi_batch(
        output.reshape(1, -1),
        initial.reshape(1, -1),
        generation.reshape(1, -1),
        row_unk,
        runtime,
    )[0]
    assert not record["all_pass"]
    assert "diffusion_completion" in record["failures"]


def test_match_metrics_exclude_structural_cell_unk():
    from scripts.eval.batch_infer_xmidi_getmusic import quick_match_stats
    from scripts.eval.objective_evaluation import paired_content_tokens

    gt = [["a"] * NUM_COLS for _ in range(NUM_ROWS)]
    gen = [["a"] * NUM_COLS for _ in range(NUM_ROWS)]
    gt[4][NUM_PROMPT_COLS] = "<UNK>"
    gen[4][NUM_PROMPT_COLS] = "wrong-but-fixed"
    stats = quick_match_stats(gt, gen)
    assert stats["all_target_content"]["match"] == 1.0
    assert stats["all_target_content"]["n_positions"] == 5951

    row_unk = [False] * NUM_ROWS
    row_unk[15] = True
    gt_tokens, gen_tokens = paired_content_tokens(gt, gen, row_unk)
    assert len(gt_tokens) == len(gen_tokens) == 5455
    assert "<UNK>" not in gt_tokens


def test_generation_mask_rle_and_per_sample_seed_are_stable():
    mask = torch.zeros(NUM_ROWS, NUM_COLS, dtype=torch.bool)
    mask[4, 16:32] = True
    mask[12, 100:103] = True
    runs = mask_to_rle(mask)
    assert torch.equal(rle_to_mask(runs).reshape(NUM_ROWS, NUM_COLS), mask)
    assert validate_generation_mask(mask).shape == (1, NUM_ROWS, NUM_COLS)
    assert derive_sample_seed(1, 42) == derive_sample_seed(1, 42)
    assert derive_sample_seed(1, 42) != derive_sample_seed(1, 43)

    invalid = mask.clone()
    invalid[0, 0] = True
    with pytest.raises(ValueError, match="source"):
        validate_generation_mask(invalid)
    condition = torch.zeros(1, NUM_ROWS, NUM_COLS, dtype=torch.long)
    condition[0, 4, 16] = UNK_ID
    with pytest.raises(ValueError, match="UNK"):
        validate_generation_mask(mask.unsqueeze(0), condition)


def test_explicit_daw_condition_grid_preserves_fixed_target_content():
    pytest.importorskip("omegaconf")
    from omegaconf import OmegaConf
    from diffusion import Diffusion

    dummy = type("Dummy", (), {})()
    dummy.config = OmegaConf.create({"data": {"train": "xmidi_getmusic"}})
    dummy.mask_index = MASK_ID

    source = torch.arange(NUM_SOURCE_ROWS * NUM_COLS).reshape(
        1, NUM_SOURCE_ROWS, NUM_COLS
    ) % DICT_LEN
    prompts = torch.zeros(
        1, NUM_ROWS - NUM_SOURCE_ROWS, NUM_PROMPT_COLS, dtype=torch.long
    )
    condition = torch.full((1, NUM_ROWS, NUM_COLS), PAD_ID, dtype=torch.long)
    condition[:, :NUM_SOURCE_ROWS] = source
    condition[:, NUM_SOURCE_ROWS:, :NUM_PROMPT_COLS] = prompts
    condition[:, 4, 40] = 1234
    condition[:, 15] = UNK_ID
    prompts[:, -1] = UNK_ID
    generation = torch.zeros_like(condition, dtype=torch.bool)
    generation[:, 4, 16:32] = True
    row_unk = torch.zeros(1, NUM_ROWS, dtype=torch.bool)
    row_unk[:, 15] = True

    prior = Diffusion._xmidi_sample_prior(
        dummy,
        source,
        prompts,
        row_unk,
        condition_grid=condition,
        generation_mask=generation,
    ).reshape(1, NUM_ROWS, NUM_COLS)
    assert torch.all(prior[generation] == MASK_ID)
    assert prior[0, 4, 40] == 1234
    assert torch.equal(prior[:, :NUM_SOURCE_ROWS], source)
    assert torch.all(prior[:, 15] == UNK_ID)

    conflicting_unk = torch.zeros(
        1, NUM_ROWS - NUM_SOURCE_ROWS, NUM_COLS - NUM_PROMPT_COLS,
        dtype=torch.bool,
    )
    conflicting_unk[0, 0, 40 - NUM_PROMPT_COLS] = True
    with pytest.raises(ValueError, match="content_unk_mask differs"):
        Diffusion._xmidi_sample_prior(
            dummy,
            source,
            prompts,
            row_unk,
            content_unk_mask=conflicting_unk,
            condition_grid=condition,
            generation_mask=generation,
        )


def test_bleu_fields_are_cumulative_scores_not_ngram_precisions():
    from scripts.eval.objective_evaluation import sentence_bleu_scores

    precisions, bleu = sentence_bleu_scores(
        ["a", "b", "c", "d"], ["a", "b", "x", "d"]
    )
    assert bleu[0] == pytest.approx(precisions[0])
    assert bleu[1] == pytest.approx((precisions[0] * precisions[1]) ** 0.5)
    assert bleu[1] != precisions[1]
    exact_precisions, exact_bleu = sentence_bleu_scores(
        ["a", "b", "c", "d"], ["a", "b", "c", "d"]
    )
    assert exact_precisions == [1.0] * 4
    assert exact_bleu == [1.0] * 4

    from scripts.eval.objective_evaluation import segmented_bleu_scores
    segmented_precision, _ = segmented_bleu_scores(
        [["a"], ["b"]], [["a"], ["x"]], max_order=2
    )
    flattened_precision, _ = sentence_bleu_scores(
        ["a", "b"], ["a", "x"], max_order=2
    )
    assert segmented_precision[1] == 0.0
    assert flattened_precision[1] > 0.0


def test_standard_nucleus_keeps_threshold_crossing_token():
    pytest.importorskip("omegaconf")
    from omegaconf import OmegaConf
    from diffusion import Diffusion

    dummy = type("Dummy", (), {})()
    dummy.config = OmegaConf.create({"sampling": {"nucleus_p": 0.7}})
    dummy.block_size = 1
    probabilities = torch.tensor([[[0.60, 0.25, 0.15]]])
    filtered = Diffusion._nucleus_sample(dummy, probabilities.clone())
    assert filtered[0, 0, 0] == pytest.approx(0.60 / 0.85)
    assert filtered[0, 0, 1] == pytest.approx(0.25 / 0.85)
    assert filtered[0, 0, 2] == 0.0
    assert filtered.sum() == pytest.approx(1.0)


def test_v1_checkpoint_is_rejected_before_model_construction(tmp_path):
    from xmidi_getmusic.eval_protocol import inspect_v2_checkpoint

    path = tmp_path / "v1.ckpt"
    torch.save(
        {
            "state_dict": {
                "backbone.vocab_embed.embedding": torch.empty(32695, 8)
            },
            "epoch": 0,
            "global_step": 1,
        },
        path,
    )
    with pytest.raises(ValueError, match="V1 checkpoint"):
        inspect_v2_checkpoint(path, compute_sha256=False)


def test_tied_selected_projection_matches_dense_softmax():
    pytest.importorskip("omegaconf")
    from models.dit import DDiTFinalLayer

    torch.manual_seed(7)
    layer = DDiTFinalLayer(8, 17, 4, adaLN=False, tie_word_embeddings=True)
    embedding = torch.randn(17, 8)
    hidden = torch.randn(2, 5, 8)
    targets = torch.randint(0, 16, (2, 5))
    selected = torch.tensor(
        [[True, False, True, False, False], [False, True, False, False, True]]
    )
    dense = layer(hidden, None, embedding)
    dense[:, :, 16] = torch.finfo(dense.dtype).min
    expected = dense.log_softmax(-1).gather(-1, targets[..., None]).squeeze(-1)
    actual = layer.selected_log_probs(
        hidden,
        None,
        targets,
        selected,
        embedding,
        excluded_token_id=16,
        chunk_size=2,
    )
    assert layer.linear is None
    assert torch.allclose(actual[selected], expected[selected], atol=1e-6)
    assert not actual[~selected].any()


def test_empty_selected_projection_returns_differentiable_zero():
    pytest.importorskip("omegaconf")
    from models.dit import DDiTFinalLayer

    layer = DDiTFinalLayer(8, 17, 4, adaLN=False, tie_word_embeddings=True)
    embedding = torch.randn(17, 8, requires_grad=True)
    hidden = torch.randn(1, 5, 8, requires_grad=True)
    targets = torch.zeros(1, 5, dtype=torch.long)
    selected = torch.zeros_like(targets, dtype=torch.bool)

    output = layer.selected_log_probs(
        hidden, None, targets, selected, embedding, excluded_token_id=16
    )
    loss = output.sum()
    assert loss.requires_grad
    assert loss.item() == 0.0
    loss.backward()
    assert hidden.grad is not None and not hidden.grad.any()
    assert embedding.grad is not None and not embedding.grad.any()
    assert layer.bias.grad is not None and not layer.bias.grad.any()
