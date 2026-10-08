from __future__ import annotations

import ast
import inspect
import math
from pathlib import Path

import pytest
import torch

from xmidi_getmusic.constants import PAD_ID
from xmidi_getmusic.focal_subs_loss import (
    ALPHA_NORMALIZATION_NONE,
    ALPHA_NORMALIZATION_TRAIN_PRIOR,
    FocalSubsLossConfig,
    expected_group_alpha,
    focal_subs_token_loss,
    resolve_group_alpha,
    stable_one_minus_probability,
)

TRAIN_PAD_PROBABILITY = 217_999_140 / 323_238_700


def _example(dtype: torch.dtype = torch.float64, device: str = "cpu"):
    selected = torch.tensor(
        [[True, True, False], [True, False, True]], device=device
    )
    targets = torch.tensor(
        [[PAD_ID, 17, 123], [42, PAD_ID, PAD_ID]], device=device
    )
    values = torch.tensor(
        [[-0.1, -2.0, 0.0], [-0.7, 0.0, -4.0]],
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    scale = torch.tensor(
        [[-2.0], [-5.0]], dtype=torch.float32, device=device
    )
    return values, targets, scale, selected


def test_config_schema_defaults_to_strict_legacy_branch():
    config = FocalSubsLossConfig()
    assert config.use_focal is False
    assert config.pad_token_id == PAD_ID
    assert config.to_dict()["gamma"] == 2.0
    assert FocalSubsLossConfig.from_mapping(config.to_dict()) == config
    disabled_with_stale_values = FocalSubsLossConfig(
        use_focal=False,
        gamma=float("nan"),
        alpha_pad=-1.0,
        alpha_event=float("inf"),
        pad_token_id=604,
        alpha_normalization="stale-disabled-value",
    )
    assert disabled_with_stale_values.use_focal is False


@pytest.mark.parametrize(
    "values,error",
    [
        ({"use_focal": "true"}, TypeError),
        ({"use_focal": True, "gamma": -1.0}, ValueError),
        ({"use_focal": True, "gamma": float("nan")}, ValueError),
        ({"use_focal": True, "alpha_pad": float("inf")}, ValueError),
        ({"use_focal": True, "alpha_event": -0.1}, ValueError),
        ({"use_focal": True, "pad_token_id": -1}, ValueError),
        ({"use_focal": True, "pad_token_id": 604}, ValueError),
        ({"use_focal": True, "pad_token_id": 605.0}, TypeError),
        ({"use_focal": True, "alpha_normalization": "dynamic_batch"}, ValueError),
        ({"use_focal": True, "alpha_normalization": ALPHA_NORMALIZATION_TRAIN_PRIOR}, ValueError),
        ({"unknown": 1}, ValueError),
    ],
)
def test_config_rejects_invalid_or_unknown_values(values, error):
    with pytest.raises(error):
        FocalSubsLossConfig.from_mapping(values)


@pytest.mark.parametrize("ratio", [1.0, 1.5, 2.0, TRAIN_PAD_PROBABILITY / (1.0 - TRAIN_PAD_PROBABILITY)])
def test_normalized_group_alpha_has_requested_ratio_and_unit_expectation(ratio):
    resolved = resolve_group_alpha(TRAIN_PAD_PROBABILITY, ratio)
    assert resolved.normalization == ALPHA_NORMALIZATION_TRAIN_PRIOR
    assert resolved.alpha_event / resolved.alpha_pad == pytest.approx(ratio, abs=1e-15)
    assert resolved.expected_alpha == pytest.approx(1.0, abs=2e-16)
    assert expected_group_alpha(
        TRAIN_PAD_PROBABILITY, resolved.alpha_pad, resolved.alpha_event
    ) == pytest.approx(1.0, abs=2e-16)


def test_unnormalized_group_alpha_preserves_pad_weight_one():
    resolved = resolve_group_alpha(
        TRAIN_PAD_PROBABILITY, 1.5, normalize_expected_alpha=False
    )
    assert resolved.normalization == ALPHA_NORMALIZATION_NONE
    assert resolved.alpha_pad == 1.0
    assert resolved.alpha_event == 1.5
    assert resolved.expected_alpha > 1.0


@pytest.mark.parametrize(
    "pad_probability,ratio",
    [(0.0, 1.0), (1.0, 1.0), (0.5, 0.0), (float("nan"), 1.0)],
)
def test_group_alpha_rejects_invalid_prior_or_ratio(pad_probability, ratio):
    with pytest.raises((TypeError, ValueError)):
        resolve_group_alpha(pad_probability, ratio)


def test_stable_one_minus_probability_is_accurate_near_one():
    log_probability = torch.tensor([-1e-12, -1e-8, -1e-4, -10.0], dtype=torch.float64)
    actual = stable_one_minus_probability(log_probability)
    expected = -torch.expm1(log_probability)
    assert torch.equal(actual, expected)
    assert actual[0] > 0


def test_stable_one_minus_probability_rejects_invalid_positive_log_probability():
    with pytest.raises(ValueError, match="must be <= 0"):
        stable_one_minus_probability(torch.tensor([1e-3], dtype=torch.float32))


def test_extreme_target_probabilities_have_finite_forward_and_backward():
    log_p = torch.tensor(
        [[-1e-12, -1e-8, -20.0, -100.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    targets = torch.tensor([[PAD_ID, 1, 2, 3]])
    selected = torch.ones_like(targets, dtype=torch.bool)
    scale = torch.tensor([[-1000.0]], dtype=torch.float64)
    output = focal_subs_token_loss(
        log_p,
        targets,
        scale,
        selected,
        FocalSubsLossConfig(
            use_focal=True,
            gamma=2.0,
            alpha_pad=0.86,
            alpha_event=1.29,
        ),
    )
    gradient = torch.autograd.grad(output.sum(), log_p)[0]
    assert torch.isfinite(output).all()
    assert torch.isfinite(gradient).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_false_branch_is_elementwise_and_gradient_exact_to_v2(dtype):
    log_p, targets, loss_scale, selected = _example(dtype=dtype)
    expected_input = log_p.detach().clone().requires_grad_(True)
    expected = loss_scale * expected_input
    actual = focal_subs_token_loss(
        log_p,
        targets,
        loss_scale,
        selected,
        FocalSubsLossConfig(use_focal=False),
    )
    assert torch.equal(actual, expected)
    weights = torch.arange(1, actual.numel() + 1, dtype=actual.dtype).reshape_as(actual)
    actual_gradient = torch.autograd.grad((actual * weights).sum(), log_p)[0]
    expected_gradient = torch.autograd.grad((expected * weights).sum(), expected_input)[0]
    assert torch.equal(actual_gradient, expected_gradient)


def test_false_branch_has_no_focal_specific_runtime_validation():
    log_p = torch.tensor([[float("nan"), -1.0]], requires_grad=True)
    targets = torch.tensor([[PAD_ID, 1]])
    selected = torch.tensor([[False, True]])
    positive_scale = torch.tensor([[2.0]])
    actual = focal_subs_token_loss(
        log_p,
        targets,
        positive_scale,
        selected,
        FocalSubsLossConfig(use_focal=False),
    )
    expected = positive_scale * log_p
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    assert torch.equal(actual[~torch.isnan(actual)], expected[~torch.isnan(expected)])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_gamma_zero_unit_alpha_is_elementwise_and_gradient_exact_to_v2(dtype):
    log_p, targets, loss_scale, selected = _example(dtype=dtype)
    expected_input = log_p.detach().clone().requires_grad_(True)
    expected = loss_scale * expected_input
    actual = focal_subs_token_loss(
        log_p,
        targets,
        loss_scale,
        selected,
        FocalSubsLossConfig(
            use_focal=True,
            gamma=0.0,
            alpha_pad=1.0,
            alpha_event=1.0,
        ),
    )
    assert torch.equal(actual, expected)
    weights = torch.arange(1, actual.numel() + 1, dtype=actual.dtype).reshape_as(actual)
    actual_gradient = torch.autograd.grad((actual * weights).sum(), log_p)[0]
    expected_gradient = torch.autograd.grad((expected * weights).sum(), expected_input)[0]
    assert torch.equal(actual_gradient, expected_gradient)


def test_float64_focal_matches_manual_multiclass_reference():
    log_p, targets, loss_scale, selected = _example(dtype=torch.float64)
    config = FocalSubsLossConfig(
        use_focal=True,
        gamma=2.0,
        alpha_pad=0.86,
        alpha_event=1.29,
    )
    actual = focal_subs_token_loss(log_p, targets, loss_scale, selected, config)
    expected = loss_scale * log_p
    expected = expected.clone()
    for row, column in selected.nonzero(as_tuple=False).tolist():
        probability = math.exp(float(log_p[row, column].detach()))
        alpha = (
            config.alpha_pad
            if int(targets[row, column]) == PAD_ID
            else config.alpha_event
        )
        expected[row, column] *= alpha * (1.0 - probability) ** config.gamma
    assert torch.allclose(actual, expected, rtol=1e-14, atol=1e-14)


def test_pad_event_alpha_mapping_for_pad_only_event_only_and_mixed_targets():
    selected = torch.ones((1, 4), dtype=torch.bool)
    log_p = torch.full((1, 4), -math.log(2.0), dtype=torch.float64)
    scale = torch.full((1, 1), -1.0, dtype=torch.float64)
    config = FocalSubsLossConfig(
        use_focal=True,
        gamma=0.0,
        alpha_pad=0.5,
        alpha_event=2.0,
    )
    pad_targets = torch.full((1, 4), PAD_ID, dtype=torch.long)
    event_targets = torch.arange(4, dtype=torch.long).reshape(1, 4)
    mixed_targets = torch.tensor([[PAD_ID, 1, PAD_ID, 2]])
    base = -log_p
    assert torch.equal(
        focal_subs_token_loss(log_p, pad_targets, scale, selected, config),
        base * 0.5,
    )
    assert torch.equal(
        focal_subs_token_loss(log_p, event_targets, scale, selected, config),
        base * 2.0,
    )
    mixed = focal_subs_token_loss(log_p, mixed_targets, scale, selected, config)
    assert torch.equal(mixed, base * torch.tensor([[0.5, 2.0, 0.5, 2.0]]))


def test_focal_factor_suppresses_easy_target_more_than_hard_target():
    log_p = torch.tensor([[-0.01, -3.0]], dtype=torch.float64)
    targets = torch.tensor([[1, 2]])
    selected = torch.ones_like(targets, dtype=torch.bool)
    scale = torch.tensor([[-1.0]], dtype=torch.float64)
    focal = focal_subs_token_loss(
        log_p,
        targets,
        scale,
        selected,
        FocalSubsLossConfig(use_focal=True, gamma=2.0),
    )
    base = scale * log_p
    multiplier = focal / base
    assert multiplier[0, 0] < multiplier[0, 1]
    assert multiplier[0, 0] < 1e-3
    assert multiplier[0, 1] > 0.9


def test_unselected_positions_preserve_exact_legacy_graph_zero():
    log_p, targets, loss_scale, selected = _example(dtype=torch.float64)
    output = focal_subs_token_loss(
        log_p,
        targets,
        loss_scale,
        selected,
        FocalSubsLossConfig(use_focal=True, gamma=2.0),
    )
    assert torch.equal(output[~selected], torch.zeros_like(output[~selected]))
    gradient = torch.autograd.grad(output.sum(), log_p)[0]
    expected_legacy_gradient = loss_scale.expand_as(log_p)
    assert torch.equal(
        gradient[~selected], expected_legacy_gradient[~selected]
    )


@pytest.mark.parametrize("gamma", [0.0, 0.5, 1.0, 2.0])
def test_empty_selected_set_preserves_graph_connected_zero(gamma):
    parameter = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
    log_p = parameter.sum() * 0.0 + torch.zeros((2, 3), dtype=torch.float64)
    targets = torch.zeros((2, 3), dtype=torch.long)
    selected = torch.zeros((2, 3), dtype=torch.bool)
    scale = torch.full((2, 1), -2.0, dtype=torch.float64)
    output = focal_subs_token_loss(
        log_p,
        targets,
        scale,
        selected,
        FocalSubsLossConfig(use_focal=True, gamma=gamma),
    )
    assert output.requires_grad
    assert torch.equal(output, torch.zeros_like(output))
    output.sum().backward()
    assert parameter.grad is not None
    assert torch.equal(parameter.grad, torch.zeros_like(parameter.grad))
    assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize("gamma", [0.25, 0.5, 0.75])
def test_fractional_gamma_does_not_pollute_unselected_gradients(gamma):
    log_p, targets, loss_scale, selected = _example(dtype=torch.float64)
    output = focal_subs_token_loss(
        log_p,
        targets,
        loss_scale,
        selected,
        FocalSubsLossConfig(use_focal=True, gamma=gamma),
    )
    gradient = torch.autograd.grad(output.sum(), log_p)[0]
    assert torch.isfinite(output).all()
    assert torch.isfinite(gradient).all()
    assert torch.equal(
        gradient[~selected], loss_scale.expand_as(log_p)[~selected]
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fractional_gamma_selected_probability_one_has_finite_gradient(dtype):
    log_p = torch.tensor(
        [[0.0, -1e-12, -0.1]], dtype=dtype, requires_grad=True
    )
    targets = torch.tensor([[PAD_ID, 1, 2]])
    selected = torch.ones_like(targets, dtype=torch.bool)
    loss_scale = torch.tensor([[-2.0]], dtype=dtype)
    output = focal_subs_token_loss(
        log_p,
        targets,
        loss_scale,
        selected,
        FocalSubsLossConfig(use_focal=True, gamma=0.25),
    )
    gradient = torch.autograd.grad(output.sum(), log_p)[0]
    assert torch.isfinite(output).all()
    assert torch.isfinite(gradient).all()
    assert output[0, 0] == 0.0


def test_focal_branch_passes_float64_gradcheck():
    targets = torch.tensor([[PAD_ID, 11, 12]], dtype=torch.long)
    selected = torch.ones_like(targets, dtype=torch.bool)
    scale = torch.tensor([[-2.0]], dtype=torch.float64)
    config = FocalSubsLossConfig(
        use_focal=True,
        gamma=1.5,
        alpha_pad=0.8,
        alpha_event=1.2,
    )

    def function(values):
        return focal_subs_token_loss(values, targets, scale, selected, config)

    values = torch.tensor([[-0.2, -1.0, -4.0]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(function, (values,), eps=1e-6, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize(
    "mutation,error_match",
    [
        ("target_shape", "targets shape"),
        ("mask_shape", "selected_mask shape"),
        ("float_targets", "integer tensor"),
        ("non_bool_mask", "boolean tensor"),
        ("positive_scale", "non-positive"),
        ("bad_scale_shape", "broadcastable"),
        ("unselected_nonzero", "zero outside selected_mask"),
        ("selected_positive", "must be <= 0"),
        ("nonfinite_logp", "finite values"),
        ("negative_target", "non-negative"),
    ],
)
def test_token_loss_rejects_invalid_runtime_contracts(mutation, error_match):
    log_p, targets, scale, selected = _example(dtype=torch.float32)
    if mutation == "target_shape":
        targets = targets[:, :2]
    elif mutation == "mask_shape":
        selected = selected[:, :2]
    elif mutation == "float_targets":
        targets = targets.float()
    elif mutation == "non_bool_mask":
        selected = selected.long()
    elif mutation == "positive_scale":
        scale = scale.abs()
    elif mutation == "bad_scale_shape":
        scale = torch.ones(4)
    elif mutation == "unselected_nonzero":
        log_p = log_p.detach().clone()
        log_p[~selected] = -1.0
    elif mutation == "selected_positive":
        log_p = log_p.detach().clone()
        log_p[selected] = 1e-3
    elif mutation == "nonfinite_logp":
        log_p = log_p.detach().clone()
        log_p[selected] = float("nan")
    elif mutation == "negative_target":
        targets = targets.clone()
        targets[selected] = -1
    with pytest.raises((TypeError, ValueError), match=error_match):
        focal_subs_token_loss(
            log_p,
            targets,
            scale,
            selected,
            FocalSubsLossConfig(use_focal=True),
        )


def test_cpu_bfloat16_forward_and_gradient_are_finite():
    log_p, targets, scale, selected = _example(dtype=torch.bfloat16)
    output = focal_subs_token_loss(
        log_p,
        targets,
        scale,
        selected,
        FocalSubsLossConfig(
            use_focal=True,
            gamma=2.0,
            alpha_pad=0.86,
            alpha_event=1.29,
        ),
    )
    gradient = torch.autograd.grad(output.sum(), log_p)[0]
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()
    assert torch.isfinite(gradient).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_bfloat16_forward_and_gradient_are_finite():
    log_p, targets, scale, selected = _example(dtype=torch.bfloat16, device="cuda")
    output = focal_subs_token_loss(
        log_p,
        targets,
        scale,
        selected,
        FocalSubsLossConfig(
            use_focal=True,
            gamma=2.0,
            alpha_pad=0.86,
            alpha_event=1.29,
        ),
    )
    gradient = torch.autograd.grad(output.sum(), log_p)[0]
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()
    assert torch.isfinite(gradient).all()


def test_module_has_no_vocabulary_projection_or_dense_logits_dependency():
    import xmidi_getmusic.focal_subs_loss as module

    source = inspect.getsource(module)
    tree = ast.parse(source)
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    forbidden_calls = {
        "softmax",
        "log_softmax",
        "one_hot",
        "linear",
        "embedding",
    }
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "models" not in imported_modules
    assert "diffusion" not in imported_modules
    assert forbidden_calls.isdisjoint(called)
    assert "vocab_size" not in inspect.signature(focal_subs_token_loss).parameters
    assert Path(module.__file__).name == "focal_subs_loss.py"
