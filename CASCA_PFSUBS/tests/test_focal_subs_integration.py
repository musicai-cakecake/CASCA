from __future__ import annotations

import ast
import inspect
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir

import dataloader
from diffusion import Diffusion
from main import _translate_use_focal_cli
from xmidi_getmusic.constants import MASK_ID, PAD_ID
from xmidi_getmusic.focal_subs_loss import (
    FocalSubsLossConfig,
    focal_subs_token_loss,
)

ROOT = Path(__file__).resolve().parents[1]


class _FixedNoise:
    def __init__(self, loss_scale: torch.Tensor, move_chance: torch.Tensor):
        self.loss_scale = loss_scale
        self.move_chance = move_chance

    def __call__(self, _t):
        return self.loss_scale, self.move_chance


class _LossReductionHarness:
    def __init__(self):
        self.parameterization = "subs"
        self.ignore_bos = False
        self.training = True
        self.config = SimpleNamespace(data=SimpleNamespace(train="xmidi_getmusic"))
        self.backbone = SimpleNamespace()
        self._last_diffusion_objective = None
        self._last_diffusion_diagnostics = None

    def _maybe_sub_sample(self, x0, attention_mask):
        return x0, x0, attention_mask

    def _forward_pass_diffusion(self, input_tokens, **kwargs):
        del input_tokens, kwargs
        base = torch.tensor(
            [[2.0, 4.0, 0.0], [3.0, 0.0, 9.0]], requires_grad=True
        )
        self._last_diffusion_objective = torch.tensor(
            [[1.0, 2.0, 0.0], [6.0, 0.0, 3.0]], requires_grad=True
        )
        self._last_diffusion_diagnostics = {"selected_count": torch.tensor(4)}
        return base


class _EndpointHarness:
    def __init__(self, *, use_focal: bool):
        self.config = SimpleNamespace(data=SimpleNamespace(train="xmidi_getmusic"))
        self.parameterization = "subs"
        self.mask_index = MASK_ID
        self.mdlm_loss_scale = False
        self.ignore_bos = False
        self.cross_attn = True
        self.noise = _FixedNoise(
            torch.tensor([[-2.0], [-5.0]], dtype=torch.float64),
            torch.tensor([[0.5], [0.5]], dtype=torch.float64),
        )
        self.focal_loss_config = FocalSubsLossConfig(
            use_focal=use_focal,
            gamma=2.0,
            alpha_pad=0.86,
            alpha_event=1.29,
        )
        self._last_diffusion_objective = None
        self._last_diffusion_diagnostics = None
        self.selected_log_p = torch.tensor(
            [[-0.1, -2.0, 0.0], [-0.7, 0.0, -4.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        self.xt = torch.tensor(
            [[MASK_ID, MASK_ID, 123], [MASK_ID, PAD_ID, MASK_ID]],
            dtype=torch.long,
        )

    def _sample_t(self, *args, **kwargs):
        raise AssertionError("test supplies t explicitly")

    def _sigma_from_p(self, p):
        return p

    def q_xt(self, *args, **kwargs):
        return self.xt

    def _selected_subs_log_probs(self, x, sigma, targets, output_positions):
        del x, sigma, targets
        expected = torch.tensor(
            [[True, True, False], [True, False, True]], dtype=torch.bool
        )
        assert torch.equal(output_positions, expected)
        return self.selected_log_p


def _endpoint_output(harness: _EndpointHarness):
    x0 = torch.tensor(
        [[PAD_ID, 17, 123], [42, PAD_ID, PAD_ID]], dtype=torch.long
    )
    region = torch.ones_like(x0, dtype=torch.bool)
    t = torch.tensor([[0.5], [0.5]], dtype=torch.float64)
    output = Diffusion._forward_pass_diffusion(
        harness, x0, t=t, region_mask=region
    )
    selected = harness.xt.eq(MASK_ID) & region
    return output, x0, selected


def test_hydra_default_loss_namespace_is_legacy_disabled_with_focal_defaults():
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        config = compose(
            config_name="config",
            overrides=[
                "data=xmidi_getmusic",
                "model=xmidi_getmusic_tiny",
                "algo=xmidi_getmusic_bd3lm",
                "block_size=16",
            ],
        )
    resolved = FocalSubsLossConfig.from_mapping(config.loss)
    assert resolved.use_focal is False
    assert resolved.gamma == 2.0
    assert resolved.alpha_pad == 0.8600010833864916
    assert resolved.alpha_event == 1.2900016250797375
    assert config.loss.pad_token_id == PAD_ID
    assert config.trainer.use_distributed_sampler is False


def test_hydra_focal_override_resolves_exact_config():
    overrides = [
        "data=xmidi_getmusic",
        "model=xmidi_getmusic_tiny",
        "algo=xmidi_getmusic_bd3lm",
        "block_size=16",
        "loss.use_focal=true",
        "loss.gamma=2.0",
        "loss.alpha_pad=0.8600010833864916",
        "loss.alpha_event=1.2900016250797375",
        "loss.pad_token_id=605",
        "loss.alpha_normalization=none",
    ]
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        config = compose(config_name="config", overrides=overrides)
    resolved = FocalSubsLossConfig.from_mapping(config.loss)
    assert resolved.use_focal is True
    assert resolved.gamma == 2.0
    assert resolved.alpha_pad == 0.8600010833864916
    assert resolved.alpha_event == 1.2900016250797375
    assert resolved.pad_token_id == PAD_ID


@pytest.mark.parametrize(
    "argv,expected",
    [
        (["main.py", "--use_focal", "True"], "loss.use_focal=true"),
        (["main.py", "--use_focal=False"], "loss.use_focal=false"),
        (["main.py", "--use_focal", "1"], "loss.use_focal=true"),
        (["main.py", "loss.use_focal=false"], "loss.use_focal=false"),
    ],
)
def test_public_use_focal_cli_translates_to_hydra(argv, expected):
    translated = _translate_use_focal_cli(argv)
    assert expected in translated
    assert not any(argument.startswith("--use_focal") for argument in translated)


def test_public_use_focal_cli_rejects_missing_invalid_or_duplicate_values():
    with pytest.raises(ValueError, match="requires"):
        _translate_use_focal_cli(["main.py", "--use_focal"])
    with pytest.raises(ValueError, match="expects"):
        _translate_use_focal_cli(["main.py", "--use_focal=maybe"])
    with pytest.raises(ValueError, match="only once"):
        _translate_use_focal_cli(
            ["main.py", "--use_focal=true", "loss.use_focal=false"]
        )


def test_integrated_false_branch_matches_direct_v2_forward_and_gradient_exactly():
    harness = _EndpointHarness(use_focal=False)
    output, _, _ = _endpoint_output(harness)
    expected_input = harness.selected_log_p.detach().clone().requires_grad_(True)
    expected = harness.noise.loss_scale * expected_input
    assert torch.equal(output, expected)
    weights = torch.arange(1, output.numel() + 1, dtype=output.dtype).reshape_as(output)
    actual_gradient = torch.autograd.grad(
        (output * weights).sum(), harness.selected_log_p
    )[0]
    expected_gradient = torch.autograd.grad(
        (expected * weights).sum(), expected_input
    )[0]
    assert torch.equal(actual_gradient, expected_gradient)


def test_integrated_focal_branch_preserves_base_and_caches_objective_exactly():
    harness = _EndpointHarness(use_focal=True)
    base_output, targets, selected = _endpoint_output(harness)
    expected_base = harness.noise.loss_scale * harness.selected_log_p
    expected_objective = focal_subs_token_loss(
        harness.selected_log_p,
        targets,
        harness.noise.loss_scale,
        selected,
        harness.focal_loss_config,
    )
    assert torch.equal(base_output, expected_base)
    assert torch.equal(harness._last_diffusion_objective, expected_objective)
    gradient = torch.autograd.grad(
        harness._last_diffusion_objective.sum(), harness.selected_log_p
    )[0]
    assert torch.isfinite(gradient).all()


def test_loss_reduction_reports_base_nll_and_optimizes_focal_objective():
    harness = _LossReductionHarness()
    x0 = torch.zeros((2, 3), dtype=torch.long)
    region = torch.tensor([[True, True, False], [True, False, True]])
    losses = Diffusion._loss(harness, x0, region)
    assert torch.equal(
        losses.nlls, torch.tensor([[2.0, 4.0, 0.0], [3.0, 0.0, 9.0]])
    )
    assert torch.equal(
        losses.objective_nlls,
        torch.tensor([[1.0, 2.0, 0.0], [6.0, 0.0, 3.0]]),
    )
    assert losses.base_loss.item() == pytest.approx((3.0 + 6.0) / 2.0)
    assert losses.loss.item() == pytest.approx((1.5 + 4.5) / 2.0)
    assert losses.diagnostics["selected_count"].item() == 4
    assert harness._last_diffusion_objective is None
    assert harness._last_diffusion_diagnostics is None


def test_integration_preserves_selected_projection_and_ng_reduction_contract():
    source = inspect.getsource(Diffusion)
    tree = ast.parse(textwrap.dedent(source))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "focal_subs_token_loss"
    ]
    assert len(calls) == 1
    assert "region_sizes = token_mask.sum(dim=-1)" in source
    assert "nlls.sum(dim=-1) / region_sizes.clamp_min(1)" in source
    assert "objective_nlls.sum(dim=-1) / region_sizes.clamp_min(1)" in source
    assert "self.metrics.valid_nlls.update(losses.nlls" in source
    dit_source = (ROOT / "models" / "dit.py").read_text(encoding="utf-8")
    assert "focal_subs_loss" not in dit_source


def test_checkpoint_hook_writes_only_focal_metadata_not_rng_or_sampler_state():
    save_source = inspect.getsource(Diffusion.on_save_checkpoint)
    load_source = inspect.getsource(Diffusion.on_load_checkpoint)
    train_source = inspect.getsource(Diffusion.on_train_start)
    assert "checkpoint['focal_loss_config']" in save_source
    assert "self.focal_loss_config.to_dict()" in save_source
    assert "checkpoint['sampler']" not in save_source
    assert "checkpoint['rng_state']" not in save_source
    assert "_checkpoint_sampler_state" not in load_source
    assert "_checkpoint_rng_state" not in load_source
    assert "set_rng_state" not in train_source


def test_random_fault_tolerant_sampler_restores_exact_remaining_order():
    dataset = list(range(31))
    generator = torch.Generator().manual_seed(20260730)
    sampler = dataloader.RandomFaultTolerantSampler(
        dataset, generator=generator
    )
    iterator = iter(sampler)
    consumed = [next(iterator) for _ in range(7)]
    state = sampler.state_dict()
    expected_remaining = list(iterator)

    restored = dataloader.RandomFaultTolerantSampler(
        dataset, generator=torch.Generator().manual_seed(1)
    )
    restored.load_state_dict(state)
    actual_remaining = list(iter(restored))
    assert len(consumed) == 7
    assert actual_remaining == expected_remaining


def test_distributed_fault_tolerant_sampler_restores_exact_remaining_order():
    dataset = list(range(32))
    sampler = dataloader.FaultTolerantDistributedSampler(
        dataset, num_replicas=2, rank=1, shuffle=True, seed=17
    )
    iterator = iter(sampler)
    consumed = [next(iterator) for _ in range(5)]
    state = sampler.state_dict()
    expected_remaining = list(iterator)

    restored = dataloader.FaultTolerantDistributedSampler(
        dataset, num_replicas=2, rank=1, shuffle=True, seed=17
    )
    restored.load_state_dict(state)
    actual_remaining = list(iter(restored))
    assert len(consumed) == 5
    assert actual_remaining == expected_remaining
