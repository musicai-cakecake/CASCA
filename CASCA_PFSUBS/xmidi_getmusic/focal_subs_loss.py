"""Standalone PAD-aware multiclass Focal SUBS token loss.

This module consumes exact selected-target log probabilities. It does not own
corruption, vocabulary projection, region construction, or loss reduction.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch

from .constants import PAD_ID

ALPHA_NORMALIZATION_NONE = "none"
ALPHA_NORMALIZATION_TRAIN_PRIOR = "train_prior_expected_one"
_ALLOWED_ALPHA_NORMALIZATIONS = {
    ALPHA_NORMALIZATION_NONE,
    ALPHA_NORMALIZATION_TRAIN_PRIOR,
}


@dataclass(frozen=True)
class GroupAlpha:
    """Resolved PAD/event alpha values and their train-prior expectation."""

    pad_probability: float
    event_probability: float
    event_to_pad_ratio: float
    alpha_pad: float
    alpha_event: float
    expected_alpha: float
    normalization: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FocalSubsLossConfig:
    """Configuration contract for the standalone token-loss router."""

    use_focal: bool = False
    gamma: float = 2.0
    alpha_pad: float = 1.0
    alpha_event: float = 1.0
    pad_token_id: int = PAD_ID
    alpha_normalization: str = ALPHA_NORMALIZATION_NONE

    def __post_init__(self) -> None:
        if type(self.use_focal) is not bool:
            raise TypeError("use_focal must be a bool")
        # Disabled focal settings are intentionally ignored so the legacy V2
        # branch cannot be affected by stale focal-only configuration values.
        if not self.use_focal:
            return
        _require_finite_nonnegative("gamma", self.gamma)
        _require_finite_nonnegative("alpha_pad", self.alpha_pad)
        _require_finite_nonnegative("alpha_event", self.alpha_event)
        if type(self.pad_token_id) is not int:
            raise TypeError("pad_token_id must be an int")
        if self.pad_token_id != PAD_ID:
            raise ValueError(
                f"pad_token_id must equal the V2 vocabulary PAD_ID={PAD_ID}, "
                f"got {self.pad_token_id}"
            )
        if self.alpha_normalization not in _ALLOWED_ALPHA_NORMALIZATIONS:
            raise ValueError(
                "alpha_normalization must be one of "
                f"{sorted(_ALLOWED_ALPHA_NORMALIZATIONS)}, got "
                f"{self.alpha_normalization!r}"
            )
        if self.alpha_normalization == ALPHA_NORMALIZATION_TRAIN_PRIOR:
            raise ValueError(
                "alpha_normalization='train_prior_expected_one' is a resolver "
                "provenance label, not an executable runtime transformation; "
                "pass resolved alpha values with alpha_normalization='none'"
            )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "FocalSubsLossConfig":
        """Build a strict config from a mapping without string-bool coercion."""

        allowed = {
            "use_focal",
            "gamma",
            "alpha_pad",
            "alpha_event",
            "pad_token_id",
            "alpha_normalization",
        }
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"unknown Focal SUBS config keys: {unknown}")
        return cls(**dict(values))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_finite_nonnegative(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite non-negative number")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"{name} must be finite and non-negative, got {value!r}")
    return resolved


def resolve_group_alpha(
    pad_probability: float,
    event_to_pad_ratio: float,
    *,
    normalize_expected_alpha: bool = True,
) -> GroupAlpha:
    """Resolve two-group alpha values from the train PAD prior.

    With normalization enabled, alpha_event / alpha_pad equals the requested
    ratio and E_train[alpha] is one.
    """

    pad_probability = _require_finite_nonnegative(
        "pad_probability", pad_probability
    )
    ratio = _require_finite_nonnegative(
        "event_to_pad_ratio", event_to_pad_ratio
    )
    if not 0.0 < pad_probability < 1.0:
        raise ValueError("pad_probability must be strictly between zero and one")
    if ratio <= 0.0:
        raise ValueError("event_to_pad_ratio must be strictly positive")
    event_probability = 1.0 - pad_probability
    if normalize_expected_alpha:
        denominator = pad_probability + ratio * event_probability
        alpha_pad = 1.0 / denominator
        alpha_event = ratio / denominator
        normalization = ALPHA_NORMALIZATION_TRAIN_PRIOR
    else:
        alpha_pad = 1.0
        alpha_event = ratio
        normalization = ALPHA_NORMALIZATION_NONE
    expected_alpha = (
        pad_probability * alpha_pad + event_probability * alpha_event
    )
    return GroupAlpha(
        pad_probability=pad_probability,
        event_probability=event_probability,
        event_to_pad_ratio=ratio,
        alpha_pad=alpha_pad,
        alpha_event=alpha_event,
        expected_alpha=expected_alpha,
        normalization=normalization,
    )


def expected_group_alpha(
    pad_probability: float,
    alpha_pad: float,
    alpha_event: float,
) -> float:
    """Return E_train[alpha] for a PAD/event prior."""

    pad_probability = _require_finite_nonnegative(
        "pad_probability", pad_probability
    )
    alpha_pad = _require_finite_nonnegative("alpha_pad", alpha_pad)
    alpha_event = _require_finite_nonnegative("alpha_event", alpha_event)
    if not 0.0 <= pad_probability <= 1.0:
        raise ValueError("pad_probability must be in [0, 1]")
    return (
        pad_probability * alpha_pad
        + (1.0 - pad_probability) * alpha_event
    )


def stable_one_minus_probability(log_probability: torch.Tensor) -> torch.Tensor:
    """Compute 1-exp(log_probability) stably in FP32 or FP64."""

    if not torch.is_tensor(log_probability) or not log_probability.is_floating_point():
        raise TypeError("log_probability must be a floating-point tensor")
    compute_dtype = (
        torch.float64 if log_probability.dtype == torch.float64 else torch.float32
    )
    log_probability_compute = log_probability.to(dtype=compute_dtype)
    tolerance = 8.0 * torch.finfo(compute_dtype).eps
    if log_probability_compute.numel():
        maximum = float(log_probability_compute.detach().max().cpu())
        if maximum > tolerance:
            raise ValueError(
                "log_probability must be <= 0 up to floating-point tolerance; "
                f"observed max={maximum}"
            )
    log_probability_compute = torch.minimum(
        log_probability_compute,
        torch.zeros((), dtype=compute_dtype, device=log_probability.device),
    )
    return -torch.expm1(log_probability_compute)


def focal_subs_token_loss(
    log_p_target: torch.Tensor,
    targets: torch.Tensor,
    loss_scale: torch.Tensor,
    selected_mask: torch.Tensor,
    config: FocalSubsLossConfig,
) -> torch.Tensor:
    """Return per-token V2 or PAD-aware Focal SUBS loss.

    `loss_scale` follows V2 and is negative (`-1/t`). `log_p_target` is the
    exact selected clean-target log probability and is zero outside
    `selected_mask`. The returned tensor has the broadcast shape of
    `log_p_target` and `loss_scale`; no reduction is performed here.
    """

    # This expression is intentionally the exact legacy V2 path. Do not run
    # focal-specific validation or emulate it through a multiplier when focal
    # loss is disabled.
    if not config.use_focal:
        return loss_scale * log_p_target

    _validate_focal_inputs(
        log_p_target, targets, loss_scale, selected_mask, config
    )
    base_loss = loss_scale * log_p_target
    selected_mask = selected_mask.bool()
    if not selected_mask.any():
        return base_loss

    selected_log_p = log_p_target[selected_mask]
    compute_dtype = (
        torch.float64 if log_p_target.dtype == torch.float64 else torch.float32
    )
    selected_compute = selected_log_p.to(dtype=compute_dtype)
    tolerance = 8.0 * torch.finfo(compute_dtype).eps
    maximum = float(selected_compute.detach().max().cpu())
    if maximum > tolerance:
        raise ValueError(
            "selected log_p_target must be <= 0 up to floating-point "
            f"tolerance; observed max={maximum}"
        )

    # Compute the power only on selected positions. Unselected positions have
    # log_p=0; evaluating 0**gamma on them would produce an infinite local
    # derivative for 0<gamma<1 and can contaminate backward through 0*inf.
    one_minus_selected = stable_one_minus_probability(selected_log_p)
    gamma = float(config.gamma)
    if 0.0 < gamma < 1.0:
        # A very easy target can round to log(p_y)=0 in float32. The exact
        # fractional-power derivative at 1-p_y=0 is singular and produces
        # 0*inf during backward even though the analytical Focal gradient
        # tends to zero. Clamp only this saturated endpoint to the smallest
        # normal compute value; ordinary probabilities are unchanged.
        one_minus_for_power = one_minus_selected.clamp_min(
            torch.finfo(compute_dtype).tiny
        )
    else:
        one_minus_for_power = one_minus_selected
    focal_factor = one_minus_for_power.pow(gamma)
    selected_targets = targets[selected_mask]
    alpha = torch.where(
        selected_targets.eq(config.pad_token_id),
        torch.as_tensor(
            config.alpha_pad,
            dtype=focal_factor.dtype,
            device=log_p_target.device,
        ),
        torch.as_tensor(
            config.alpha_event,
            dtype=focal_factor.dtype,
            device=log_p_target.device,
        ),
    )
    output = base_loss.clone()
    output[selected_mask] = base_loss[selected_mask] * alpha * focal_factor
    return output


def _validate_focal_inputs(
    log_p_target: torch.Tensor,
    targets: torch.Tensor,
    loss_scale: torch.Tensor,
    selected_mask: torch.Tensor,
    config: FocalSubsLossConfig,
) -> None:
    if not isinstance(config, FocalSubsLossConfig):
        raise TypeError("config must be FocalSubsLossConfig")
    if not torch.is_tensor(log_p_target) or not log_p_target.is_floating_point():
        raise TypeError("log_p_target must be a floating-point tensor")
    if not torch.is_tensor(loss_scale) or not loss_scale.is_floating_point():
        raise TypeError("loss_scale must be a floating-point tensor")
    if not torch.is_tensor(targets) or targets.dtype == torch.bool:
        raise TypeError("targets must be an integer tensor")
    if targets.is_floating_point() or targets.is_complex():
        raise TypeError("targets must be an integer tensor")
    if not torch.is_tensor(selected_mask) or selected_mask.dtype != torch.bool:
        raise TypeError("selected_mask must be a boolean tensor")
    if targets.shape != log_p_target.shape:
        raise ValueError("targets shape must exactly match log_p_target shape")
    if selected_mask.shape != log_p_target.shape:
        raise ValueError("selected_mask shape must exactly match log_p_target shape")
    if targets.device != log_p_target.device:
        raise ValueError("targets and log_p_target must be on the same device")
    if selected_mask.device != log_p_target.device:
        raise ValueError("selected_mask and log_p_target must be on the same device")
    if loss_scale.device != log_p_target.device:
        raise ValueError("loss_scale and log_p_target must be on the same device")
    try:
        broadcast_shape = torch.broadcast_shapes(
            log_p_target.shape, loss_scale.shape
        )
    except RuntimeError as error:
        raise ValueError(
            "loss_scale must be broadcastable to log_p_target"
        ) from error
    if broadcast_shape != log_p_target.shape:
        raise ValueError(
            "loss_scale broadcasting must not expand beyond log_p_target shape"
        )
    if not bool(torch.isfinite(loss_scale).all()):
        raise ValueError("loss_scale must contain only finite values")
    if bool((loss_scale > 0).any()):
        raise ValueError("V2 SUBS loss_scale must be non-positive")
    finite_log_probability = torch.isfinite(log_p_target)
    if not bool(finite_log_probability.all()):
        nonfinite = ~finite_log_probability
        selected_nonfinite = int((nonfinite & selected_mask).sum().detach().cpu())
        unselected_nonfinite = int((nonfinite & ~selected_mask).sum().detach().cpu())
        raise ValueError(
            "log_p_target must contain only finite values: "
            f"selected_nonfinite={selected_nonfinite}, "
            f"unselected_nonfinite={unselected_nonfinite}"
        )
    if selected_mask.any():
        selected_targets = targets[selected_mask]
        if bool((selected_targets < 0).any()):
            raise ValueError("selected targets must be non-negative")
    unselected = log_p_target[~selected_mask]
    if unselected.numel() and bool((unselected != 0).any()):
        raise ValueError(
            "log_p_target must be exactly zero outside selected_mask"
        )
