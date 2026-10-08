#!/usr/bin/env python3
"""Calibrate PAD/event Focal Loss from selected probabilities of the V2 EMA model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import omegaconf
import torch
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from xmidi_getmusic.constants import (  # noqa: E402
    NUM_COLS,
    NUM_ROWS,
    PAD_ID,
    ROW_NAMES,
)
from xmidi_getmusic.eval_protocol import (  # noqa: E402
    canonical_sha256,
    dataset_fingerprint,
    fingerprint_file,
    inspect_v2_checkpoint,
    materialize_inference_weights,
    sha256_file,
)
from xmidi_getmusic.runtime_requirements import require_project_path  # noqa: E402

DEFAULT_CHECKPOINT = Path(os.environ.get(
    "CASCA_BASE_CHECKPOINT", str(REPO.parent / "CASCA_base" / "checkpoints" / "best.ckpt")))
DEFAULT_OUTPUT_DIR = REPO / "artifacts" / "focal_calibration"
DEFAULT_GAMMAS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)
DEFAULT_BIASES = (1.0, 1.05, 1.1, 1.15)
T_BIN_EDGES = (0.001, 0.1, 0.25, 0.5, 0.75, 1.0000001)
T_BIN_NAMES = ("[0.001,0.1)", "[0.1,0.25)", "[0.25,0.5)", "[0.5,0.75)", "[0.75,1.0]")
LOG_P_MIN = -100.0
LOG_P_MAX = 0.0
LOG_P_HIST_BINS = 20_000
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


def register_resolvers() -> None:
    for name, function in (
        ("cwd", os.getcwd),
        ("device_count", torch.cuda.device_count),
        ("eval", eval),
        ("div_up", lambda x, y: (x + y - 1) // y),
    ):
        try:
            omegaconf.OmegaConf.register_new_resolver(name, function)
        except ValueError:
            pass


def build_config(checkpoint: Path, weight_source: str):
    from hydra import compose, initialize_config_dir

    overrides = [
        "data=xmidi_getmusic",
        "model=xmidi_getmusic_small",
        "model.attn_backend=flex",
        "algo=xmidi_getmusic_bd3lm",
        "block_size=16",
        "loader.batch_size=1",
        "loader.global_batch_size=1",
        "loader.eval_batch_size=1",
        "loader.eval_global_batch_size=1",
        "loader.num_workers=0",
        "trainer.devices=1",
        "trainer.num_nodes=1",
        "wandb=null",
        "mode=ppl_eval",
        "loss.use_focal=false",
        "training.from_pretrained=null",
        "training.ema=0.9999" if weight_source == "ema" else "training.ema=0.0",
        "eval.disable_ema=false" if weight_source == "ema" else "eval.disable_ema=true",
        f"eval.checkpoint_path={checkpoint}",
    ]
    with initialize_config_dir(version_base=None, config_dir=str(REPO / "configs")):
        return compose(config_name="config", overrides=overrides)


def derive_corruption_seed(base_seed: int, dataset_index: int, repeat: int) -> int:
    payload = f"v2-focal-calibration:{base_seed}:{dataset_index}:{repeat}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def select_subset(dataset_size: int, sample_count: int, subset_seed: int) -> list[int]:
    if sample_count <= 0 or sample_count > dataset_size:
        raise ValueError(f"sample_count must be in [1, {dataset_size}], got {sample_count}")
    generator = np.random.default_rng(subset_seed)
    return [int(value) for value in generator.choice(dataset_size, sample_count, replace=False)]


def new_stats(gammas: tuple[float, ...]) -> dict[str, Any]:
    return {
        "selected_count": 0,
        "sum_probability": 0.0,
        "sum_log_probability": 0.0,
        "sum_negative_log_probability": 0.0,
        "sum_t": 0.0,
        "sum_inverse_t": 0.0,
        "base_target_gradient_mass": 0.0,
        "base_loss_mass": 0.0,
        "unscaled_focal_target_gradient_mass": {str(gamma): 0.0 for gamma in gammas},
        "unscaled_focal_loss_mass": {str(gamma): 0.0 for gamma in gammas},
        "log_probability_histogram": [0] * LOG_P_HIST_BINS,
        "log_probability_below_min": 0,
    }


def update_stats(
    stats: dict[str, Any],
    log_probability: np.ndarray,
    t: np.ndarray,
    inverse_t: np.ndarray,
    region_size: float,
    gammas: tuple[float, ...],
) -> None:
    if log_probability.size == 0:
        return
    probability = np.exp(log_probability)
    one_minus = -np.expm1(log_probability)
    base_gradient_weight = inverse_t * one_minus / region_size
    base_loss_weight = inverse_t * (-log_probability) / region_size

    stats["selected_count"] += int(log_probability.size)
    stats["sum_probability"] += float(probability.sum(dtype=np.float64))
    stats["sum_log_probability"] += float(log_probability.sum(dtype=np.float64))
    stats["sum_negative_log_probability"] += float((-log_probability).sum(dtype=np.float64))
    stats["sum_t"] += float(t.sum(dtype=np.float64))
    stats["sum_inverse_t"] += float(inverse_t.sum(dtype=np.float64))
    stats["base_target_gradient_mass"] += float(base_gradient_weight.sum(dtype=np.float64))
    stats["base_loss_mass"] += float(base_loss_weight.sum(dtype=np.float64))

    clipped = np.clip(log_probability, LOG_P_MIN, np.nextafter(LOG_P_MAX, LOG_P_MIN))
    indices = np.floor(
        (clipped - LOG_P_MIN) * LOG_P_HIST_BINS / (LOG_P_MAX - LOG_P_MIN)
    ).astype(np.int64)
    indices = np.clip(indices, 0, LOG_P_HIST_BINS - 1)
    histogram = np.bincount(indices, minlength=LOG_P_HIST_BINS)
    stats["log_probability_histogram"] = (
        np.asarray(stats["log_probability_histogram"], dtype=np.int64) + histogram
    ).tolist()
    stats["log_probability_below_min"] += int(np.count_nonzero(log_probability < LOG_P_MIN))

    safe_one_minus = np.maximum(one_minus, np.finfo(np.float64).tiny)
    for gamma in gammas:
        key = str(gamma)
        if gamma == 0.0:
            gradient_multiplier = np.ones_like(probability)
            loss_multiplier = np.ones_like(probability)
        else:
            loss_multiplier = np.power(one_minus, gamma)
            gradient_multiplier = loss_multiplier * (
                1.0
                + gamma
                * probability
                * (-log_probability)
                / safe_one_minus
            )
        stats["unscaled_focal_target_gradient_mass"][key] += float(
            (base_gradient_weight * gradient_multiplier).sum(dtype=np.float64)
        )
        stats["unscaled_focal_loss_mass"][key] += float(
            (base_loss_weight * loss_multiplier).sum(dtype=np.float64)
        )


def merge_stats(destination: dict[str, Any], source: dict[str, Any]) -> None:
    scalar_fields = (
        "selected_count",
        "sum_probability",
        "sum_log_probability",
        "sum_negative_log_probability",
        "sum_t",
        "sum_inverse_t",
        "base_target_gradient_mass",
        "base_loss_mass",
        "log_probability_below_min",
    )
    for field in scalar_fields:
        destination[field] += source[field]
    destination["log_probability_histogram"] = (
        np.asarray(destination["log_probability_histogram"], dtype=np.int64)
        + np.asarray(source["log_probability_histogram"], dtype=np.int64)
    ).tolist()
    for field in (
        "unscaled_focal_target_gradient_mass",
        "unscaled_focal_loss_mass",
    ):
        for key, value in source[field].items():
            destination[field][key] += value


def histogram_quantiles(histogram: list[int]) -> dict[str, float | None]:
    counts = np.asarray(histogram, dtype=np.int64)
    cumulative = np.cumsum(counts)
    total = int(cumulative[-1]) if cumulative.size else 0
    if total == 0:
        return {f"p{int(q * 100):02d}": None for q in QUANTILES}
    width = (LOG_P_MAX - LOG_P_MIN) / LOG_P_HIST_BINS
    result: dict[str, float | None] = {}
    for quantile in QUANTILES:
        target = max(1, math.ceil(quantile * total))
        index = int(np.searchsorted(cumulative, target, side="left"))
        log_probability = LOG_P_MIN + (index + 0.5) * width
        result[f"p{int(quantile * 100):02d}"] = math.exp(log_probability)
    return result


def summarize_stats(stats: dict[str, Any], gammas: tuple[float, ...]) -> dict[str, Any]:
    count = int(stats["selected_count"])
    base_gradient_mass = float(stats["base_target_gradient_mass"])
    base_loss_mass = float(stats["base_loss_mass"])
    if count == 0:
        return {
            "selected_count": 0,
            "mean_probability": None,
            "probability_quantiles_approx": histogram_quantiles(
                stats["log_probability_histogram"]
            ),
            "mean_log_probability": None,
            "mean_negative_log_probability": None,
            "mean_t_at_selected_positions": None,
            "mean_inverse_t_at_selected_positions": None,
            "base_target_gradient_mass": 0.0,
            "base_loss_mass": 0.0,
            "log_probability_below_histogram_min": 0,
            "gradient_retention_by_gamma": {str(gamma): None for gamma in gammas},
            "loss_retention_by_gamma": {str(gamma): None for gamma in gammas},
        }
    return {
        "selected_count": count,
        "mean_probability": float(stats["sum_probability"]) / count,
        "probability_quantiles_approx": histogram_quantiles(
            stats["log_probability_histogram"]
        ),
        "mean_log_probability": float(stats["sum_log_probability"]) / count,
        "mean_negative_log_probability": (
            float(stats["sum_negative_log_probability"]) / count
        ),
        "mean_t_at_selected_positions": float(stats["sum_t"]) / count,
        "mean_inverse_t_at_selected_positions": (
            float(stats["sum_inverse_t"]) / count
        ),
        "base_target_gradient_mass": base_gradient_mass,
        "base_loss_mass": base_loss_mass,
        "log_probability_below_histogram_min": int(
            stats["log_probability_below_min"]
        ),
        "gradient_retention_by_gamma": {
            str(gamma): (
                float(stats["unscaled_focal_target_gradient_mass"][str(gamma)])
                / base_gradient_mass
            )
            for gamma in gammas
        },
        "loss_retention_by_gamma": {
            str(gamma): (
                float(stats["unscaled_focal_loss_mass"][str(gamma)])
                / base_loss_mass
            )
            for gamma in gammas
        },
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def run_worker(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "worker requires exactly one visible CUDA device; set CUDA_VISIBLE_DEVICES"
        )
    if not 0 <= args.rank < args.world_size:
        raise ValueError("rank must be in [0, world_size)")

    checkpoint = Path(args.checkpoint).resolve()
    output = require_project_path(args.output, label="calibration shard output")
    gammas = tuple(float(value) for value in args.gammas)

    register_resolvers()
    config = build_config(checkpoint, args.weight_source)

    import dataloader as bd3_dataloader
    import diffusion as bd3_diffusion
    from xmidi_getmusic import get_xmidi_getmusic_dataset

    tokenizer = bd3_dataloader.get_tokenizer(config)
    checkpoint_metadata = inspect_v2_checkpoint(
        checkpoint,
        expected_hidden_size=int(config.model.hidden_size),
        require_ema=(args.weight_source == "ema"),
        expected_use_focal=False,
        compute_sha256=False,
    )
    model = bd3_diffusion.Diffusion.load_from_checkpoint(
        checkpoint,
        tokenizer=tokenizer,
        config=config,
        strict=True,
        weights_only=False,
        map_location="cpu",
    )
    materialize_inference_weights(model, args.weight_source)
    model = model.to("cuda").eval()

    dataset = get_xmidi_getmusic_dataset(config, tokenizer, split="train")
    subset = select_subset(len(dataset), args.sample_count, args.subset_seed)
    local_indices = subset[args.rank :: args.world_size]

    groups = {"pad": new_stats(gammas), "event": new_stats(gammas)}
    rows = {str(row): new_stats(gammas) for row in range(4, NUM_ROWS)}
    time_bins = {name: {"pad": new_stats(gammas), "event": new_stats(gammas)} for name in T_BIN_NAMES}
    region_positions = 0
    selected_positions = 0
    empty_draws = 0

    progress = tqdm(local_indices, desc=f"calibration-rank-{args.rank}", dynamic_ncols=True)
    with torch.inference_mode():
        for dataset_index in progress:
            sample = dataset[dataset_index]
            input_ids = sample["input_ids"].unsqueeze(0).to("cuda", non_blocking=True)
            region_mask = sample["attention_mask"].unsqueeze(0).to(
                "cuda", non_blocking=True
            ).bool()
            row_unk_mask = sample["row_unk_mask"].unsqueeze(0).to(
                "cuda", non_blocking=True
            ).bool()
            region_size = float(region_mask.sum().item())
            region_positions += int(region_size) * args.corruptions

            for repeat in range(args.corruptions):
                seed = derive_corruption_seed(args.base_seed, dataset_index, repeat)
                random.seed(seed)
                np.random.seed(seed % (2**32))
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

                t = model._sample_t(
                    input_ids.shape,
                    input_ids.device,
                    args.sampling_eps_min,
                    args.sampling_eps_max,
                )
                loss_scale, move_probability = model.noise(t)
                sigma = model._sigma_from_p(move_probability[:, 0].unsqueeze(-1))
                xt = model.q_xt(
                    input_ids,
                    move_probability,
                    sampling_eps_min=args.sampling_eps_min,
                    sampling_eps_max=args.sampling_eps_max,
                    region_mask=region_mask,
                )
                selected = xt.eq(model.mask_index) & region_mask
                count = int(selected.sum().item())
                if count == 0:
                    empty_draws += 1
                    continue
                selected_positions += count
                model.backbone.set_xmidi_row_unk_mask(row_unk_mask)
                try:
                    model_input = torch.cat((xt, input_ids), dim=-1)
                    log_probability = model._selected_subs_log_probs(
                        model_input, sigma, input_ids, selected
                    )
                finally:
                    model.backbone.set_xmidi_row_unk_mask(None)

                selected_log_probability = (
                    log_probability[selected].detach().to(torch.float64).cpu().numpy()
                )
                selected_targets = input_ids[selected].detach().cpu().numpy()
                selected_t = t[selected].detach().to(torch.float64).cpu().numpy()
                selected_inverse_t = (
                    (-loss_scale.expand_as(t)[selected])
                    .detach()
                    .to(torch.float64)
                    .cpu()
                    .numpy()
                )
                selected_positions_flat = selected.nonzero(as_tuple=False)[:, 1]
                selected_rows = (selected_positions_flat // NUM_COLS).cpu().numpy()

                pad_mask = selected_targets == PAD_ID
                for group_name, group_mask in (
                    ("pad", pad_mask),
                    ("event", ~pad_mask),
                ):
                    update_stats(
                        groups[group_name],
                        selected_log_probability[group_mask],
                        selected_t[group_mask],
                        selected_inverse_t[group_mask],
                        region_size,
                        gammas,
                    )

                for row in range(4, NUM_ROWS):
                    row_mask = selected_rows == row
                    update_stats(
                        rows[str(row)],
                        selected_log_probability[row_mask],
                        selected_t[row_mask],
                        selected_inverse_t[row_mask],
                        region_size,
                        gammas,
                    )

                for bin_index, bin_name in enumerate(T_BIN_NAMES):
                    lower, upper = T_BIN_EDGES[bin_index : bin_index + 2]
                    in_bin = (selected_t >= lower) & (selected_t < upper)
                    for group_name, group_mask in (
                        ("pad", pad_mask),
                        ("event", ~pad_mask),
                    ):
                        mask = in_bin & group_mask
                        update_stats(
                            time_bins[bin_name][group_name],
                            selected_log_probability[mask],
                            selected_t[mask],
                            selected_inverse_t[mask],
                            region_size,
                            gammas,
                        )

            progress.set_postfix(selected=selected_positions)

    payload = {
        "schema_version": "bd3lms-music-v2-focal.calibration-shard.v1",
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "rank": args.rank,
        "world_size": args.world_size,
        "checkpoint": checkpoint_metadata,
        "protocol": {
            "weight_source": args.weight_source,
            "dataset_split": "train",
            "dataset_size": len(dataset),
            "sample_count_total": args.sample_count,
            "sample_count_rank": len(local_indices),
            "subset_seed": args.subset_seed,
            "base_seed": args.base_seed,
            "corruptions_per_sample": args.corruptions,
            "sampling_eps_min": args.sampling_eps_min,
            "sampling_eps_max": args.sampling_eps_max,
            "batch_size_per_gpu": 1,
            "model_eval_mode": True,
            "selected_definition": "region_mask & (x_t == <MASK>)",
            "gradient_weight": "(1-p_y)/t/N_G",
            "gammas": list(gammas),
            "log_probability_histogram": {
                "min": LOG_P_MIN,
                "max": LOG_P_MAX,
                "bins": LOG_P_HIST_BINS,
            },
        },
        "subset_indices": local_indices,
        "subset_indices_sha256": canonical_sha256({"indices": local_indices}),
        "region_positions_across_draws": region_positions,
        "selected_positions": selected_positions,
        "empty_draws": empty_draws,
        "groups_raw": groups,
        "rows_raw": rows,
        "time_bins_raw": time_bins,
    }
    payload["artifact_sha256"] = canonical_sha256(payload)
    atomic_write_json(output, payload)
    print(json.dumps({"output": str(output), "selected_positions": selected_positions}, indent=2))


def solve_candidate(
    pad_stats: dict[str, Any],
    event_stats: dict[str, Any],
    gamma: float,
    target_event_over_pad_multiplier: float,
) -> dict[str, float]:
    key = str(gamma)
    pad_gradient = float(pad_stats["base_target_gradient_mass"])
    event_gradient = float(event_stats["base_target_gradient_mass"])
    h_pad = (
        float(pad_stats["unscaled_focal_target_gradient_mass"][key]) / pad_gradient
    )
    h_event = (
        float(event_stats["unscaled_focal_target_gradient_mass"][key])
        / event_gradient
    )
    alpha_ratio = target_event_over_pad_multiplier * h_pad / h_event
    scale = (pad_gradient + event_gradient) / (
        h_pad * pad_gradient + alpha_ratio * h_event * event_gradient
    )
    alpha_pad = scale
    alpha_event = scale * alpha_ratio
    objective_pad = alpha_pad * h_pad * pad_gradient
    objective_event = alpha_event * h_event * event_gradient
    return {
        "gamma": gamma,
        "target_event_over_pad_gradient_multiplier": target_event_over_pad_multiplier,
        "h_pad": h_pad,
        "h_event": h_event,
        "unscaled_h_event_over_h_pad": h_event / h_pad,
        "alpha_pad": alpha_pad,
        "alpha_event": alpha_event,
        "alpha_event_over_alpha_pad": alpha_ratio,
        "predicted_event_over_pad_gradient_multiplier": (
            alpha_event * h_event / (alpha_pad * h_pad)
        ),
        "base_pad_gradient_share": pad_gradient / (pad_gradient + event_gradient),
        "predicted_objective_pad_gradient_share": objective_pad
        / (objective_pad + objective_event),
        "predicted_total_gradient_scale_vs_v2": (objective_pad + objective_event)
        / (pad_gradient + event_gradient),
    }


def run_merge(args: argparse.Namespace) -> None:
    output_dir = require_project_path(args.output_dir, label="calibration report directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_paths = [Path(path).resolve() for path in args.shards]
    if not shard_paths:
        raise ValueError("at least one --shards path is required")
    shards = [json.loads(path.read_text(encoding="utf-8")) for path in shard_paths]
    world_sizes = {int(shard["world_size"]) for shard in shards}
    ranks = {int(shard["rank"]) for shard in shards}
    if world_sizes != {len(shards)} or ranks != set(range(len(shards))):
        raise ValueError(f"incomplete shard set: world_sizes={world_sizes}, ranks={ranks}")
    if any(shard.get("status") != "completed" for shard in shards):
        raise ValueError("all calibration shards must be completed")

    protocol = shards[0]["protocol"]
    gammas = tuple(float(value) for value in protocol["gammas"])
    for shard in shards[1:]:
        if shard["protocol"] != protocol:
            raise ValueError("calibration shard protocol mismatch")

    groups = {"pad": new_stats(gammas), "event": new_stats(gammas)}
    rows = {str(row): new_stats(gammas) for row in range(4, NUM_ROWS)}
    time_bins = {name: {"pad": new_stats(gammas), "event": new_stats(gammas)} for name in T_BIN_NAMES}
    subset_indices: list[int] = []
    for shard in shards:
        subset_indices.extend(int(value) for value in shard["subset_indices"])
        for group in groups:
            merge_stats(groups[group], shard["groups_raw"][group])
        for row in rows:
            merge_stats(rows[row], shard["rows_raw"][row])
        for bin_name in time_bins:
            for group in groups:
                merge_stats(
                    time_bins[bin_name][group],
                    shard["time_bins_raw"][bin_name][group],
                )

    expected_subset = select_subset(
        int(protocol["dataset_size"]),
        int(protocol["sample_count_total"]),
        int(protocol["subset_seed"]),
    )
    if sorted(subset_indices) != sorted(expected_subset):
        raise ValueError("merged shard indices do not match deterministic subset")

    checkpoint = Path(args.checkpoint).resolve()
    checkpoint_metadata = inspect_v2_checkpoint(
        checkpoint,
        expected_hidden_size=768,
        require_ema=(protocol["weight_source"] == "ema"),
        expected_use_focal=False,
        compute_sha256=True,
    )
    data_root = Path(args.data_root).resolve()
    data_metadata = dataset_fingerprint(
        data_root / "train", "train", data_root / "whole_dict.txt"
    )

    old_event_odds = (1.0 - args.old_generated_pad_ratio) / args.old_generated_pad_ratio
    reference_event_odds = (1.0 - args.reference_pad_ratio) / args.reference_pad_ratio
    occupancy_odds_correction = reference_event_odds / old_event_odds
    biases = sorted(set(float(value) for value in args.biases) | {occupancy_odds_correction})
    candidates = [
        solve_candidate(groups["pad"], groups["event"], gamma, bias)
        for gamma in gammas
        for bias in biases
    ]
    recommended = min(
        candidates,
        key=lambda item: (
            abs(item["gamma"] - args.recommended_gamma),
            abs(
                item["target_event_over_pad_gradient_multiplier"]
                - args.recommended_bias
            ),
        ),
    )
    occupancy_odds_matched = min(
        candidates,
        key=lambda item: (
            abs(item["gamma"] - args.recommended_gamma),
            abs(
                item["target_event_over_pad_gradient_multiplier"]
                - occupancy_odds_correction
            ),
        ),
    )
    current_gamma = 2.0
    current_alpha_pad = 0.8600010833864916
    current_alpha_event = 1.2900016250797375
    current_key = str(current_gamma)
    pad_gradient = float(groups["pad"]["base_target_gradient_mass"])
    event_gradient = float(groups["event"]["base_target_gradient_mass"])
    current_h_pad = (
        float(groups["pad"]["unscaled_focal_target_gradient_mass"][current_key])
        / pad_gradient
    )
    current_h_event = (
        float(groups["event"]["unscaled_focal_target_gradient_mass"][current_key])
        / event_gradient
    )
    current_pad_mass = current_alpha_pad * current_h_pad * pad_gradient
    current_event_mass = current_alpha_event * current_h_event * event_gradient
    current_configuration_diagnostic = {
        "gamma": current_gamma,
        "alpha_pad": current_alpha_pad,
        "alpha_event": current_alpha_event,
        "alpha_event_over_alpha_pad": current_alpha_event / current_alpha_pad,
        "h_pad": current_h_pad,
        "h_event": current_h_event,
        "unscaled_h_event_over_h_pad": current_h_event / current_h_pad,
        "effective_event_over_pad_gradient_multiplier": (
            current_alpha_event * current_h_event
            / (current_alpha_pad * current_h_pad)
        ),
        "predicted_total_gradient_scale_vs_v2": (
            current_pad_mass + current_event_mass
        )
        / (pad_gradient + event_gradient),
        "predicted_objective_pad_gradient_share": current_pad_mass
        / (current_pad_mass + current_event_mass),
    }

    summarized_groups = {
        group: summarize_stats(stats, gammas) for group, stats in groups.items()
    }
    summarized_rows = {
        row: {"row_name": ROW_NAMES[int(row)], **summarize_stats(stats, gammas)}
        for row, stats in rows.items()
    }
    summarized_time_bins = {
        bin_name: {
            group: summarize_stats(stats, gammas)
            for group, stats in group_stats.items()
        }
        for bin_name, group_stats in time_bins.items()
    }

    report = {
        "schema_version": "bd3lms-music-v2-focal.calibration-report.v1",
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": checkpoint_metadata,
        "dataset": data_metadata,
        "source": fingerprint_file(Path(__file__)),
        "protocol": protocol,
        "subset": {
            "indices": expected_subset,
            "indices_sha256": canonical_sha256({"indices": expected_subset}),
        },
        "coverage": {
            "region_positions_across_draws": sum(
                int(shard["region_positions_across_draws"]) for shard in shards
            ),
            "selected_positions": sum(int(shard["selected_positions"]) for shard in shards),
            "empty_draws": sum(int(shard["empty_draws"]) for shard in shards),
        },
        "groups": summarized_groups,
        "rows": summarized_rows,
        "time_bins": summarized_time_bins,
        "calibration": {
            "method": {
                "base_target_gradient_weight": "(1-p_y)/t/N_G",
                "focal_gradient_retention": (
                    "h_gamma(p)=(1-p)^(gamma-1)*((1-p)+gamma*p*(-log(p)))"
                ),
                "alpha_ratio": "B*H_pad/H_event",
                "normalization": "preserve total expected V2 target-gradient mass",
            },
            "old_generated_pad_ratio": args.old_generated_pad_ratio,
            "reference_pad_ratio": args.reference_pad_ratio,
            "event_odds_correction_reference_over_old": occupancy_odds_correction,
            "current_fixed_configuration_diagnostic": current_configuration_diagnostic,
            "occupancy_odds_matched_candidate": occupancy_odds_matched,
            "candidate_grid": candidates,
            "recommended_candidate": {
                **recommended,
                "selection_rule": (
                    "gamma=0.25 with conservative B=1.10; alpha solved from V2 EMA "
                    "selected target-gradient retention"
                ),
                "interpretation": "best calibrated starting point; requires fixed pilot validation",
            },
        },
        "checks": {
            "all_shards_completed": True,
            "complete_deterministic_subset": True,
            "checkpoint_is_legacy_v2": checkpoint_metadata["focal_loss_config"] is None
            or not checkpoint_metadata["focal_loss_config"].get("use_focal", False),
            "ema_used": protocol["weight_source"] == "ema",
            "no_empty_draws": sum(int(shard["empty_draws"]) for shard in shards) == 0,
            "finite_recommended_parameters": all(
                math.isfinite(recommended[key])
                for key in ("gamma", "alpha_pad", "alpha_event")
            ),
            "total_gradient_scale_preserved": math.isclose(
                recommended["predicted_total_gradient_scale_vs_v2"],
                1.0,
                rel_tol=1e-10,
                abs_tol=1e-10,
            ),
        },
    }
    report["artifact_sha256"] = canonical_sha256(report)
    report_path = output_dir / "v2_target_probability_summary.json"
    atomic_write_json(report_path, report)

    csv_path = output_dir / "focal_parameter_candidates.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidates[0]))
        writer.writeheader()
        writer.writerows(candidates)

    compact_path = output_dir / "recommended_focal_parameters.json"
    compact = {
        "schema_version": "bd3lms-music-v2-focal.recommended-parameters.v1",
        "source_report": str(report_path),
        "source_report_sha256": sha256_file(report_path),
        "checkpoint_sha256": checkpoint_metadata["sha256"],
        "recommended": report["calibration"]["recommended_candidate"],
        "requires_fixed_pilot_validation": True,
    }
    compact["artifact_sha256"] = canonical_sha256(compact)
    atomic_write_json(compact_path, compact)

    print(
        json.dumps(
            {
                "report": str(report_path),
                "candidates": str(csv_path),
                "recommended_parameters": str(compact_path),
                "selected_positions": report["coverage"]["selected_positions"],
                "recommended": report["calibration"]["recommended_candidate"],
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    worker = subparsers.add_parser("worker")
    worker.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    worker.add_argument("--output", required=True)
    worker.add_argument("--rank", type=int, required=True)
    worker.add_argument("--world_size", type=int, default=8)
    worker.add_argument("--sample_count", type=int, default=2048)
    worker.add_argument("--corruptions", type=int, default=4)
    worker.add_argument("--subset_seed", type=int, default=20260802)
    worker.add_argument("--base_seed", type=int, default=1)
    worker.add_argument("--weight_source", choices=("ema", "raw"), default="ema")
    worker.add_argument("--sampling_eps_min", type=float, default=1e-3)
    worker.add_argument("--sampling_eps_max", type=float, default=1.0)
    worker.add_argument("--gammas", type=float, nargs="+", default=DEFAULT_GAMMAS)

    merge = subparsers.add_parser("merge")
    merge.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    merge.add_argument("--data_root", default=str(REPO / "XMIDI_Dataset_output_v2"))
    merge.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    merge.add_argument("--shards", nargs="+", required=True)
    merge.add_argument("--biases", type=float, nargs="+", default=DEFAULT_BIASES)
    merge.add_argument("--recommended_gamma", type=float, default=0.25)
    merge.add_argument("--recommended_bias", type=float, default=1.10)
    merge.add_argument("--old_generated_pad_ratio", type=float, default=0.6331759345297748)
    merge.add_argument("--reference_pad_ratio", type=float, default=0.603355)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "worker":
        run_worker(args)
    else:
        run_merge(args)


if __name__ == "__main__":
    main()
