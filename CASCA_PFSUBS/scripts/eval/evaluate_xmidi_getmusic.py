"""Fine-grained XMIDI-GetMusic evaluation.

Mirrors :mod:`scripts.eval.evaluate_xmidi`, but adapted for the new
[16, 512] grid + 142421-class V2 vocabulary. Reports:

  1. Aggregate NLL / PPL / BPD over the validation set.
  2. Per-target-row NLL: which of V2 rows 4..15 is hardest to predict.
  3. Per-content-block NLL (block_size=16): NLL for each of the 31
     content blocks within a target row (32 blocks/row, the first one
     is the "prompts" prefix).
  4. Per-UNK-density NLL: bucket samples by ``int(row_unk_mask.sum())``
     (number of fully-<UNK> rows) and report aggregate NLL per bucket.
  5. Loss-time-step variance: K random t draws per batch.

Usage
-----
::

    python scripts/eval/evaluate_xmidi_getmusic.py \\
        --ckpt outputs/bd3lms-music-v2-xmidi_getmusic_small-bs16/checkpoints/last.ckpt \\
        --model xmidi_getmusic_small \\
        --block_size 16 \\
        --bsz 2 \\
        --num_t_samples 4 \\
        --max_batches 0 \\
        --out_csv eval_xmidi_getmusic_summary.csv
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import omegaconf
import torch
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from xmidi_getmusic.constants import (
    NUM_COLS,
    NUM_PROMPT_COLS,
    NUM_ROWS,
    NUM_SOURCE_ROWS,
    UNK_ID,
)
from xmidi_getmusic.runtime_requirements import (
    require_conda_environment,
    require_project_path,
    require_tmux_identity,
    require_wandb_project,
)

os.environ.setdefault("HF_HOME", str(REPO / "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("WANDB_PROJECT", "CASCA_focal")


def register_resolvers():
    for name, fn in [
        ("cwd", os.getcwd),
        ("device_count", torch.cuda.device_count),
        ("eval", eval),
        ("div_up", lambda x, y: (x + y - 1) // y),
    ]:
        try:
            omegaconf.OmegaConf.register_new_resolver(name, fn)
        except ValueError:
            pass


def build_eval_config(args):
    from hydra import compose, initialize_config_dir
    cfg_dir = str(REPO / "configs")
    overrides = [
        "data=xmidi_getmusic",
        f"model={args.model}",
        "model.attn_backend=flex",
        f"block_size={args.block_size}",
        f"loader.eval_batch_size={args.bsz}",
        f"loader.eval_global_batch_size={args.bsz}",
        f"loader.batch_size={args.bsz}",
        "trainer.devices=1",
        "trainer.num_nodes=1",
        "wandb=null",
        "algo=xmidi_getmusic_bd3lm",
        f"algo.T={args.t}",
        "training.from_pretrained=null",
        "training.ema=0.9999" if args.weight_source == "ema" else "training.ema=0.0",
        "mode=ppl_eval",
        f"eval.checkpoint_path={args.ckpt}",
        "eval.disable_ema=False" if args.weight_source == "ema" else "eval.disable_ema=True",
    ]
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        cfg = compose(config_name="config", overrides=overrides)
    return cfg


# ---------------------------------------------------------------------------
def evaluate(args):
    from xmidi_getmusic import (
        NUM_ROWS, NUM_COLS,
        NUM_SOURCE_ROWS, NUM_PROMPT_COLS,
        UNK_ID,
    )
    from xmidi_getmusic.eval_protocol import (
        dataset_fingerprint,
        inspect_v2_checkpoint,
        materialize_inference_weights,
        seed_everything,
        sha256_file,
    )

    register_resolvers()
    cfg = build_eval_config(args)
    seed_everything(args.seed)
    checkpoint_meta = inspect_v2_checkpoint(
        args.ckpt,
        expected_hidden_size=int(cfg.model.hidden_size),
        require_ema=(args.weight_source == "ema"),
        compute_sha256=True,
    )
    data_meta = dataset_fingerprint(
        Path(cfg.data.cache_dir) / "valid", "valid", cfg.data.dict_path)

    import dataloader as bd3_dataloader
    import diffusion as bd3_diffusion

    print(f"[evaluate_xmidi_getmusic] ckpt = {args.ckpt}")
    tokenizer = bd3_dataloader.get_tokenizer(cfg)

    model = bd3_diffusion.Diffusion.load_from_checkpoint(
        args.ckpt,
        tokenizer=tokenizer,
        config=cfg,
        strict=True,
        weights_only=False,
        map_location="cpu",
    )
    materialize_inference_weights(model, args.weight_source)
    model = model.to("cuda").eval()

    _, valid_loader = bd3_dataloader.get_dataloaders(
        cfg, tokenizer, skip_train=True, valid_seed=None)

    blocks_per_row = NUM_COLS // cfg.block_size       # 32

    agg_num = 0.0
    agg_den = 0.0

    per_row_num = torch.zeros(NUM_ROWS, dtype=torch.float64)
    per_row_den = torch.zeros(NUM_ROWS, dtype=torch.float64)

    per_block_num = torch.zeros(blocks_per_row, dtype=torch.float64)
    per_block_den = torch.zeros(blocks_per_row, dtype=torch.float64)

    per_unkdensity_num: dict[int, float] = defaultdict(float)
    per_unkdensity_den: dict[int, float] = defaultdict(float)

    K = max(1, args.num_t_samples)
    var_acc = []

    n_batches = 0
    n_samples = 0
    pbar = tqdm(valid_loader, desc="eval", dynamic_ncols=True)
    with torch.no_grad():
        for batch in pbar:
            ids = batch["input_ids"].to("cuda")
            attn_mask = batch["attention_mask"].to("cuda")
            row_unk_mask = batch.get("row_unk_mask", None)
            if row_unk_mask is not None:
                row_unk_mask = row_unk_mask.to("cuda").bool()
            B = ids.shape[0]

            losses_per_pass = []
            for _ in range(K):
                losses = model._loss(
                    ids,
                    attn_mask.clone(),
                    sampling_eps_min=1e-3,
                    sampling_eps_max=1.0,
                    row_unk_mask=row_unk_mask,
                )
                losses_per_pass.append(losses.nlls.detach().to(torch.float64).cpu())

            nlls_mean = torch.stack(losses_per_pass, dim=0).mean(dim=0)
            mask_cpu = attn_mask.detach().to(torch.float64).cpu()

            per_pass_means = torch.stack(
                [(p * mask_cpu).sum() / max(mask_cpu.sum().item(), 1.0)
                 for p in losses_per_pass]
            )
            var_acc.append((
                per_pass_means.mean().item(),
                per_pass_means.std(unbiased=False).item(),
            ))

            agg_num += float((nlls_mean * mask_cpu).sum().item())
            agg_den += float(mask_cpu.sum().item())

            nlls_grid = nlls_mean.view(B, NUM_ROWS, NUM_COLS)
            mask_grid = mask_cpu.view(B, NUM_ROWS, NUM_COLS)
            per_row_num += (nlls_grid * mask_grid).sum(dim=(0, 2))
            per_row_den += mask_grid.sum(dim=(0, 2))

            nlls_blocks = nlls_grid.view(
                B, NUM_ROWS, blocks_per_row, cfg.block_size)
            mask_blocks = mask_grid.view(
                B, NUM_ROWS, blocks_per_row, cfg.block_size)
            per_block_num += (nlls_blocks * mask_blocks).sum(dim=(0, 1, 3))
            per_block_den += mask_blocks.sum(dim=(0, 1, 3))

            if row_unk_mask is not None:
                unk_density = row_unk_mask[:, NUM_SOURCE_ROWS:].sum(dim=1).cpu()
                for b in range(B):
                    k = int(unk_density[b].item())
                    n = float((nlls_mean[b] * mask_cpu[b]).sum().item())
                    d = float(mask_cpu[b].sum().item())
                    per_unkdensity_num[k] += n
                    per_unkdensity_den[k] += d

            n_batches += 1
            n_samples += B
            pbar.set_postfix({
                "agg_nll": f"{agg_num / max(agg_den, 1.0):.3f}",
                "agg_ppl": f"{math.exp(agg_num / max(agg_den, 1.0)):.1f}",
                "B": n_samples,
            })
            if args.max_batches and n_batches >= args.max_batches:
                break

    is_full_validation = not args.max_batches
    if is_full_validation and n_samples != args.expect_valid_size:
        raise RuntimeError(
            f"full validation requires {args.expect_valid_size} unique samples; "
            f"DataLoader yielded {n_samples}"
        )

    summary = {
        "metric_definition": (
            "Monte-Carlo estimate of the region-aware subs-NELBO upper bound; "
            "PPL/BPD are deterministic transforms of this bound"
        ),
        "checkpoint": checkpoint_meta,
        "dataset": data_meta,
        "evaluator_sha256": sha256_file(Path(__file__).resolve()),
        "weight_source": args.weight_source,
        "seed": args.seed,
    }
    summary["num_batches"] = n_batches
    summary["num_samples"] = n_samples
    summary["is_full_validation_set"] = is_full_validation
    summary["expected_valid_size"] = args.expect_valid_size
    summary["block_size"] = cfg.block_size
    summary["k_t_samples"] = K

    nll_agg = agg_num / max(agg_den, 1.0)
    summary["aggregate"] = {
        "nll": nll_agg,
        "ppl": math.exp(nll_agg),
        "bpd": nll_agg / math.log(2.0),
        "num_loss_positions": int(agg_den),
    }

    summary["per_row_nll"] = {}
    for r in range(NUM_ROWS):
        d = float(per_row_den[r].item())
        summary["per_row_nll"][r] = {
            "nll": (per_row_num[r].item() / d) if d > 0 else None,
            "n_positions": int(d),
            "kind": "source" if r < NUM_SOURCE_ROWS else "target",
        }

    summary["per_content_block_nll"] = {}
    for j in range(blocks_per_row):
        d = float(per_block_den[j].item())
        is_prompt = j * cfg.block_size < NUM_PROMPT_COLS
        col_lo = j * cfg.block_size
        col_hi = col_lo + cfg.block_size - 1
        summary["per_content_block_nll"][j] = {
            "col_range": [col_lo, col_hi],
            "is_prompt_block": bool(is_prompt),
            "nll": (per_block_num[j].item() / d) if d > 0 else None,
            "n_positions": int(d),
        }

    summary["per_unk_density_nll"] = {}
    for k in sorted(per_unkdensity_num):
        d = per_unkdensity_den[k]
        summary["per_unk_density_nll"][int(k)] = {
            "nll": (per_unkdensity_num[k] / d) if d > 0 else None,
            "n_positions": int(d),
        }

    if var_acc:
        means, stds = zip(*var_acc)
        summary["t_sample_variance"] = {
            "k_t_samples_per_batch": K,
            "mean_of_per_batch_means": sum(means) / len(means),
            "mean_of_per_batch_stds":  sum(stds) / len(stds),
            "max_per_batch_std":       max(stds),
        }

    return summary


# ---------------------------------------------------------------------------
def print_summary(s):
    print()
    print("=" * 78)
    print(f"XMIDI-GetMusic eval summary  (n_samples={s['num_samples']}, "
          f"n_batches={s['num_batches']}, K={s['k_t_samples']})")
    print("=" * 78)

    a = s["aggregate"]
    print(f"\n[aggregate over loss region]")
    print(f"  NLL = {a['nll']:.4f} nats/token")
    print(f"  PPL = {a['ppl']:.4f}")
    print(f"  BPD = {a['bpd']:.4f} bits/token")
    print(f"  loss positions = {a['num_loss_positions']}")

    print(f"\n[per row, content-only]")
    print(f"  {'row':>4}  {'kind':<8}  {'NLL':>8}  {'PPL':>10}  {'#pos':>10}")
    for r, v in s["per_row_nll"].items():
        nll = v["nll"]
        if nll is None or v["n_positions"] == 0:
            continue
        print(f"  {r:>4}  {v['kind']:<8}  {nll:>8.4f}  "
              f"{math.exp(nll):>10.2f}  {v['n_positions']:>10d}")

    print(f"\n[per content block (block index along the row)]")
    print(f"  {'blk':>4}  {'col-range':>12}  {'is_prompt':>10}  "
          f"{'NLL':>8}  {'PPL':>10}  {'#pos':>10}")
    for j, v in s["per_content_block_nll"].items():
        if v["nll"] is None or v["n_positions"] == 0:
            continue
        cr = f"{v['col_range'][0]}-{v['col_range'][1]}"
        print(f"  {j:>4}  {cr:>12}  {str(v['is_prompt_block']):>10}  "
              f"{v['nll']:>8.4f}  {math.exp(v['nll']):>10.2f}  "
              f"{v['n_positions']:>10d}")

    print(f"\n[per UNK-row count bucket]")
    print(f"  {'k':>4}  {'NLL':>8}  {'PPL':>10}  {'#pos':>10}")
    for k, v in s["per_unk_density_nll"].items():
        if v["nll"] is None:
            continue
        print(f"  {k:>4}  {v['nll']:>8.4f}  {math.exp(v['nll']):>10.2f}  "
              f"{v['n_positions']:>10d}")

    if "t_sample_variance" in s:
        v = s["t_sample_variance"]
        print(f"\n[noise-step variance]")
        print(f"  K={v['k_t_samples_per_batch']} t draws/batch")
        print(f"  mean of per-batch NLL means = {v['mean_of_per_batch_means']:.4f}")
        print(f"  mean of per-batch NLL stds  = {v['mean_of_per_batch_stds']:.4f}")
        print(f"  max  per-batch NLL std      = {v['max_per_batch_std']:.4f}")

    print("=" * 78)


def write_csv(path, s):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["section", "key", "metric", "value"])
        a = s["aggregate"]
        for m, v in a.items():
            w.writerow(["aggregate", "", m, v])
        for r, v in s["per_row_nll"].items():
            for m, val in v.items():
                w.writerow(["per_row_nll", r, m, val])
        for j, v in s["per_content_block_nll"].items():
            for m, val in v.items():
                if isinstance(val, list):
                    val = f"{val[0]}-{val[1]}"
                w.writerow(["per_content_block_nll", j, m, val])
        for k, v in s["per_unk_density_nll"].items():
            for m, val in v.items():
                w.writerow(["per_unk_density_nll", k, m, val])
        if "t_sample_variance" in s:
            for m, val in s["t_sample_variance"].items():
                w.writerow(["t_sample_variance", "", m, val])


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--model", default="xmidi_getmusic_small")
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument("--bsz", type=int, default=2)
    parser.add_argument("--t", type=int, default=0,
                        help="algo.T (only matters for sampling; ignored here)")
    parser.add_argument("--num_t_samples", type=int, default=4,
                        help="how many random t draws to MC-average per batch")
    parser.add_argument("--max_batches", type=int, default=0,
                        help="0 = full valid set; otherwise explicit subset")
    parser.add_argument("--expect_valid_size", type=int, default=6817)
    parser.add_argument("--out_csv", default=None,
                        help="optional CSV summary path")
    parser.add_argument("--out_json", default=None,
                        help="optional JSON summary path")
    parser.add_argument("--weight_source", choices=("ema", "raw"), default="ema")
    parser.add_argument("--use_focal", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="require matching Focal checkpoint metadata")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed for Monte-Carlo diffusion-time draws")
    parser.add_argument("--no_ema", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.no_ema:
        if "--weight_source" in sys.argv and args.weight_source != "raw":
            parser.error("--no_ema conflicts with --weight_source=ema")
        args.weight_source = "raw"

    args.ckpt = str(Path(args.ckpt).resolve())
    require_conda_environment()
    require_tmux_identity()
    require_wandb_project()
    if args.out_csv:
        require_project_path(args.out_csv, label="evaluation CSV output")
    if args.out_json:
        require_project_path(args.out_json, label="evaluation JSON output")

    s = evaluate(args)
    print_summary(s)

    if args.out_csv:
        write_csv(args.out_csv, s)
        print(f"\n[evaluate_xmidi_getmusic] wrote CSV  -> {args.out_csv}")
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(s, f, indent=2)
        print(f"[evaluate_xmidi_getmusic] wrote JSON -> {args.out_json}")


if __name__ == "__main__":
    main()
