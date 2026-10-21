"""Batch source-conditioned inference on the **full** XMIDI-GetMusic valid set.

For every sample in ``valid.data`` we run the BD3-LM block-diffusion
semi-AR sampler (R&D_Requirements_for_getmusic.md §推理) and write a
single JSONL where each line carries one sample dict::

    {
      "valid_index":   <int>,                 # row index inside valid set
      "ground_truth":  [[...512 token str...], ...16 rows...],
      "gen":           [[...512 token str...], ...16 rows...],
      "row_unk_mask":  [false, ..., true, ...],     # 16 bools
      "generation_mask_rle": [[start, length], ...],
      "sample_seed": int,
      "match_stats": {
        "per_target_row":     { "4": {...}, ..., "15": {...} },
        "all_target_content": { "match": 0.32, "exact_matches": 1234,
                                "n_positions": 3856 }
      },
      "diffusion_nfes": <int>,
      "backbone_forward_calls": <int>         # includes fixed-block commits
    }

Adaptive-mask compliance (R&D §"局部 attention 算法"):
  - Source rows (0..3) are copied verbatim from valid GT.
  - Prompts (cols 0..15 of every target row) are copied verbatim.
  - <UNK> positions are fixed at <UNK> for the whole diffusion chain
    (see ``_xmidi_sample_prior`` + the per-block UNK overwrite at the
    end of each content block).
  - Fully-<UNK> rows additionally have their tokens DROPPED from
    attention (both as queries and as keys) via the ``row_unk_mask``
    that is registered on ``DIT`` for the duration of each batch
    (see ``DIT._build_xmidi_sampling_attn_mask`` for the sampling-time
    construction; ``DIT._maybe_get_xmidi_mask`` for the matching
    train-time construction). This brings inference behaviour
    byte-for-byte in line with training.

This script does NOT touch training code paths: it imports the same
``Diffusion`` / ``DIT`` modules as training, but builds an eval-only
config via Hydra ``compose`` and runs everything in a single Python
process (no Lightning trainer, no wandb, no DDP).

Usage
-----
::

    python scripts/eval/batch_infer_xmidi_getmusic.py \\
        --ckpt   outputs/bd3lms-music-v2-xmidi_getmusic_small-bs16/checkpoints/best.ckpt \\
        --model  xmidi_getmusic_small \\
        --block_size 16 \\
        --bsz    1 \\
        --T      1000 \\
        --out    sample_logs/xmidi_getmusic_valid_full.jsonl
        # optional:
        # --max_samples 16        # limit to first N (debug)
        # --start_idx   100       # resume from here
        # --seed        1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import omegaconf
import torch
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from xmidi_getmusic.runtime_requirements import (
    require_conda_environment,
    require_project_path,
    require_tmux_identity,
    require_wandb_project,
)

# Symbolic-music inference is offline and does not require a language model.
os.environ.setdefault("HF_HOME", str(REPO / "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
# This entry point records the fixed project identity in its manifest. It does
# not instantiate a training logger.
os.environ.setdefault("WANDB_PROJECT", "CASCA_focal")


# ---------------------------------------------------------------------------
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
    """Construct an inference Hydra config (mode=sample_eval) without
    actually running ``main.py``.

    The config matches what ``scripts/sample/sample_xmidi_getmusic_bd3lm.sh``
    passes, so the loaded ckpt sees identical model / algo / data settings.
    """
    from hydra import compose, initialize_config_dir
    cfg_dir = str(REPO / "configs")
    overrides = [
        "data=xmidi_getmusic",
        f"model={args.model}",
        "model.attn_backend=sdpa",
        f"block_size={args.block_size}",
        f"loader.batch_size={args.bsz}",
        f"loader.eval_batch_size={args.bsz}",
        f"loader.global_batch_size={args.bsz}",
        f"loader.eval_global_batch_size={args.bsz}",
        "trainer.devices=1",
        "trainer.num_nodes=1",
        "wandb=null",
        "algo=xmidi_getmusic_bd3lm",
        f"loss.use_focal={'true' if args.use_focal else 'false'}",
        f"algo.T={args.T}",
        "training.from_pretrained=null",
        "training.ema=0.9999" if args.weight_source == "ema" else "training.ema=0.0",
        "mode=sample_eval",
        f"eval.checkpoint_path={args.ckpt}",
        "eval.disable_ema=False",
        # sampling.* must exist on the cfg for DIT.forward to work
        f"sampling.kv_cache={'true' if args.kv_cache else 'false'}",
        f"sampling.first_hitting={'true' if args.first_hitting else 'false'}",
        "sampling.var_length=false",
        f"sampling.nucleus_p={args.nucleus_p}",
        f"sampling.num_sample_batches=1",
        # logdir is unused in this script (we write our own jsonl)
        f"sampling.logdir={REPO}/sample_logs/_unused.csv",
        f"seed={args.seed}",
    ]
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        cfg = compose(config_name="config", overrides=overrides)
    return cfg


# ---------------------------------------------------------------------------
def ids_to_grid_tokens(tokenizer, ids_1d) -> list[list[str]]:
    """Convert a (SEQ_LEN,) int64 tensor / list to a (NUM_ROWS, NUM_COLS)
    list-of-lists of exact dictionary token strings. Dictionary entries are
    unique, so preserving spaces makes this representation reversible."""
    from xmidi_getmusic import NUM_ROWS, NUM_COLS, SEQ_LEN
    if isinstance(ids_1d, torch.Tensor):
        ids_1d = ids_1d.tolist()
    if len(ids_1d) != SEQ_LEN:
        raise ValueError(f"expected {SEQ_LEN} ids, got {len(ids_1d)}")
    out: list[list[str]] = []
    for r in range(NUM_ROWS):
        row = []
        for c in range(NUM_COLS):
            token_id = int(ids_1d[r * NUM_COLS + c])
            tok = tokenizer._convert_id_to_token(token_id)
            if tokenizer._convert_token_to_id(tok) != token_id:
                raise RuntimeError(
                    f"non-reversible dictionary token at id={token_id}: {tok!r}")
            row.append(tok)
        out.append(row)
    return out


def quick_match_stats(gt_grid: list[list[str]],
                      gen_grid: list[list[str]],
                      generation_mask=None) -> dict:
    """Exact match on the explicit DAW generation region ``G=1``.

    ``generation_mask`` is preferred.  The GT-``<UNK>`` fallback exists only
    for legacy callers and must not be used to redefine a canonical result.
    """
    from xmidi_getmusic import (
        NUM_ROWS, NUM_COLS, NUM_SOURCE_ROWS, NUM_PROMPT_COLS,
    )
    if generation_mask is not None:
        generation_mask = torch.as_tensor(generation_mask, dtype=torch.bool)
        if tuple(generation_mask.shape) != (NUM_ROWS, NUM_COLS):
            raise ValueError("generation_mask must have shape (16, 512)")
    res = {"per_target_row": {}, "all_target_content": {}}
    matches_total = 0
    n_total = 0
    for r in range(NUM_SOURCE_ROWS, NUM_ROWS):
        gt_c = gt_grid[r][NUM_PROMPT_COLS:NUM_COLS]
        gn_c = gen_grid[r][NUM_PROMPT_COLS:NUM_COLS]
        scored_pairs = [
            (gt_tok, gen_tok)
            for offset, (gt_tok, gen_tok) in enumerate(zip(gt_c, gn_c))
            if (bool(generation_mask[r, NUM_PROMPT_COLS + offset])
                if generation_mask is not None else gt_tok != "<UNK>")
        ]
        if not scored_pairs:
            res["per_target_row"][r] = {
                "is_unk_row": True, "match": None, "n_positions": 0}
            continue
        m = sum(1 for gt_tok, gen_tok in scored_pairs if gt_tok == gen_tok)
        n = len(scored_pairs)
        matches_total += m
        n_total += n
        res["per_target_row"][r] = {
            "is_unk_row": False,
            "match": m / n,
            "n_positions": n,
            "exact_matches": m,
        }
    res["all_target_content"] = {
        "match": (matches_total / n_total) if n_total else None,
        "exact_matches": matches_total,
        "n_positions": n_total,
    }
    return res


def _manifest_path(output_path: Path) -> Path:
    return Path(str(output_path) + ".manifest.json")


def _load_resume_indices(output_path: Path, provenance_id: str) -> set[int]:
    """Validate an existing canonical JSONL and return completed indices."""
    completed: set[int] = set()
    with open(output_path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("schema_version") != "bd3lms-music-v2.inference.v3":
                raise RuntimeError(
                    f"Cannot resume legacy/incompatible JSONL at line {line_number}")
            if record.get("provenance_id") != provenance_id:
                raise RuntimeError(
                    f"Provenance mismatch at JSONL line {line_number}")
            valid_index = int(record["valid_index"])
            if valid_index in completed:
                raise RuntimeError(f"Duplicate valid_index={valid_index} in resume JSONL")
            completed.add(valid_index)
    return completed


# ---------------------------------------------------------------------------
def run(args):
    from xmidi_getmusic import (
        NUM_ROWS, NUM_COLS,
        NUM_SOURCE_ROWS, NUM_PROMPT_COLS,
        PAD_ID, SEQ_LEN, UNK_ID,
    )
    from xmidi_getmusic.eval_protocol import (
        INFERENCE_SCHEMA_VERSION,
        MANIFEST_SCHEMA_VERSION,
        atomic_write_json,
        canonical_sha256,
        dataset_fingerprint,
        derive_sample_seed,
        inspect_v2_checkpoint,
        mask_to_rle,
        materialize_inference_weights,
        protocol_constants,
        seed_everything,
        source_fingerprint,
        validate_generation_mask,
        validate_manifest_provenance,
    )

    register_resolvers()
    cfg = build_eval_config(args)

    print(f"[batch_infer] ckpt          = {args.ckpt}")
    print(f"[batch_infer] weights       = {args.weight_source}")
    print(f"[batch_infer] model         = {args.model}")
    print(f"[batch_infer] block_size    = {args.block_size}")
    print(f"[batch_infer] bsz           = {args.bsz}")
    print(f"[batch_infer] T             = {args.T}")
    print(f"[batch_infer] seed          = {args.seed} (per-sample derived)")
    print(f"[batch_infer] kv_cache      = {args.kv_cache}")
    print(f"[batch_infer] first_hitting = {args.first_hitting}")
    print(f"[batch_infer] start_idx     = {args.start_idx}")
    print(f"[batch_infer] max_samples   = {args.max_samples}")
    print(f"[batch_infer] out           = {args.out}")

    if not torch.cuda.is_available():
        raise SystemExit("V2 block-diffusion inference requires a CUDA device")
    if not args.kv_cache:
        raise SystemExit(
            "BD3LMs-Music-V2 requires KV-cache sampling. The legacy no-cache "
            "mask does not implement the audited absolute-position contract.")
    if not args.first_hitting:
        raise SystemExit(
            "Canonical V2 inference requires first-hitting; the legacy general "
            "reverse branch is not part of the audited experiment protocol.")
    if args.first_hitting and args.T < args.block_size:
        raise SystemExit(
            f"first_hitting requires T >= block_size ({args.block_size}); "
            "each reverse step resolves at most one token.")
    if args.bsz != 1:
        raise SystemExit(
            "Canonical V2 inference requires bsz=1 so each valid_index has an "
            "independent, shard/resume-stable random seed.")

    import dataloader as bd3_dataloader
    import diffusion as bd3_diffusion

    checkpoint_meta = inspect_v2_checkpoint(
        args.ckpt,
        expected_hidden_size=int(cfg.model.hidden_size),
        require_ema=(args.weight_source == "ema"),
        expected_use_focal=args.use_focal,
        compute_sha256=True,
    )
    tokenizer = bd3_dataloader.get_tokenizer(cfg)
    print(f"[batch_infer] vocab_size    = {tokenizer.vocab_size}")
    print(f"[batch_infer] mask_id       = {tokenizer.mask_token_id}")
    print(f"[batch_infer] ckpt sha256   = {checkpoint_meta['sha256']}")
    print(f"[batch_infer] global_step   = {checkpoint_meta['global_step']}")

    print("[batch_infer] loading checkpoint ...")
    t0 = time.time()
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
    print(f"[batch_infer] ckpt loaded in {time.time()-t0:.1f}s")

    from xmidi_getmusic import get_xmidi_getmusic_dataset
    valid_set = get_xmidi_getmusic_dataset(cfg, tokenizer, split="valid")
    dataset_size = len(valid_set)
    print(f"[batch_infer] valid set size = {dataset_size}")

    end_idx = (
        dataset_size if args.max_samples is None
        else min(dataset_size, args.start_idx + args.max_samples)
    )
    if args.start_idx >= dataset_size:
        raise SystemExit(
            f"start_idx={args.start_idx} >= valid size {dataset_size}")

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = _manifest_path(out_path)
    print("[batch_infer] hashing validation data and inference sources ...")
    data_identity = dataset_fingerprint(
        Path(cfg.data.cache_dir) / "valid", "valid", cfg.data.dict_path)
    code_identity = source_fingerprint(
        REPO,
        (
            "diffusion.py",
            "noise_schedule.py",
            "xmidi_compat.py",
            "models/dit.py",
            "models/ema.py",
            "xmidi_getmusic/audit.py",
            "xmidi_getmusic/constants.py",
            "xmidi_getmusic/eval_protocol.py",
            "xmidi_getmusic/tokenizer.py",
            "scripts/eval/batch_infer_xmidi_getmusic.py",
        ),
    )
    identity = {
        "schema_version": INFERENCE_SCHEMA_VERSION,
        "checkpoint_sha256": checkpoint_meta["sha256"],
        "checkpoint_global_step": checkpoint_meta["global_step"],
        "weight_source": args.weight_source,
        "model": args.model,
        "data_split": "valid",
        "data_size": dataset_size,
        "dataset": data_identity,
        "inference_code": code_identity,
        "resolved_config_sha256": canonical_sha256(
            omegaconf.OmegaConf.to_container(cfg, resolve=True)
        ),
        "token_encoding": "exact unique whole_dict entries; spaces preserved",
        "base_seed": args.seed,
        "sampling": {
            "block_size": args.block_size,
            "T": args.T,
            "nucleus_p": args.nucleus_p,
            "first_hitting": args.first_hitting,
            "kv_cache": args.kv_cache,
            "attention_backend": "sdpa",
        },
        "protocol_constants": protocol_constants(),
    }
    provenance_id = canonical_sha256(identity)
    manifest = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "inference_schema_version": INFERENCE_SCHEMA_VERSION,
        "status": "running",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_jsonl": str(out_path),
        "provenance_id": provenance_id,
        "identity": identity,
        "checkpoint": checkpoint_meta,
        "requested_range": [args.start_idx, end_idx],
        "records_written": 0,
    }

    completed_indices: set[int] = set()
    if args.resume:
        if not out_path.is_file() or not manifest_path.is_file():
            raise RuntimeError("--resume requires both existing JSONL and manifest")
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_manifest_provenance(existing_manifest)
        if existing_manifest.get("provenance_id") != provenance_id:
            raise RuntimeError("resume manifest provenance does not match this run")
        completed_indices = _load_resume_indices(out_path, provenance_id)
        manifest["created_at_utc"] = existing_manifest.get(
            "created_at_utc", manifest["created_at_utc"])
    elif out_path.exists() and out_path.stat().st_size:
        raise RuntimeError(
            f"output already exists: {out_path}; use a new path or --resume")

    indices = [
        index for index in range(args.start_idx, end_idx)
        if index not in completed_indices
    ]
    manifest["records_written"] = len(completed_indices)
    atomic_write_json(manifest_path, manifest)
    f_out = open(out_path, "a" if args.resume else "w", encoding="utf-8")

    n_written = 0
    sum_match = 0.0
    n_match_records = 0
    total_nfes = 0
    total_backbone_forwards = 0
    t_start = time.time()
    pbar = tqdm(range(0, len(indices), args.bsz),
                desc="batch_infer", dynamic_ncols=True)

    with torch.no_grad():
        for i_off in pbar:
            batch_idx = indices[i_off:i_off + args.bsz]
            sample_seeds = [derive_sample_seed(args.seed, index) for index in batch_idx]
            # Canonical inference uses bsz=1, so this seeds every sample
            # independently of resume/shard boundaries.
            seed_everything(sample_seeds[0])

            batch_items = [valid_set[j] for j in batch_idx]
            ids = torch.stack(
                [it["input_ids"] for it in batch_items], dim=0).to("cuda")
            row_unk_mask = torch.stack(
                [it["row_unk_mask"] for it in batch_items], dim=0).to("cuda").bool()
            B = ids.shape[0]

            grid = ids.view(B, NUM_ROWS, NUM_COLS)
            source = grid[:, :NUM_SOURCE_ROWS].clone()
            target_prompts = grid[:, NUM_SOURCE_ROWS:, :NUM_PROMPT_COLS].clone()
            content_unk_mask = (
              grid[:, NUM_SOURCE_ROWS:, NUM_PROMPT_COLS:] == UNK_ID)
            generation_mask = torch.zeros_like(grid, dtype=torch.bool)
            generation_mask[:, NUM_SOURCE_ROWS:, NUM_PROMPT_COLS:] = (
                ~content_unk_mask)
            generation_mask = validate_generation_mask(
                generation_mask, condition_ids=grid)
            # Remove every reference music value under G before calling the
            # sampler. The placeholder is ignored and replaced by <MASK>.
            condition_grid = grid.clone()
            condition_grid[generation_mask] = PAD_ID

            sample_ids, nfes = model._xmidi_semi_ar_sampler(
                source, target_prompts, row_unk_mask, args.T,
                content_unk_mask=content_unk_mask,
                condition_grid=condition_grid,
                generation_mask=generation_mask)
            audit_records = getattr(model, "_last_xmidi_audit", None)
            if audit_records is None or len(audit_records) != B:
                raise RuntimeError("Sampler did not emit one V2 audit per sample")
            total_nfes += int(nfes)
            batch_backbone_forwards = int(
                audit_records[0].get("runtime", {}).get("forward_calls", 0))
            total_backbone_forwards += batch_backbone_forwards

            sample_ids = sample_ids.detach().cpu()
            gt_ids = ids.detach().cpu()
            generation_cpu = generation_mask.detach().cpu()
            row_unk_cpu = row_unk_mask.detach().cpu().tolist()

            for b in range(B):
                vidx = batch_idx[b]
                gt_grid = ids_to_grid_tokens(tokenizer, gt_ids[b])
                gen_grid = ids_to_grid_tokens(tokenizer, sample_ids[b])
                mask_rle = mask_to_rle(generation_cpu[b])

                stats = quick_match_stats(
                    gt_grid, gen_grid, generation_mask=generation_cpu[b])
                if stats["all_target_content"]["match"] is not None:
                    sum_match += stats["all_target_content"]["match"]
                    n_match_records += 1

                rec = {
                    "schema_version": INFERENCE_SCHEMA_VERSION,
                    "provenance_id": provenance_id,
                    "valid_index": int(vidx),
                    "sample_seed": int(sample_seeds[b]),
                    "ground_truth": gt_grid,
                    "gen": gen_grid,
                    "row_unk_mask": row_unk_cpu[b],
                    "generation_mask_rle": mask_rle,
                    "generation_mask_sha256": canonical_sha256({"rle": mask_rle}),
                    "generated_positions": int(generation_cpu[b].sum().item()),
                    "structural_audit": audit_records[b],
                    "match_stats": stats,
                    "diffusion_nfes": int(nfes) // max(B, 1),
                    "backbone_forward_calls": batch_backbone_forwards,
                    "nfes": int(nfes) // max(B, 1),  # legacy alias
                }
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_written += 1

            f_out.flush()
            if n_written % 25 == 0:
                manifest["records_written"] = len(completed_indices) + n_written
                manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
                atomic_write_json(manifest_path, manifest)

            avg_match = (sum_match / n_match_records) if n_match_records else float("nan")
            pbar.set_postfix({
                "n":      n_written,
                "avg_em": f"{avg_match:.4f}",
            })

    f_out.close()
    dt = time.time() - t_start
    manifest["status"] = "completed"
    manifest["records_written"] = len(completed_indices) + n_written
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["wall_time_seconds_this_invocation"] = dt
    manifest["total_diffusion_nfes_this_invocation"] = total_nfes
    manifest["total_backbone_forward_calls_this_invocation"] = total_backbone_forwards
    atomic_write_json(manifest_path, manifest)
    print(f"\n[batch_infer] wrote {n_written} records to {out_path}")
    print(f"[batch_infer] manifest        = {manifest_path}")
    print(f"[batch_infer] diffusion NFEs   = {total_nfes}")
    print(f"[batch_infer] backbone calls   = {total_backbone_forwards}")
    print(f"[batch_infer] wall time        = {dt:.1f}s "
          f"({n_written / max(dt, 1.0):.3f} samples/s)")
    if n_match_records:
        print(f"[batch_infer] avg target-content exact-match accuracy "
              f"(over {n_match_records} samples with non-empty GT) = "
              f"{sum_match / n_match_records:.4f}")


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True,
                   help="path to a Diffusion ckpt (e.g. .../checkpoints/best.ckpt)")
    p.add_argument("--model", default="xmidi_getmusic_small")
    p.add_argument("--block_size", type=int, default=16)
    p.add_argument("--bsz", type=int, default=1,
                   help="per-step batch size; first_hitting=true requires bsz=1")
    p.add_argument("--T", type=int, default=1000,
                   help="diffusion steps per content block")
    p.add_argument("--out", required=True,
                   help="output jsonl path")
    p.add_argument("--start_idx", type=int, default=0,
                   help="start from this valid index (for resume / sharding)")
    p.add_argument("--max_samples", type=int, default=None,
                   help="if set, only infer this many starting from start_idx")
    p.add_argument("--seed", type=int, default=1,
                   help="base seed; a stable seed is derived per valid_index")
    p.add_argument("--nucleus_p", type=float, default=0.95)
    p.add_argument("--weight_source", choices=("ema", "raw"), default="ema",
                   help="explicit checkpoint weight source; missing EMA fails fast")
    p.add_argument("--use_focal", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="resolved loss config identity; sampling must be invariant")
    p.add_argument("--no_ema", action="store_true",
                   help=argparse.SUPPRESS)
    p.add_argument("--no_first_hitting", action="store_true",
                   help="legacy flag; rejected by the canonical V2 protocol")
    p.add_argument("--no_kv_cache", action="store_true",
                   help="legacy compatibility flag; rejected by V2")
    p.add_argument("--resume", action="store_true",
                   help="append to --out instead of overwriting")
    args = p.parse_args()
    if args.no_ema:
        if "--weight_source" in sys.argv and args.weight_source != "raw":
            p.error("--no_ema conflicts with --weight_source=ema")
        args.weight_source = "raw"
    args.first_hitting = not args.no_first_hitting
    args.kv_cache = not args.no_kv_cache
    args.ckpt = str(Path(args.ckpt).resolve())
    return args


def main():
    args = parse_args()
    require_conda_environment()
    require_tmux_identity()
    require_wandb_project()
    require_project_path(args.out, label="inference output")
    run(args)


if __name__ == "__main__":
    main()
