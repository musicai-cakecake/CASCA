"""Inspect / export an XMIDI-GetMusic sampler CSV with ground truth from valid set.

For each generated sample we:
  1) Re-derive which valid-set sample was fed to the sampler. BD3-LM's
     ``_xmidi_sample_full`` builds the valid DataLoader with
     ``valid_seed=cfg.seed``, ``shuffle=True``, ``generator.manual_seed(seed)``.
     PyTorch's DataLoader (>=1.6) does **two** things with this generator
     on first iter:
       (a) ``_BaseDataLoaderIter.__init__`` pulls one int64 out of the
           generator to seed worker init / persistent_workers tracking,
           advancing the generator state by one int64 *before* the
           sampler gets a chance to consume it.
       (b) The ``RandomSampler(generator=loader.generator)`` then yields
           ``torch.randperm(N, generator=g).tolist()``.
     Replicating both gives the original valid-dataset index.
  2) Load that valid sample, decode it into 16x512 token strings.
  3) Combine with the generated row to produce a JSONL record:

        {
          "csv_index":       i,                    # 0..N-1 in the CSV
          "valid_index":     <int>,
          "prompts":         [[T_00..T_0_511], ..., [T_13_0..T_13_511]],
          "ground_trues":    [[...], ..., [...]],  # (16, 512) full valid sample
          "samples":         [[...], ..., [...]],  # (16, 512) full model output
          "match_stats":     {...}
        }

  All three fields share the same (NUM_ROWS, NUM_COLS) = (16, 512) shape
  so they can be compared element-wise.

Usage::

    python scripts/eval/inspect_xmidi_getmusic_sample.py \\
        --csv     sample_logs/xmidi_getmusic_bs16_T1000 \\
        --out     sample_logs/xmidi_getmusic_bs16_T1000.jsonl \\
        --seed    1 \\
        --pretty  sample_logs/xmidi_getmusic_bs16_T1000.row0.txt \\
        --idx     0
"""

from __future__ import annotations

import argparse
import ast
import csv as _csv
import json
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from xmidi_getmusic.constants import (
    NUM_COLS,
    NUM_PROMPT_COLS,
    NUM_ROWS,
    NUM_SOURCE_ROWS,
    SEQ_LEN,
)

os.environ.setdefault("HF_HOME", str(REPO / "hf_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(REPO / "hf_cache" / "transformers"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


# ---------------------------------------------------------------------
def parse_csv(csv_path: Path) -> list[list[str]]:
    """Return list of token-string lists, one per CSV row."""
    out: list[list[str]] = []
    with open(csv_path) as f:
        reader = _csv.reader(f)
        for row in reader:
            if not row:
                continue
            cell = row[0]
            parsed = ast.literal_eval(cell)
            if not (isinstance(parsed, list) and len(parsed) == 1):
                raise ValueError(
                    f"unexpected CSV cell: {cell[:80]}…  ({type(parsed)})")
            toks = parsed[0].split()
            if len(toks) != SEQ_LEN:
                raise ValueError(
                    f"sample {len(out)} has {len(toks)} tokens, expected {SEQ_LEN}")
            out.append(toks)
    return out


def reproduce_valid_indices(valid_set_len: int, seed: int) -> list[int]:
    """Reproduce the order in which the validation DataLoader yielded
    samples to ``_xmidi_sample_full`` when ``valid_seed=seed``.

    See :mod:`scripts.eval.inspect_xmidi_sample` for a detailed write-up
    of why we need to consume one int64 *before* calling randperm; same
    quirk applies to the new XMIDI-GetMusic loader.
    """
    g = torch.Generator().manual_seed(seed)
    _ = torch.empty((), dtype=torch.int64).random_(generator=g).item()
    return torch.randperm(valid_set_len, generator=g).tolist()


def build_tokenizer_and_dataset():
    """Construct the same XMIDIGetmusicTokenizer + dataset that
    ``main.py + dataloader.get_dataloaders`` would have built.
    """
    from xmidi_getmusic import XMIDIGetmusicTokenizer, get_xmidi_getmusic_dataset
    import omegaconf

    cfg = omegaconf.OmegaConf.create({
        "data": {
            "valid":      "xmidi_getmusic",
            "cache_dir":  str(REPO / "XMIDI_Dataset_output_v2"),
            "data_format": "idx",
        },
    })
    tokenizer = XMIDIGetmusicTokenizer(
        dict_path=str(REPO / "XMIDI_Dataset_output_v2" / "whole_dict.txt"))
    valid_set = get_xmidi_getmusic_dataset(cfg, tokenizer, split="valid")
    return tokenizer, valid_set


def ids_to_tokens(tokenizer, ids: torch.Tensor) -> list[str]:
    """(8192,) int64 -> list[8192] of token strings."""
    out = []
    for value in ids.tolist():
        token_id = int(value)
        token = tokenizer._convert_id_to_token(token_id)
        if tokenizer._convert_token_to_id(token) != token_id:
            raise RuntimeError(f"non-reversible token id {token_id}: {token!r}")
        out.append(token)
    return out


def make_prompts_view(gt_tokens: list[str], mask_token: str) -> list[str]:
    """Build the user-visible "prompts" view of a sample.

      - rows 0..3 (source rows): all 512 cols are visible
      - rows 4..15 (target rows): only cols 0..15 are visible (the prompts);
        cols 16..511 (target content) are replaced by mask_token to mark
        "the model is asked to fill these in".
    """
    if len(gt_tokens) != SEQ_LEN:
        raise ValueError(f"expected {SEQ_LEN} tokens, got {len(gt_tokens)}")
    out = list(gt_tokens)
    for r in range(NUM_SOURCE_ROWS, NUM_ROWS):
        base = r * NUM_COLS
        for c in range(NUM_PROMPT_COLS, NUM_COLS):
            out[base + c] = mask_token
    return out


def to_grid(tokens: list[str]) -> list[list[str]]:
    """Row-major reshape (8192,) -> (16, 512) list of lists."""
    if len(tokens) != SEQ_LEN:
        raise ValueError(f"expected {SEQ_LEN} tokens, got {len(tokens)}")
    return [tokens[r * NUM_COLS:(r + 1) * NUM_COLS] for r in range(NUM_ROWS)]


# ---------------------------------------------------------------------
def write_pretty(path: Path, prompts: list[str], gt: list[str], gen: list[str],
                 valid_index: int, csv_index: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(f"# csv_index   = {csv_index}\n")
        f.write(f"# valid_index = {valid_index}\n\n")

        for r in range(NUM_ROWS):
            kind = "SRC " if r < NUM_SOURCE_ROWS else "TGT "
            base = r * NUM_COLS
            f.write(f"### row {r}  ({kind})\n")
            f.write("PROMPTS  [0..15]   : "
                    + " ".join(prompts[base:base + NUM_PROMPT_COLS]) + "\n")
            if r < NUM_SOURCE_ROWS:
                f.write("SRC      [16..511] : "
                        + " ".join(prompts[base + NUM_PROMPT_COLS:base + NUM_COLS]) + "\n")
            else:
                f.write("GT       [16..511] : "
                        + " ".join(gt[base + NUM_PROMPT_COLS:base + NUM_COLS]) + "\n")
                f.write("SAMPLE   [16..511] : "
                        + " ".join(gen[base + NUM_PROMPT_COLS:base + NUM_COLS]) + "\n")
            f.write("\n")


def quick_match_stats(gt: list[str], gen: list[str]) -> dict:
    """Per-target-row exact-match accuracy on the content region only."""
    res = {"per_target_row": {}, "all_target_content": {}}
    matches_total = 0
    n_total = 0
    for r in range(NUM_SOURCE_ROWS, NUM_ROWS):
        base = r * NUM_COLS
        gt_c = gt[base + NUM_PROMPT_COLS:base + NUM_COLS]
        gn_c = gen[base + NUM_PROMPT_COLS:base + NUM_COLS]
        scored_pairs = [
            (gt_tok, gen_tok)
            for gt_tok, gen_tok in zip(gt_c, gn_c)
            if gt_tok != "<UNK>"
        ]
        if not scored_pairs:
            res["per_target_row"][r] = {"is_unk_row": True, "match": None,
                                         "n_positions": 0}
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


# ---------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="path to the sampling CSV")
    p.add_argument("--out", required=True, help="output JSONL path")
    p.add_argument("--seed", type=int, default=1,
                   help="cfg.seed used during sampling (default 1)")
    p.add_argument("--pretty", default=None,
                   help="also write a single human-readable .txt for --idx")
    p.add_argument("--idx", type=int, default=None,
                   help="which CSV row to dump for --pretty (default 0)")
    args = p.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    print(f"[inspect_xmidi_getmusic] csv  = {csv_path}")
    print(f"[inspect_xmidi_getmusic] out  = {out_path}")
    print(f"[inspect_xmidi_getmusic] seed = {args.seed}")

    gen_samples = parse_csv(csv_path)
    print(f"[inspect_xmidi_getmusic] {len(gen_samples)} generated samples in CSV")

    tokenizer, valid_set = build_tokenizer_and_dataset()
    print(f"[inspect_xmidi_getmusic] valid set size = {len(valid_set)}")
    mask_token = tokenizer.mask_token  # "<MASK>"

    order = reproduce_valid_indices(len(valid_set), args.seed)
    if len(gen_samples) > len(order):
        raise SystemExit(
            f"CSV has more samples ({len(gen_samples)}) than valid set ({len(order)})")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pretty_path = Path(args.pretty) if args.pretty else None
    pretty_idx = args.idx if args.idx is not None else 0

    n_written = 0
    sum_match = 0.0
    n_match_records = 0

    with open(out_path, "w") as f_out:
        for i, gen_tokens in enumerate(gen_samples):
            valid_idx = int(order[i])
            sample_dict = valid_set[valid_idx]
            gt_ids = sample_dict["input_ids"]            # (8192,) int64
            gt_tokens = ids_to_tokens(tokenizer, gt_ids)
            prompts_view = make_prompts_view(gt_tokens, mask_token)

            stats = quick_match_stats(gt_tokens, gen_tokens)
            if stats["all_target_content"]["match"] is not None:
                sum_match += stats["all_target_content"]["match"]
                n_match_records += 1

            rec = {
                "csv_index":     i,
                "valid_index":   valid_idx,
                # all three reshaped to (16, 512) so they can be indexed
                # as arr[row][col] and compared element-wise.
                "prompts":       to_grid(prompts_view),
                "ground_trues":  to_grid(gt_tokens),
                "samples":       to_grid(gen_tokens),
                "match_stats":   stats,
            }
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1

            if pretty_path is not None and i == pretty_idx:
                write_pretty(pretty_path, prompts_view, gt_tokens, gen_tokens,
                             valid_idx, i)

    print(f"[inspect_xmidi_getmusic] wrote {n_written} jsonl records -> {out_path}")
    if n_match_records:
        avg_match = sum_match / n_match_records
        print(f"[inspect_xmidi_getmusic] avg target-content exact-match accuracy "
              f"(over {n_match_records} samples with non-empty GT) = {avg_match:.4f}")
    if pretty_path is not None:
        print(f"[inspect_xmidi_getmusic] wrote pretty view (csv idx={pretty_idx}) -> {pretty_path}")


if __name__ == "__main__":
    main()
