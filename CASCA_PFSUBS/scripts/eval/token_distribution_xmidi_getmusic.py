"""Tally token distributions in an XMIDI-GetMusic batch-inference JSONL.

Usage:
    python scripts/eval/token_distribution_xmidi_getmusic.py \\
        --jsonl sample_logs/xmidi_getmusic_valid_full.jsonl \\
        --out_dir sample_logs/token_dist \\
        --top_k 30

For every sample dict in the input jsonl this script accumulates two
``collections.Counter``\\ s:

  * ``gt_counts``  - over all 16*512 = 8192 tokens of ``ground_truth``
  * ``gen_counts`` - over all 16*512 = 8192 tokens of ``gen``

Then it prints a top-K table to stdout and writes 4 CSVs to ``--out_dir``:

  * ``gt_token_distribution.csv``       per-token count + share for GT
  * ``gen_token_distribution.csv``      per-token count + share for gen
  * ``gt_vs_gen_distribution.csv``      union-key table with both columns
  * ``per_sample_metrics.csv``          one row per sample with:
        - n_gt_tokens / n_gen_tokens (after region filter)
        - gt_unique  / gen_unique
        - js_divergence (bits, in [0, 1]; 0 = identical)
        - bleu1 / bleu2 / bleu3 / bleu4 / bleu_avg (sentence-level BLEU)
        - exact_match (re-uses match_stats from the inference jsonl when
          available, else recomputed on the same region)

Notes
-----
* Canonical tokens are exact, unique ``whole_dict.txt`` entries; spaces are
  preserved so ID/token conversion is reversible.
* The JSONL is streamed line-by-line; memory remains O(V2 vocabulary size).
* The default region is the explicit per-record DAW ``generation_mask_rle``
  (G=1). ``--full_grid`` is diagnostic only and includes copied conditions.
* BLEU reuses the canonical implementation and never forms n-grams across
  target rows or across fixed G=0 holes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from xmidi_getmusic.constants import (
    NUM_COLS,
    NUM_PROMPT_COLS,
    NUM_ROWS,
    NUM_SOURCE_ROWS,
    SEQ_LEN,
)
from xmidi_getmusic.eval_protocol import (
    INFERENCE_SCHEMA_VERSION,
    generation_mask_from_record,
    selected_token_segments,
    validate_manifest_provenance,
)
from scripts.eval.objective_evaluation import segmented_bleu_scores


# ---------------------------------------------------------------------------
def paired_grid_tokens_in_region(gt_grid, gen_grid, only_target_content: bool,
                                 row_unk_mask, generation_mask=None):
    """Return aligned GT/generated tokens under the selected region filter.

    In target-content mode, the GT ``<UNK>`` layout is a fixed DAW condition;
    both sequences exclude those positions before metrics are computed.
    """
    if not only_target_content:
        return (
            [tok for row in gt_grid for tok in row],
            [tok for row in gen_grid for tok in row],
        )
    gt_out: list[str] = []
    gen_out: list[str] = []
    for r in range(NUM_SOURCE_ROWS, NUM_ROWS):
        if row_unk_mask is not None and r < len(row_unk_mask) and row_unk_mask[r]:
            continue
        gt_row = gt_grid[r][NUM_PROMPT_COLS:NUM_COLS]
        gen_row = gen_grid[r][NUM_PROMPT_COLS:NUM_COLS]
        for offset, (gt_tok, gen_tok) in enumerate(zip(gt_row, gen_row)):
            selected = (
                bool(generation_mask[r, NUM_PROMPT_COLS + offset])
                if generation_mask is not None else gt_tok != "<UNK>"
            )
            if selected:
                gt_out.append(gt_tok)
                gen_out.append(gen_tok)
    return gt_out, gen_out


# ---------------------------------------------------------------------------
def js_divergence_from_counters(p: Counter, q: Counter,
                                np_total: int, nq_total: int) -> float:
    """Jensen-Shannon divergence (bits) between two empirical token
    distributions. Returns NaN if either side has zero tokens.

    JSD is symmetric, lies in [0, 1] when computed with log base 2, and
    equals 0 iff the two distributions are identical."""
    if np_total == 0 or nq_total == 0:
        return float("nan")
    keys = set(p) | set(q)
    js = 0.0
    for k in keys:
        pp = p.get(k, 0) / np_total
        qq = q.get(k, 0) / nq_total
        m = 0.5 * (pp + qq)
        if pp > 0 and m > 0:
            js += 0.5 * pp * math.log2(pp / m)
        if qq > 0 and m > 0:
            js += 0.5 * qq * math.log2(qq / m)
    return js


# ---------------------------------------------------------------------------
def write_single_csv(out_path: Path, counter: Counter, total: int):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_items = counter.most_common()
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "token", "count", "share"])
        for rank, (tok, cnt) in enumerate(sorted_items, start=1):
            share = cnt / total if total else 0.0
            w.writerow([rank, tok, cnt, f"{share:.8f}"])


def write_combined_csv(out_path: Path, gt: Counter, gen: Counter,
                       n_gt: int, n_gen: int):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(set(gt) | set(gen),
                  key=lambda k: -(gt.get(k, 0) + gen.get(k, 0)))
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "token",
                    "gt_count", "gt_share",
                    "gen_count", "gen_share",
                    "share_diff_gen_minus_gt"])
        for rank, k in enumerate(keys, start=1):
            gc = gt.get(k, 0)
            gn = gen.get(k, 0)
            gs = gc / n_gt if n_gt else 0.0
            ns = gn / n_gen if n_gen else 0.0
            w.writerow([rank, k, gc, f"{gs:.8f}",
                        gn, f"{ns:.8f}",
                        f"{(ns - gs):+.8f}"])


# ---------------------------------------------------------------------------
def tally(jsonl_path: Path, only_target_content: bool, per_sample_csv: Path,
          provenance_id: str):
    """Single streaming pass:
       1) accumulate global gt / gen Counters,
       2) for every sample compute per-sample JS divergence + BLEU + EM
          and append a row to ``per_sample_csv``.
    """
    gt_global = Counter()
    gen_global = Counter()
    n_samples = 0
    n_gt_tokens = 0
    n_gen_tokens = 0
    seen_indices = set()

    per_sample_csv.parent.mkdir(parents=True, exist_ok=True)

    js_acc: list[float] = []
    bleu_acc: list[float] = []
    bleu_n_acc: list[list[float]] = [[], [], [], []]
    em_acc: list[float] = []

    with open(per_sample_csv, "w", newline="") as fcsv, \
         open(jsonl_path) as fin:
        w = csv.writer(fcsv)
        w.writerow([
            "valid_index", "n_gt_tokens", "n_gen_tokens",
            "gt_unique", "gen_unique",
            "js_divergence_bits",
            "bleu1", "bleu2", "bleu3", "bleu4", "bleu_avg",
            "exact_match",
        ])

        for ln, line in enumerate(fin):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"malformed canonical JSONL at line {ln + 1}: {error}"
                ) from error
            if rec.get("schema_version") != INFERENCE_SCHEMA_VERSION:
                raise RuntimeError(f"invalid canonical schema at line {ln + 1}")
            if rec.get("provenance_id") != provenance_id:
                raise RuntimeError(f"provenance mismatch at line {ln + 1}")
            valid_index = int(rec.get("valid_index", -1))
            if valid_index in seen_indices:
                raise RuntimeError(f"duplicate valid_index={valid_index}")
            seen_indices.add(valid_index)
            audit = rec.get("structural_audit", {})
            if not audit.get("all_pass", False):
                raise RuntimeError(f"structural audit failed at line {ln + 1}")

            row_unk_mask = rec.get("row_unk_mask", None)
            generation_mask = (
                generation_mask_from_record(rec) if only_target_content else None
            )

            gt_seq, gen_seq = paired_grid_tokens_in_region(
                rec["ground_truth"], rec["gen"],
                only_target_content, row_unk_mask, generation_mask)

            # per-sample counters
            gt_c = Counter(gt_seq)
            gen_c = Counter(gen_seq)

            # global accumulation
            gt_global.update(gt_c)
            gen_global.update(gen_c)
            n_gt_tokens += sum(gt_c.values())
            n_gen_tokens += sum(gen_c.values())

            # metrics
            js = js_divergence_from_counters(
                gt_c, gen_c, sum(gt_c.values()), sum(gen_c.values()))
            if only_target_content:
                segments = selected_token_segments(rec)
                reference_segments = [segment[2] for segment in segments]
                generated_segments = [segment[3] for segment in segments]
            else:
                reference_segments = [gt_seq]
                generated_segments = [gen_seq]
            _, bleu_scores = segmented_bleu_scores(
                reference_segments, generated_segments, max_order=4)
            bleu_avg = sum(bleu_scores) / len(bleu_scores)
            # exact-match: prefer pre-computed value from the inference
            # jsonl when in --gt_only_in_loss_region mode (definitions
            # match), else recompute element-wise on the chosen region.
            if (only_target_content
                    and "match_stats" in rec
                    and rec["match_stats"].get("all_target_content", {}).get("match")
                    is not None):
                em = rec["match_stats"]["all_target_content"]["match"]
            else:
                if len(gt_seq) == 0:
                    em = float("nan")
                else:
                    L = min(len(gt_seq), len(gen_seq))
                    em = sum(1 for a, b in zip(gt_seq[:L], gen_seq[:L])
                             if a == b) / max(len(gt_seq), 1)

            w.writerow([
                valid_index, len(gt_seq), len(gen_seq),
                len(gt_c), len(gen_c),
                f"{js:.6f}",
                f"{bleu_scores[0]:.6f}", f"{bleu_scores[1]:.6f}",
                f"{bleu_scores[2]:.6f}", f"{bleu_scores[3]:.6f}",
                f"{bleu_avg:.6f}",
                f"{em:.6f}" if em == em else "",  # NaN -> ""
            ])
            fcsv.flush()

            if not math.isnan(js):
                js_acc.append(js)
            bleu_acc.append(bleu_avg)
            for i in range(4):
                bleu_n_acc[i].append(bleu_scores[i])
            if em == em:  # not NaN
                em_acc.append(em)
            n_samples += 1

    return (gt_global, gen_global, n_samples, n_gt_tokens, n_gen_tokens,
            js_acc, bleu_acc, bleu_n_acc, em_acc)


# ---------------------------------------------------------------------------
def fmt_top_table(name: str, counter: Counter, total: int, top_k: int) -> str:
    lines = []
    lines.append(f"\n[{name}]  unique tokens = {len(counter):>6d}   "
                 f"total tokens = {total:>10d}")
    lines.append(f"  {'#':>4}  {'count':>10}  {'share':>10}  token")
    lines.append("  " + "-" * 60)
    for rank, (tok, cnt) in enumerate(counter.most_common(top_k), start=1):
        share = cnt / total if total else 0.0
        lines.append(f"  {rank:>4}  {cnt:>10d}  {share:>9.4%}  {tok}")
    if len(counter) > top_k:
        rest = sum(c for _, c in counter.most_common()[top_k:])
        rest_share = rest / total if total else 0.0
        lines.append(f"  {'...':>4}  {rest:>10d}  {rest_share:>9.4%}  "
                     f"(rest of {len(counter) - top_k} tokens)")
    return "\n".join(lines)


def _summary(name: str, xs: list[float]) -> str:
    if not xs:
        return f"  {name}: (no samples)"
    n = len(xs)
    mean = sum(xs) / n
    xs_s = sorted(xs)
    p50 = xs_s[n // 2]
    p10 = xs_s[max(0, n * 10 // 100)]
    p90 = xs_s[min(n - 1, n * 90 // 100)]
    return (f"  {name:<22s}: n={n:<5d}  mean={mean:.4f}  "
            f"p10={p10:.4f}  p50={p50:.4f}  p90={p90:.4f}  "
            f"min={min(xs):.4f}  max={max(xs):.4f}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True,
                    help="batch_infer_xmidi_getmusic.py output jsonl")
    ap.add_argument("--out_dir", default="sample_logs/token_dist",
                    help="directory for output CSVs")
    ap.add_argument("--top_k", type=int, default=30,
                    help="how many top tokens to print")
    ap.add_argument("--gt_only_in_loss_region", action="store_true",
                    help="deprecated compatibility flag; canonical G=1 is default")
    ap.add_argument("--full_grid", action="store_true",
                    help="diagnostic only: include copied source/prompt/fixed cells")
    args = ap.parse_args()
    args.gt_only_in_loss_region = not args.full_grid

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        raise SystemExit(f"jsonl not found: {jsonl_path}")
    manifest_path = Path(str(jsonl_path) + ".manifest.json")
    if not manifest_path.is_file():
        raise RuntimeError(f"canonical manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest_provenance(manifest)
    if manifest.get("status") != "completed":
        raise RuntimeError(f"manifest status is {manifest.get('status')!r}")
    provenance_id = manifest["provenance_id"]
    out_dir = Path(args.out_dir)
    per_sample_csv = out_dir / "per_sample_metrics.csv"

    print(f"[token_dist] reading  : {jsonl_path}")
    print(f"[token_dist] out_dir  : {out_dir}")
    print(f"[token_dist] mode     : "
          f"{'target-content only' if args.gt_only_in_loss_region else 'full grid (16x512)'}")

    (gt, gen, n_samples, n_gt_tok, n_gen_tok,
     js_acc, bleu_acc, bleu_n_acc, em_acc) = tally(
        jsonl_path, args.gt_only_in_loss_region, per_sample_csv, provenance_id)
    if int(manifest.get("records_written", -1)) != n_samples:
        raise RuntimeError(
            f"manifest records_written={manifest.get('records_written')}, "
            f"JSONL has {n_samples} unique records"
        )
    print(f"[token_dist] samples  : {n_samples}")
    print(f"[token_dist] tokens   : gt={n_gt_tok}, gen={n_gen_tok}")

    # Distribution CSVs
    write_single_csv(out_dir / "gt_token_distribution.csv", gt, n_gt_tok)
    write_single_csv(out_dir / "gen_token_distribution.csv", gen, n_gen_tok)
    write_combined_csv(out_dir / "gt_vs_gen_distribution.csv",
                       gt, gen, n_gt_tok, n_gen_tok)
    print(f"[token_dist] wrote: {out_dir / 'gt_token_distribution.csv'}")
    print(f"[token_dist] wrote: {out_dir / 'gen_token_distribution.csv'}")
    print(f"[token_dist] wrote: {out_dir / 'gt_vs_gen_distribution.csv'}")
    print(f"[token_dist] wrote: {per_sample_csv}")

    # stdout tables
    print(fmt_top_table("ground_truth", gt, n_gt_tok, args.top_k))
    print(fmt_top_table("gen",          gen, n_gen_tok, args.top_k))

    gt_set, gen_set = set(gt), set(gen)
    only_gen = gen_set - gt_set
    print(f"\n[set comparison]")
    print(f"  unique tokens in gt  : {len(gt_set)}")
    print(f"  unique tokens in gen : {len(gen_set)}")
    print(f"  intersection         : {len(gt_set & gen_set)}")
    print(f"  only in gt           : {len(gt_set - gen_set)}")
    print(f"  only in gen          : {len(only_gen)}")
    if only_gen and len(only_gen) <= 20:
        print(f"  (gen-only tokens)    : {sorted(only_gen)}")

    # ---- divergence + BLEU summary ----
    js_global = js_divergence_from_counters(gt, gen, n_gt_tok, n_gen_tok)
    print(f"\n[corpus-level divergence]")
    print(f"  Jensen-Shannon divergence (bits, in [0,1]) = {js_global:.6f}")
    print(f"  -> 0 means identical; ~1 means disjoint supports")

    print(f"\n[per-sample summary]   (saved to {per_sample_csv.name})")
    print(_summary("JS divergence (bits)", js_acc))
    for i in range(4):
        print(_summary(f"BLEU-{i+1}",        bleu_n_acc[i]))
    print(_summary("BLEU (avg of 1..4)", bleu_acc))
    print(_summary("exact-match",        em_acc))


if __name__ == "__main__":
    main()
