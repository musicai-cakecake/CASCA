"""Canonical objective evaluation for BD3LMs-Music-V2.

The evaluator consumes the manifest-backed JSONL produced by
``batch_infer_xmidi_getmusic.py``.  Every content metric uses the explicit
DAW generation mask ``G=1`` stored in each record; source, prompts and fixed
structural cells never contribute to music-quality scores.

Reported metric names are literal:

* BLEU-1..4 are cumulative sentence BLEU scores with orders 1..4.
* Modified n-gram precision is reported separately and is not called BLEU.
* Token-embedding greedy match is a static embedding metric, not BERTScore.
* Top-k frequency coverage is not retrieval Recall@k.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("WANDB_MODE", "disabled")

from xmidi_getmusic.constants import (  # noqa: E402
    NUM_COLS,
    NUM_PROMPT_COLS,
    NUM_ROWS,
    NUM_SOURCE_ROWS,
    ROW_NAMES,
    VOCAB_SIZE,
)
from xmidi_getmusic.eval_protocol import (  # noqa: E402
    INFERENCE_SCHEMA_VERSION,
    canonical_sha256,
    inspect_v2_checkpoint,
    iter_selected_token_pairs,
    selected_token_segments,
    sha256_file,
    validate_manifest_provenance,
)

UNK_TOK = "<UNK>"
PAD_TOK = "<pad>"
MASK_TOK = "<MASK>"
SPECIALS = {UNK_TOK, PAD_TOK, MASK_TOK}
AUDIT_NAMES = (
    "shape_and_vocabulary",
    "source_invariance",
    "prompt_invariance",
    "cell_unk_invariance",
    "fixed_condition_invariance",
    "inactive_row_invariance",
    "diffusion_completion",
    "attention_cache_consistency",
)


def paired_content_tokens(gt_grid, gen_grid, row_unk_mask, generation_mask=None):
    """Compatibility helper returning aligned target-content token sequences.

    Canonical records must pass ``generation_mask``.  The GT-``<UNK>`` branch
    exists only for legacy tests/files.
    """
    gt_out, gen_out = [], []
    for row in range(NUM_SOURCE_ROWS, NUM_ROWS):
        if row_unk_mask is not None and row_unk_mask[row]:
            continue
        for col in range(NUM_PROMPT_COLS, NUM_COLS):
            selected = (
                bool(generation_mask[row][col])
                if generation_mask is not None
                else gt_grid[row][col] != UNK_TOK
            )
            if selected:
                gt_out.append(gt_grid[row][col])
                gen_out.append(gen_grid[row][col])
    return gt_out, gen_out


def _ngrams(sequence, order):
    if order <= 0 or len(sequence) < order:
        return Counter()
    return Counter(
        tuple(sequence[index:index + order])
        for index in range(len(sequence) - order + 1)
    )


def sentence_bleu_scores(reference, hypothesis, max_order=4):
    """Return modified precisions and cumulative BLEU-1..``max_order``."""
    if not hypothesis:
        return [0.0] * max_order, [0.0] * max_order
    precisions = []
    for order in range(1, max_order + 1):
        hypothesis_ngrams = _ngrams(hypothesis, order)
        reference_ngrams = _ngrams(reference, order)
        denominator = sum(hypothesis_ngrams.values())
        if denominator == 0:
            precisions.append(0.0)
            continue
        overlap = sum(
            min(count, reference_ngrams.get(ngram, 0))
            for ngram, count in hypothesis_ngrams.items()
        )
        precisions.append(
            overlap / denominator if overlap else 1.0 / (denominator + 1)
        )
    brevity_penalty = (
        1.0
        if len(hypothesis) >= len(reference) or not reference
        else math.exp(1.0 - len(reference) / len(hypothesis))
    )
    bleu = []
    for order in range(1, max_order + 1):
        selected = precisions[:order]
        score = (
            0.0
            if any(value <= 0 for value in selected)
            else brevity_penalty
            * math.exp(sum(math.log(value) for value in selected) / order)
        )
        bleu.append(score)
    return precisions, bleu


def sentence_bleu(reference, hypothesis, max_n=4):
    """Legacy API: return modified precisions and cumulative BLEU-``max_n``."""
    precisions, bleu = sentence_bleu_scores(reference, hypothesis, max_n)
    return precisions, bleu[-1]


def segmented_bleu_scores(reference_segments, hypothesis_segments, max_order=4):
    """BLEU over contiguous same-row G=1 segments, without fake boundaries."""
    if len(reference_segments) != len(hypothesis_segments):
        raise ValueError("reference/hypothesis segment counts differ")
    reference_length = sum(len(segment) for segment in reference_segments)
    hypothesis_length = sum(len(segment) for segment in hypothesis_segments)
    if hypothesis_length == 0:
        return [0.0] * max_order, [0.0] * max_order

    precisions = []
    for order in range(1, max_order + 1):
        reference_ngrams = Counter()
        hypothesis_ngrams = Counter()
        for reference, hypothesis in zip(reference_segments, hypothesis_segments):
            reference_ngrams.update(_ngrams(reference, order))
            hypothesis_ngrams.update(_ngrams(hypothesis, order))
        denominator = sum(hypothesis_ngrams.values())
        if denominator == 0:
            precisions.append(0.0)
            continue
        overlap = sum(
            min(count, reference_ngrams.get(ngram, 0))
            for ngram, count in hypothesis_ngrams.items()
        )
        precisions.append(
            overlap / denominator if overlap else 1.0 / (denominator + 1)
        )
    brevity_penalty = (
        1.0
        if hypothesis_length >= reference_length or reference_length == 0
        else math.exp(1.0 - reference_length / hypothesis_length)
    )
    bleu = []
    for order in range(1, max_order + 1):
        selected = precisions[:order]
        bleu.append(
            0.0
            if any(value <= 0 for value in selected)
            else brevity_penalty
            * math.exp(sum(math.log(value) for value in selected) / order)
        )
    return precisions, bleu


def shannon_entropy_bits(counter):
    total = sum(counter.values())
    if total == 0:
        return float("nan")
    return -sum(
        (count / total) * math.log2(count / total)
        for count in counter.values()
        if count
    )


def js_divergence_bits(p_counter, q_counter):
    p_total, q_total = sum(p_counter.values()), sum(q_counter.values())
    if p_total == 0 or q_total == 0:
        return float("nan")
    divergence = 0.0
    for token in set(p_counter) | set(q_counter):
        p = p_counter.get(token, 0) / p_total
        q = q_counter.get(token, 0) / q_total
        mixture = 0.5 * (p + q)
        if p:
            divergence += 0.5 * p * math.log2(p / mixture)
        if q:
            divergence += 0.5 * q * math.log2(q / mixture)
    return divergence


def top_k_frequency_coverage(gt_counter, generated_counter, k):
    reference_top = [token for token, _ in gt_counter.most_common(k)]
    generated_top = {token for token, _ in generated_counter.most_common(k)}
    if not reference_top:
        return float("nan"), []
    return (
        sum(token in generated_top for token in reference_top) / len(reference_top),
        reference_top,
    )


def load_embedding_table(ckpt_path, weight_source):
    """Extract the raw or EMA tied token embedding without instantiating DIT."""
    import torch

    metadata = inspect_v2_checkpoint(
        ckpt_path, require_ema=(weight_source == "ema"), compute_sha256=False
    )
    checkpoint = torch.load(
        ckpt_path, map_location="cpu", weights_only=False, mmap=True
    )
    raw_embedding = checkpoint["state_dict"][metadata["embedding_key"]]
    if weight_source == "raw":
        embedding = raw_embedding
        source = metadata["embedding_key"]
    else:
        index = metadata["ema_embedding_index"]
        embedding = checkpoint["ema"]["shadow_params"][index]
        source = f"ema.shadow_params[{index}]"
    if tuple(embedding.shape) != tuple(metadata["embedding_shape"]):
        raise ValueError("selected embedding shape does not match checkpoint metadata")
    if embedding.shape[0] != VOCAB_SIZE:
        raise ValueError("selected embedding does not use the V2 vocabulary")
    result = embedding.float().clone()
    del checkpoint
    return result, f"{source} shape={tuple(result.shape)}"


def build_display_to_id(dict_path):
    with open(dict_path, encoding="utf-8") as handle:
        tokens = [line.rstrip("\n") for line in handle]
    while tokens and tokens[-1] == "":
        tokens.pop()
    mapping = {token: index for index, token in enumerate(tokens)}
    mapping[UNK_TOK] = len(tokens)
    mapping[MASK_TOK] = len(tokens) + 1
    if len(mapping) != len(tokens) + 2:
        raise RuntimeError("whole_dict tokens are not unique/reversible")
    return mapping, len(tokens)


def token_embedding_greedy_match(
    samples,
    display_to_id,
    embedding,
    device,
    max_samples=None,
    chunk_size=1024,
):
    """Greedy cosine matching over static tied embeddings, evaluated in chunks."""
    import torch

    embedding = embedding.to(device)
    embedding = embedding / (embedding.norm(dim=-1, keepdim=True) + 1e-12)
    precision_values, recall_values, f1_values = [], [], []
    for reference_tokens, generated_tokens in samples:
        if max_samples is not None and len(f1_values) >= max_samples:
            break
        if not reference_tokens or not generated_tokens:
            continue
        reference_ids = torch.tensor(
            [display_to_id[token] for token in reference_tokens], device=device
        )
        generated_ids = torch.tensor(
            [display_to_id[token] for token in generated_tokens], device=device
        )
        reference = embedding[reference_ids]
        generated = embedding[generated_ids]

        generated_best = []
        for start in range(0, len(generated), chunk_size):
            generated_best.append(
                (generated[start:start + chunk_size] @ reference.T).max(dim=1).values
            )
        reference_best = []
        for start in range(0, len(reference), chunk_size):
            reference_best.append(
                (reference[start:start + chunk_size] @ generated.T).max(dim=1).values
            )
        precision = torch.cat(generated_best).mean().item()
        recall = torch.cat(reference_best).mean().item()
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)

    mean = lambda values: sum(values) / len(values) if values else float("nan")
    return {
        "precision": mean(precision_values),
        "recall": mean(recall_values),
        "f1": mean(f1_values),
        "n_samples_scored": len(f1_values),
    }


def _finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite_json(item) for item in value]
    return value


def _load_manifest(jsonl, manifest_arg, allow_legacy, allow_incomplete):
    manifest_path = Path(manifest_arg) if manifest_arg else Path(str(jsonl) + ".manifest.json")
    if not manifest_path.is_file():
        if allow_legacy:
            return None, manifest_path
        raise RuntimeError(f"canonical manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest_provenance(manifest)
    if manifest.get("status") != "completed" and not allow_incomplete:
        raise RuntimeError(
            f"manifest status is {manifest.get('status')!r}; use --allow_incomplete "
            "only for an explicitly partial experiment"
        )
    return manifest, manifest_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--weight_source", choices=("auto", "ema", "raw"), default="auto")
    parser.add_argument(
        "--dict_path", default=str(REPO / "XMIDI_Dataset_output_v2/whole_dict.txt")
    )
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no_embedding_match", action="store_true")
    parser.add_argument("--max_embedding_samples", type=int, default=None)
    parser.add_argument("--allow_legacy_jsonl", action="store_true")
    parser.add_argument("--allow_incomplete", action="store_true")
    parser.add_argument(
        "--expect_valid_size", type=int, default=6817,
        help="required canonical validation-set size; use 0 for a deliberate subset"
    )
    args = parser.parse_args()

    jsonl = Path(args.jsonl).resolve()
    if not jsonl.is_file():
        raise SystemExit(f"JSONL not found: {jsonl}")
    manifest, manifest_path = _load_manifest(
        jsonl, args.manifest, args.allow_legacy_jsonl, args.allow_incomplete
    )
    expected_provenance = manifest.get("provenance_id") if manifest else None
    manifest_identity = manifest.get("identity", {}) if manifest else {}
    weight_source = args.weight_source
    manifest_weight_source = manifest_identity.get("weight_source") if manifest else None
    if weight_source == "auto":
        weight_source = manifest_weight_source
    elif manifest_weight_source and weight_source != manifest_weight_source:
        raise RuntimeError(
            f"evaluation weight_source={weight_source} differs from inference "
            f"weight_source={manifest_weight_source}"
        )
    if weight_source not in {"ema", "raw"}:
        raise RuntimeError("weight source is unknown; pass --weight_source ema|raw")

    expected_dictionary = (
        manifest_identity.get("dataset", {}).get("files", {}).get("dictionary", {})
        if manifest else {}
    )
    if expected_dictionary.get("sha256"):
        if sha256_file(args.dict_path) != expected_dictionary["sha256"]:
            raise RuntimeError("evaluation whole_dict differs from inference manifest")

    checkpoint_path = args.ckpt
    if checkpoint_path is None and manifest:
        checkpoint_path = manifest.get("checkpoint", {}).get("path")
    if not args.no_embedding_match and not checkpoint_path:
        raise RuntimeError("embedding match requires --ckpt or a checkpoint in manifest")
    if checkpoint_path and manifest:
        checkpoint_path = str(Path(checkpoint_path).resolve())
        expected_hash = manifest.get("checkpoint", {}).get("sha256")
        if expected_hash and sha256_file(checkpoint_path) != expected_hash:
            raise RuntimeError("evaluation checkpoint SHA256 differs from inference manifest")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_all, gen_all = Counter(), Counter()
    gt_musical, gen_musical = Counter(), Counter()
    gt_pitch, gen_pitch = Counter(), Counter()
    gt_duration, gen_duration = Counter(), Counter()
    audit_pass, audit_records = Counter(), 0
    modified_precision_values = [[] for _ in range(4)]
    bleu_values = [[] for _ in range(4)]
    exact_match_values = []
    samples_musical = []
    per_sample = []
    seen_indices = set()
    n_positions = 0
    exact_matches_total = 0
    duration_row_positions = 0
    pitch_row_positions = 0
    duration_token_violations = 0
    pitch_token_violations = 0

    with open(jsonl, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            valid_index = int(record.get("valid_index", line_number - 1))
            if valid_index in seen_indices:
                raise RuntimeError(f"duplicate valid_index={valid_index} at line {line_number}")
            seen_indices.add(valid_index)
            if not args.allow_legacy_jsonl:
                if record.get("schema_version") != INFERENCE_SCHEMA_VERSION:
                    raise RuntimeError(f"invalid schema at line {line_number}")
                if record.get("provenance_id") != expected_provenance:
                    raise RuntimeError(f"provenance mismatch at line {line_number}")

            audit = record.get("structural_audit")
            if audit is None:
                if not args.allow_legacy_jsonl:
                    raise RuntimeError(f"missing structural audit at line {line_number}")
            else:
                audit_records += 1
                for name in AUDIT_NAMES:
                    audit_pass[name] += bool(audit.get("checks", {}).get(name, False))
                if not audit.get("all_pass", False):
                    raise RuntimeError(
                        f"line {line_number} failed structural audit: "
                        f"{audit.get('failures', [])}"
                    )

            pairs = list(iter_selected_token_pairs(record))
            if record.get("generated_positions") not in (None, len(pairs)):
                raise RuntimeError(f"generated-position count mismatch at line {line_number}")
            if "generation_mask_rle" in record:
                observed_hash = canonical_sha256({"rle": record["generation_mask_rle"]})
                if record.get("generation_mask_sha256") != observed_hash:
                    raise RuntimeError(f"generation-mask hash mismatch at line {line_number}")

            reference = [item[2] for item in pairs]
            generated = [item[3] for item in pairs]
            segments = selected_token_segments(record)
            reference_segments = [segment[2] for segment in segments]
            generated_segments = [segment[3] for segment in segments]
            n_positions += len(pairs)
            gt_all.update(reference)
            gen_all.update(generated)

            reference_musical, generated_musical = [], []
            exact = 0
            for row, _, gt_token, gen_token in pairs:
                exact += gt_token == gen_token
                row_name = ROW_NAMES[row]
                if gt_token not in SPECIALS:
                    reference_musical.append(gt_token)
                    gt_musical[gt_token] += 1
                    if row_name.endswith("_pitch"):
                        gt_pitch[gt_token] += 1
                    else:
                        gt_duration[gt_token] += 1
                if gen_token not in SPECIALS:
                    generated_musical.append(gen_token)
                    gen_musical[gen_token] += 1
                    if row_name.endswith("_pitch"):
                        gen_pitch[gen_token] += 1
                    else:
                        gen_duration[gen_token] += 1
                if row_name.endswith("_duration"):
                    duration_row_positions += 1
                    duration_token_violations += (
                        gen_token not in SPECIALS and not gen_token.startswith("Duration_")
                    )
                else:
                    pitch_row_positions += 1
                    pitch_token_violations += (
                        gen_token not in SPECIALS and gen_token.startswith("Duration_")
                    )

            precisions, bleu = segmented_bleu_scores(
                reference_segments, generated_segments)
            for order in range(4):
                modified_precision_values[order].append(precisions[order])
                bleu_values[order].append(bleu[order])
            exact_matches_total += exact
            exact_match = exact / len(pairs) if pairs else float("nan")
            exact_match_values.append(exact_match)
            samples_musical.append((reference_musical, generated_musical))
            per_sample.append(
                [valid_index, len(pairs), exact_match, *precisions, *bleu]
            )

    if not seen_indices:
        raise RuntimeError("JSONL contains no samples")
    if manifest and manifest.get("status") == "completed":
        expected_count = int(manifest.get("records_written", -1))
        if expected_count != len(seen_indices):
            raise RuntimeError(
                f"manifest records_written={expected_count}, JSONL has {len(seen_indices)}"
            )
    if args.expect_valid_size:
        expected_indices = set(range(args.expect_valid_size))
        if seen_indices != expected_indices:
            missing = sorted(expected_indices - seen_indices)
            extra = sorted(seen_indices - expected_indices)
            raise RuntimeError(
                f"canonical evaluation requires valid_index=0.."
                f"{args.expect_valid_size - 1}; missing={missing[:10]}, "
                f"extra={extra[:10]}"
            )

    mean = lambda values: sum(values) / len(values) if values else float("nan")
    embedding_match: dict[str, Any] = {"enabled": False}
    if not args.no_embedding_match:
        embedding, embedding_note = load_embedding_table(checkpoint_path, weight_source)
        display_to_id, dictionary_length = build_display_to_id(args.dict_path)
        if dictionary_length + 2 != VOCAB_SIZE:
            raise RuntimeError("dictionary length does not match V2 vocabulary")
        missing = sorted(
            {
                token
                for pair in samples_musical
                for sequence in pair
                for token in sequence
                if token not in display_to_id
            }
        )
        if missing:
            raise RuntimeError(f"JSONL contains unknown display tokens: {missing[:10]}")
        device = args.device
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        embedding_match = token_embedding_greedy_match(
            samples_musical,
            display_to_id,
            embedding,
            device,
            args.max_embedding_samples,
        )
        embedding_match.update(
            {
                "enabled": True,
                "weight_source": weight_source,
                "embedding_source": embedding_note,
                "device": device,
                "definition": "greedy cosine over static tied token embeddings",
                "not_bertscore": True,
            }
        )

    coverage_all, top_all = top_k_frequency_coverage(
        gt_all, gen_all, args.top_k
    )
    coverage_musical, top_musical = top_k_frequency_coverage(
        gt_musical, gen_musical, args.top_k
    )
    pitch_gt_entropy = shannon_entropy_bits(gt_pitch)
    pitch_gen_entropy = shannon_entropy_bits(gen_pitch)
    duration_gt_entropy = shannon_entropy_bits(gt_duration)
    duration_gen_entropy = shannon_entropy_bits(gen_duration)

    summary = {
        "protocol": {
            "input_jsonl": str(jsonl),
            "manifest": str(manifest_path) if manifest else None,
            "evaluator_sha256": sha256_file(Path(__file__).resolve()),
            "provenance_id": expected_provenance,
            "weight_source": weight_source,
            "checkpoint": str(checkpoint_path) if checkpoint_path else None,
            "region": "explicit per-record DAW generation_mask_rle (G=1)",
            "aggregation": "per-sample macro average unless stated corpus-pooled",
            "bleu_boundaries": "contiguous G=1 runs within each target row",
            "n_samples": len(seen_indices),
            "n_generated_positions": n_positions,
            "is_full_validation_set": bool(args.expect_valid_size),
            "expected_valid_size": args.expect_valid_size or None,
            "valid_index_min": min(seen_indices),
            "valid_index_max": max(seen_indices),
        },
        "structural_audit": {
            "records_present": audit_records,
            "joint_all_pass_rate": audit_records / len(seen_indices),
            "per_check_pass_rate": {
                name: audit_pass[name] / audit_records if audit_records else float("nan")
                for name in AUDIT_NAMES
            },
        },
        "quality": {
            "exact_token_match_macro_per_sample": mean(exact_match_values),
            "exact_token_match_micro_per_position": (
                exact_matches_total / n_positions if n_positions else float("nan")
            ),
            "modified_ngram_precision_1_to_4": [
                mean(values) for values in modified_precision_values
            ],
            "bleu_1_to_4_cumulative": [mean(values) for values in bleu_values],
            "bleu_macro_average_1_to_4": mean(
                [mean(values) for values in bleu_values]
            ),
            "token_embedding_greedy_match": embedding_match,
        },
        "diversity_entropy_bits": {
            "definition": "token entropy by canonical V2 pitch/duration row semantics",
            "pitch_row_entropy_gt": pitch_gt_entropy,
            "pitch_row_entropy_gen": pitch_gen_entropy,
            "pitch_row_entropy_ratio_gen_over_gt": (
                pitch_gen_entropy / pitch_gt_entropy if pitch_gt_entropy else float("nan")
            ),
            "duration_row_entropy_gt": duration_gt_entropy,
            "duration_row_entropy_gen": duration_gen_entropy,
            "duration_row_entropy_ratio_gen_over_gt": (
                duration_gen_entropy / duration_gt_entropy
                if duration_gt_entropy else float("nan")
            ),
            "pitch_row_unique_gt": len(gt_pitch),
            "pitch_row_unique_gen": len(gen_pitch),
            "duration_row_unique_gt": len(gt_duration),
            "duration_row_unique_gen": len(gen_duration),
        },
        "row_type_validity": {
            "duration_row_token_violation_rate": (
                duration_token_violations / duration_row_positions
                if duration_row_positions else float("nan")
            ),
            "pitch_row_duration_token_violation_rate": (
                pitch_token_violations / pitch_row_positions
                if pitch_row_positions else float("nan")
            ),
        },
        "distribution_consistency": {
            "content_jsd_bits_all": js_divergence_bits(gt_all, gen_all),
            "content_jsd_bits_excluding_specials": js_divergence_bits(
                gt_musical, gen_musical
            ),
            f"top_{args.top_k}_frequency_coverage_all": coverage_all,
            f"top_{args.top_k}_frequency_coverage_excluding_specials": coverage_musical,
            "top_k_reference_tokens_all": top_all,
            "top_k_reference_tokens_excluding_specials": top_musical,
            "note": "frequency coverage is not retrieval Recall@k",
        },
    }

    strict_summary = _finite_json(summary)
    with open(out_dir / "objective_evaluation_summary.json", "w", encoding="utf-8") as handle:
        json.dump(strict_summary, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    with open(out_dir / "per_sample_metrics.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["valid_index", "n_positions", "exact_match"]
            + [f"modified_precision_{order}" for order in range(1, 5)]
            + [f"bleu_{order}" for order in range(1, 5)]
        )
        writer.writerows(per_sample)

    quality = summary["quality"]
    diversity = summary["diversity_entropy_bits"]
    distribution = summary["distribution_consistency"]
    lines = [
        "=" * 72,
        "Objective Evaluation — BD3LMs-Music-V2 (explicit G=1 region)",
        "=" * 72,
        f"samples/positions : {len(seen_indices)} / {n_positions}",
        f"weight source     : {weight_source}",
        f"audit all-pass    : {summary['structural_audit']['joint_all_pass_rate']:.4f}",
        "",
        "[Quality]",
        f"  exact macro/micro: {quality['exact_token_match_macro_per_sample']:.6f} / "
        f"{quality['exact_token_match_micro_per_position']:.6f}",
        "  BLEU-1/2/3/4    : "
        + " / ".join(f"{value:.6f}" for value in quality["bleu_1_to_4_cumulative"]),
        f"  BLEU macro avg  : {quality['bleu_macro_average_1_to_4']:.6f}",
        "  embedding metric: token-embedding greedy match (not BERTScore)",
        "",
        "[Diversity]",
        f"  pitch row entropy gt/gen    : {diversity['pitch_row_entropy_gt']:.6f} / "
        f"{diversity['pitch_row_entropy_gen']:.6f}",
        f"  duration row entropy gt/gen : {diversity['duration_row_entropy_gt']:.6f} / "
        f"{diversity['duration_row_entropy_gen']:.6f}",
        "",
        "[Distribution]",
        f"  JSD all/excl-specials       : {distribution['content_jsd_bits_all']:.6f} / "
        f"{distribution['content_jsd_bits_excluding_specials']:.6f}",
        f"  top-{args.top_k} frequency coverage : {coverage_all:.6f} / {coverage_musical:.6f}",
        "=" * 72,
    ]
    report = "\n".join(lines)
    (out_dir / "objective_evaluation_summary.txt").write_text(
        report + "\n", encoding="utf-8"
    )
    print(report)
    print(f"\n[objective_eval] wrote {out_dir}")


if __name__ == "__main__":
    main()
