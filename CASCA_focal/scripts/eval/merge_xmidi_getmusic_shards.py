"""Merge canonical BD3LMs-Music-V2 inference shards and rebuild provenance."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from xmidi_getmusic.eval_protocol import (
    MANIFEST_SCHEMA_VERSION,
    atomic_write_json,
    canonical_sha256,
    validate_manifest_provenance,
)


def _atomic_merge_jsonl(destination: Path, shard_paths: list[Path]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=destination.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        for shard_path in shard_paths:
            with open(shard_path, encoding="utf-8") as shard:
                for line in shard:
                    handle.write(line)
    os.replace(temporary, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--expected_size", type=int, default=6817)
    args = parser.parse_args()

    shards_dir = Path(args.shards_dir).resolve()
    shard_paths = sorted(shards_dir.glob("shard_*.jsonl"))
    if not shard_paths:
        raise RuntimeError(f"no shard_*.jsonl files found in {shards_dir}")

    manifests = []
    all_records: dict[int, dict] = {}
    for shard_path in shard_paths:
        manifest_path = Path(str(shard_path) + ".manifest.json")
        if not manifest_path.is_file():
            raise RuntimeError(f"missing shard manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_manifest_provenance(manifest)
        if manifest.get("status") != "completed":
            raise RuntimeError(
                f"shard {shard_path.name} has status={manifest.get('status')!r}"
            )
        manifests.append(manifest)

        shard_count = 0
        with open(shard_path, encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("provenance_id") != manifest["provenance_id"]:
                    raise RuntimeError(
                        f"record provenance mismatch: {shard_path}:{line_number}"
                    )
                valid_index = int(record["valid_index"])
                if valid_index in all_records:
                    raise RuntimeError(f"duplicate valid_index={valid_index}")
                all_records[valid_index] = record
                shard_count += 1
        if shard_count != int(manifest.get("records_written", -1)):
            raise RuntimeError(
                f"{shard_path.name}: records_written={manifest.get('records_written')}, "
                f"JSONL count={shard_count}"
            )

    expected_indices = set(range(args.expected_size))
    observed_indices = set(all_records)
    if observed_indices != expected_indices:
        missing = sorted(expected_indices - observed_indices)
        extra = sorted(observed_indices - expected_indices)
        raise RuntimeError(
            f"shards do not cover valid_index 0..{args.expected_size - 1}; "
            f"missing={missing[:20]}, extra={extra[:20]}"
        )

    identities = [manifest["identity"] for manifest in manifests]
    common_keys = (
        "schema_version",
        "checkpoint_sha256",
        "checkpoint_global_step",
        "weight_source",
        "model",
        "data_split",
        "data_size",
        "dataset",
        "inference_code",
        "resolved_config_sha256",
        "token_encoding",
        "base_seed",
        "sampling",
        "protocol_constants",
    )
    common_identity = {key: identities[0][key] for key in common_keys}
    for shard_index, identity in enumerate(identities[1:], start=1):
        for key in common_keys:
            if identity.get(key) != common_identity[key]:
                raise RuntimeError(
                    f"shard {shard_index} identity differs at key {key!r}"
                )

    output = Path(args.out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=output.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        for valid_index in range(args.expected_size):
            handle.write(json.dumps(all_records[valid_index], ensure_ascii=False) + "\n")
    os.replace(temporary, output)

    merged_identity = dict(common_identity)
    merged_identity["merge"] = {
        "expected_valid_size": args.expected_size,
        "shard_provenance_ids": [m["provenance_id"] for m in manifests],
        "shard_ranges": [m.get("requested_range") for m in manifests],
    }
    merged_provenance = canonical_sha256(merged_identity)

    # Records were generated under shard-level provenance. Preserve it and add
    # merged provenance without rewriting semantic fields.
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=output.parent, delete=False
    ) as handle:
        rewritten = Path(handle.name)
        with open(output, encoding="utf-8") as source:
            for line in source:
                record = json.loads(line)
                record["shard_provenance_id"] = record["provenance_id"]
                record["provenance_id"] = merged_provenance
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(rewritten, output)

    manifest = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "inference_schema_version": common_identity["schema_version"],
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_jsonl": str(output),
        "provenance_id": merged_provenance,
        "identity": merged_identity,
        "checkpoint": manifests[0]["checkpoint"],
        "requested_range": [0, args.expected_size],
        "records_written": args.expected_size,
        "shards": [
            {
                "jsonl": str(path),
                "provenance_id": manifest["provenance_id"],
                "requested_range": manifest.get("requested_range"),
                "records_written": manifest.get("records_written"),
            }
            for path, manifest in zip(shard_paths, manifests)
        ],
        "total_diffusion_nfes": sum(
            int(m.get("total_diffusion_nfes_this_invocation", 0))
            for m in manifests
        ),
        "total_backbone_forward_calls": sum(
            int(m.get("total_backbone_forward_calls_this_invocation", 0))
            for m in manifests
        ),
    }
    atomic_write_json(str(output) + ".manifest.json", manifest)
    print(f"[merge] shards   = {len(shard_paths)}")
    print(f"[merge] records  = {args.expected_size}")
    print(f"[merge] output   = {output}")
    print(f"[merge] manifest = {output}.manifest.json")


if __name__ == "__main__":
    main()
