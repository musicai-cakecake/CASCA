#!/usr/bin/env python3
"""Historical pre-integration Phase 0 audit, not a current-release acceptance test.

The byte-identical base/focal assertion intentionally applies before focal
integration; later-stage source trees are expected to differ.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
V2_ROOT = REPO.parent / "CASCA_base"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from xmidi_getmusic.phase0_prior import sha256_file, write_hashed_json  # noqa: E402
from xmidi_getmusic.runtime_requirements import require_project_path  # noqa: E402

CORE_FILES = (
    "diffusion.py",
    "models/dit.py",
    "noise_schedule.py",
    "dataloader.py",
    "xmidi_getmusic/constants.py",
    "xmidi_getmusic/region.py",
    "xmidi_getmusic/dataset.py",
    "xmidi_getmusic/eval_protocol.py",
)
TEXT_SUFFIXES = {".py", ".sh", ".yaml", ".yml"}
OLD_ABSOLUTE = re.compile(
    r"/(?:apdcephfs_[^/\s]+|share/kuleshov|opt/conda)/"
)
OLD_WANDB = re.compile(
    r"(?:wandb\.project\s*[:=]\s*|WANDB_PROJECT=)(?:BD3-LMs|BD3LMs-Music-V2)(?:\s|$)"
)
READ_ONLY_REFERENCE_FILES = {
    Path("scripts/phase0/run_one_step_regression.sh"),
}


def main() -> None:
    dangerous: list[dict[str, object]] = []
    explicit_read_only_references: list[dict[str, object]] = []
    old_wandb: list[dict[str, object]] = []
    scanned = 0
    for path in sorted(REPO.rglob("*")):
        if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
            continue
        relative = path.relative_to(REPO)
        if relative.parts[0] in {"outputs", "artifacts", ".git", "__pycache__"}:
            continue
        if relative == Path("scripts/phase0/audit_project_boundaries.py"):
            continue
        scanned += 1
        text = path.read_text(encoding="utf-8", errors="replace")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if OLD_ABSOLUTE.search(line):
                match = {
                    "path": str(relative),
                    "line": line_number,
                    "text": line.strip(),
                }
                if relative in READ_ONLY_REFERENCE_FILES:
                    explicit_read_only_references.append(match)
                else:
                    dangerous.append(match)
            if OLD_WANDB.search(line):
                old_wandb.append({
                    "path": str(relative),
                    "line": line_number,
                    "text": line.strip(),
                })
    symlinks = [
        {"path": str(path.relative_to(REPO)), "target": str(path.readlink())}
        for path in REPO.rglob("*")
        if path.is_symlink()
        and not any(part in {"outputs", "artifacts", ".git"} for part in path.relative_to(REPO).parts)
    ]
    core = {}
    for relative in CORE_FILES:
        v2_path = V2_ROOT / relative
        focal_path = REPO / relative
        v2_sha = sha256_file(v2_path)
        focal_sha = sha256_file(focal_path)
        core[relative] = {
            "v2_sha256": v2_sha,
            "focal_sha256": focal_sha,
            "byte_identical": v2_sha == focal_sha,
        }
    config_text = (REPO / "configs" / "config.yaml").read_text(encoding="utf-8")
    data_text = (REPO / "configs" / "data" / "xmidi_getmusic.yaml").read_text(encoding="utf-8")
    setup_text = (REPO / "setup_env.sh").read_text(encoding="utf-8")
    environment_text = (REPO / "environment.yml").read_text(encoding="utf-8")
    checks = {
        "no_old_absolute_runtime_paths": not dangerous,
        "no_old_wandb_project_names": not old_wandb,
        "no_runtime_symlinks": not symlinks,
        "core_v2_files_byte_identical": all(item["byte_identical"] for item in core.values()),
        "data_root_requires_explicit_environment": "${oc.env:BD3LM_DATA_ROOT}" in data_text,
        "wandb_project_configurable": "${oc.env:WANDB_PROJECT,CASCA_focal}" in config_text,
        "conda_constraint_configurable": "BD3LM_EXPECTED_CONDA_ENV=${BD3LM_EXPECTED_CONDA_ENV:-}" in setup_text,
        "tmux_constraint_configurable": "BD3LM_EXPECTED_TMUX=${BD3LM_EXPECTED_TMUX:-}" in setup_text,
        "wandb_identity_default": "WANDB_PROJECT=${WANDB_PROJECT:-CASCA_focal}" in setup_text,
        "environment_export_name_exact": environment_text.startswith("name: casca\n"),
        "environment_export_python312": "  - python=3.12" in environment_text,
        "environment_uses_shared_requirements": "-r requirements.txt" in environment_text,
    }
    payload = {
        "schema_version": "bd3lms-music-v2-focal.phase0-path-audit.v1",
        "project": "BD3LMs_Music_V2_focal_loss",
        "project_root": str(REPO),
        "v2_reference_root": str(V2_ROOT),
        "scanned_source_config_files": scanned,
        "dangerous_old_absolute_runtime_references": dangerous,
        "explicit_phase0_read_only_references": explicit_read_only_references,
        "old_wandb_project_references": old_wandb,
        "runtime_symlinks": symlinks,
        "core_v2_source_identity": core,
        "checks": checks,
        "passed": all(checks.values()),
        "policy": {
            "authoritative V2 source, data, and checkpoint inputs": "read-only explicit references are allowed",
            "runtime outputs": "must remain below the focal-loss project root",
            "data": "local project copy, content-verified against V2 by SHA-256; mtime ignored",
        },
    }
    out = require_project_path(
        REPO / "artifacts" / "phase0" / "project_boundary_audit.json",
        label="project boundary audit",
    )
    write_hashed_json(out, payload)
    print(json.dumps({"out": str(out), "checks": checks, "passed": payload["passed"]}, indent=2))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
