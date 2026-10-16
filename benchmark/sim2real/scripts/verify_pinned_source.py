#!/usr/bin/env python3
"""Verify a git checkout or the project-pinned source bundle copy."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_pinned_source(
    source_root: Path,
    repository_name: str,
    expected_commit: str,
    bundle_manifest: Path,
) -> dict:
    source_root = source_root.resolve()
    if (source_root / ".git").exists():
        commit = subprocess.check_output(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
        ).strip()
        if commit != expected_commit:
            raise RuntimeError(
                f"{repository_name} source HEAD must be {expected_commit}, got {commit}"
            )
        dirty = subprocess.check_output(
            ["git", "-C", str(source_root), "status", "--porcelain"], text=True
        )
        if dirty.strip():
            raise RuntimeError(f"{repository_name} git checkout has local modifications")
        return {"repository": repository_name, "commit": commit, "verification": "clean_git_checkout"}

    manifest = json.loads(bundle_manifest.read_text())
    matches = [row for row in manifest["repositories"] if row["repo"] == repository_name]
    if len(matches) != 1:
        raise ValueError(f"bundle manifest must contain exactly one {repository_name!r} record")
    record = matches[0]
    if record["commit"] != expected_commit:
        raise RuntimeError(
            f"bundle pin mismatch for {repository_name}: {record['commit']} != {expected_commit}"
        )
    rows = []
    total_bytes = 0
    for item in record["files"]:
        path = source_root / item["path"]
        content = path.read_bytes()
        digest = sha256_bytes(content)
        if len(content) != item["size_bytes"] or digest != item["sha256"]:
            raise RuntimeError(f"{repository_name} pinned source file mismatch: {path}")
        rows.append([item["path"], len(content), digest])
        total_bytes += len(content)
    tree_hash = sha256_bytes(
        json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
    if (
        len(rows) != record["included_file_count"]
        or total_bytes != record["included_size_bytes"]
        or tree_hash != record["included_tree_sha256"]
    ):
        raise RuntimeError(f"{repository_name} pinned source tree hash/count/size mismatch")
    return {
        "repository": repository_name,
        "commit": expected_commit,
        "verification": "manifest_file_sha256_and_tree_sha256",
        "files_verified": len(rows),
        "bytes_verified": total_bytes,
        "tree_sha256": tree_hash,
    }
