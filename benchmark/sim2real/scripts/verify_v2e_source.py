#!/usr/bin/env python3
"""Verify the pinned V2E source copy and external SuperSloMo artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("repository") != "https://github.com/SensorsINI/v2e":
        raise RuntimeError("V2E source pin manifest repository mismatch")
    if manifest.get("commit") != args.expected_commit:
        raise RuntimeError(
            f"V2E manifest commit {manifest.get('commit')} != expected {args.expected_commit}"
        )
    if (args.source_root / ".git").exists():
        commit = subprocess.check_output(
            ["git", "-C", str(args.source_root), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(args.source_root), "status", "--porcelain"], text=True
        ).strip()
        if commit != args.expected_commit or dirty:
            raise RuntimeError(f"V2E git checkout mismatch: commit={commit}, dirty={bool(dirty)}")
        source_verification = "clean_git_checkout"
    else:
        rows = manifest.get("files")
        if not isinstance(rows, list) or not rows:
            raise RuntimeError("V2E file manifest is empty or malformed")
        verified_bytes = 0
        for item in rows:
            path = args.source_root / item["path"]
            if not path.is_file():
                raise FileNotFoundError(f"pinned V2E source file missing: {path}")
            actual_size = path.stat().st_size
            actual_hash = sha256(path)
            if actual_size != item["size_bytes"] or actual_hash != item["sha256"]:
                raise RuntimeError(
                    f"pinned V2E source mismatch: {path} size={actual_size}, sha256={actual_hash}"
                )
            verified_bytes += actual_size
        source_verification = "per_file_sha256_manifest"
    checkpoint_pin = manifest["slomo_checkpoint"]
    actual_checkpoint_size = args.checkpoint.stat().st_size
    actual_checkpoint_hash = sha256(args.checkpoint)
    if (actual_checkpoint_size != checkpoint_pin["size_bytes"]
            or actual_checkpoint_hash != checkpoint_pin["sha256"]):
        raise RuntimeError(
            "SuperSloMo checkpoint mismatch: "
            f"size={actual_checkpoint_size}, sha256={actual_checkpoint_hash}"
        )
    result = {
        "status": "pass",
        "repository": manifest["repository"],
        "commit": args.expected_commit,
        "source_root": str(args.source_root.resolve()),
        "source_verification": source_verification,
        "files_verified": len(manifest["files"]) if source_verification != "clean_git_checkout" else None,
        "verified_source_bytes": verified_bytes if source_verification != "clean_git_checkout" else None,
        "slomo_checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "size_bytes": actual_checkpoint_size,
            "sha256": actual_checkpoint_hash,
            "status": "pass",
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
