#!/usr/bin/env python3
"""Verify the transferred pinned downstream source bundle against its manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    result = []
    for repo in manifest["repositories"]:
        base = args.source_root / repo["repo"]
        rows = []
        total_bytes = 0
        for item in repo["files"]:
            path = base / item["path"]
            content = path.read_bytes()
            digest = sha256(content)
            if digest != item["sha256"]:
                raise SystemExit(f"SHA-256 mismatch: {path}: {digest} != {item['sha256']}")
            if len(content) != item["size_bytes"]:
                raise SystemExit(f"size mismatch: {path}: {len(content)} != {item['size_bytes']}")
            rows.append([item["path"], len(content), digest])
            total_bytes += len(content)
        if len(rows) != repo["included_file_count"]:
            raise SystemExit(
                f"file-count mismatch for {repo['repo']}: {len(rows)} != {repo['included_file_count']}"
            )
        tree_hash = sha256(json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        if tree_hash != repo["included_tree_sha256"]:
            raise SystemExit(
                f"tree hash mismatch for {repo['repo']}: {tree_hash} != {repo['included_tree_sha256']}"
            )
        if total_bytes != repo["included_size_bytes"]:
            raise SystemExit(
                f"total-size mismatch for {repo['repo']}: {total_bytes} != {repo['included_size_bytes']}"
            )
        result.append({
            "repository": repo["repo"],
            "commit": repo["commit"],
            "files_verified": len(rows),
            "bytes_verified": total_bytes,
            "tree_sha256": tree_hash,
            "status": "pass",
        })
    print(json.dumps({"status": "pass", "repositories": result}, indent=2))


if __name__ == "__main__":
    main()
