#!/usr/bin/env python3
"""Hash all non-ignored Sim-to-Real files except the index being written."""
from __future__ import annotations

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
    sim2real_root = Path(__file__).resolve().parents[1]
    repo_root = sim2real_root.parents[1]
    output = sim2real_root / "artifact_index.json"
    raw = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", "benchmark/sim2real"],
        cwd=repo_root,
    )
    paths = sorted(
        {
            (repo_root / item.decode()).resolve()
            for item in raw.split(b"\0")
            if item and (repo_root / item.decode()).resolve() != output.resolve()
        },
        key=lambda path: path.as_posix(),
    )
    entries = [
        {
            "path": path.relative_to(repo_root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in paths
        if path.is_file()
    ]
    output.write_text(json.dumps({"schema_version": 1, "artifacts": entries}, indent=2) + "\n")
    print(f"indexed {len(entries)} files -> {output.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
