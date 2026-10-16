#!/usr/bin/env python3
"""Write a SHA256/size index for one Sim-to-Real evidence run directory."""
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
    parser.add_argument("run_dir", help="Evidence run directory, relative to benchmark/sim2real or absolute")
    parser.add_argument("--include", action="append", default=[], help="Additional repo-relative artifact to index")
    parser.add_argument("--include-tree", action="append", default=[], help="Recursively index a repo-relative directory")
    args = parser.parse_args()

    sim2real_root = Path(__file__).resolve().parents[1]
    repo_root = sim2real_root.parents[1]
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = sim2real_root / run_dir
    run_dir = run_dir.resolve()
    try:
        run_dir.relative_to(sim2real_root.resolve())
    except ValueError as exc:
        raise SystemExit(f"run directory must be under {sim2real_root}") from exc

    listed = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", "benchmark/sim2real"],
        cwd=repo_root,
    )
    eligible = {(repo_root / item.decode()).resolve() for item in listed.split(b"\0") if item}
    paths = {
        p.resolve()
        for p in run_dir.rglob("*")
        if p.is_file() and p.name != "artifact_index.json" and p.resolve() in eligible
    }
    for item in args.include:
        candidate = (repo_root / item).resolve()
        if not candidate.is_file():
            raise SystemExit(f"additional artifact does not exist: {item}")
        paths.add(candidate)
    for item in args.include_tree:
        tree = (repo_root / item).resolve()
        if not tree.is_dir():
            raise SystemExit(f"additional artifact directory does not exist: {item}")
        paths.update(p for p in eligible if p.is_relative_to(tree))

    entries = []
    for path in sorted(paths, key=lambda p: p.as_posix()):
        try:
            relpath = path.relative_to(repo_root.resolve()).as_posix()
        except ValueError as exc:
            raise SystemExit(f"artifact outside repository; record it in external_artifacts.json instead: {path}") from exc
        entries.append({"path": relpath, "size_bytes": path.stat().st_size, "sha256": sha256(path)})

    output = run_dir / "artifact_index.json"
    output.write_text(json.dumps({"schema_version": 1, "artifacts": entries}, indent=2) + "\n")
    print(output.relative_to(repo_root.resolve()).as_posix())
    for entry in entries:
        print(f"{entry['sha256']}  {entry['path']} ({entry['size_bytes']} bytes)")


if __name__ == "__main__":
    main()
