#!/usr/bin/env python3
"""Build an isolated Genesis 1.2.2 runtime with the repository HDR overlay.

Run this on the research server inside the pinned Eventbased_WAM environment.
The conda installation is read-only input; all copied and patched files live
under the project-owned runtime directory.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--expected-version", default="1.2.2")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    overlay_root = repo_root / "integrations/genesis_renderer_hdr_v2_overlay/genesis"
    spec = importlib.util.find_spec("genesis")
    if spec is None or spec.origin is None:
        raise SystemExit("Genesis is not importable in the active environment")
    source_package = Path(spec.origin).resolve().parent
    version = importlib.metadata.version("genesis-world")
    if version != args.expected_version:
        raise SystemExit(f"expected genesis-world=={args.expected_version}, found {version}")

    runtime_root = args.runtime_root.expanduser().resolve()
    runtime_package = runtime_root / "genesis"
    runtime_root.mkdir(parents=True, exist_ok=True)
    ignored = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(source_package, runtime_package, dirs_exist_ok=True, ignore=ignored)
    shutil.copytree(overlay_root, runtime_package, dirs_exist_ok=True, ignore=ignored)

    overlay_files = sorted(p for p in overlay_root.rglob("*") if p.is_file())
    entries = []
    for source in overlay_files:
        relative = source.relative_to(overlay_root)
        target = runtime_package / relative
        if not target.is_file() or sha256(source) != sha256(target):
            raise SystemExit(f"overlay verification failed: {relative}")
        entries.append({"relative_path": relative.as_posix(), "sha256": sha256(target), "size_bytes": target.stat().st_size})

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "genesis_world_version": version,
        "source_package": str(source_package),
        "runtime_package": str(runtime_package),
        "overlay_source": str(overlay_root),
        "cache_policy": "exclude __pycache__ and *.pyc; retain all package/runtime source files",
        "overlay_files": entries,
    }
    manifest_path = runtime_root / "runtime_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(manifest_path)
    print(f"overlay files verified: {len(entries)}")


if __name__ == "__main__":
    main()
