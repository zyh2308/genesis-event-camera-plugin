#!/usr/bin/env python3
"""Apply the tracked HDR overlay to an isolated Genesis v1.2.2 runtime."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import genesis as gs


ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / "integrations/genesis_renderer_hdr_v2_overlay/genesis"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> None:
    if getattr(gs, "__version__", None) != "1.2.2":
        raise SystemExit(f"Expected Genesis 1.2.2, found {getattr(gs, '__version__', None)}")
    destination = Path(gs.__file__).resolve().parent
    files = sorted(
        p for p in OVERLAY.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    )
    if not files:
        raise SystemExit(f"No overlay files found under {OVERLAY}")
    copied = {}
    for source in files:
        relative = source.relative_to(OVERLAY)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied[str(relative)] = {"sha256": digest(source), "runtime_path": str(target)}
    manifest = {
        "genesis_version": gs.__version__,
        "genesis_runtime_root": str(destination),
        "overlay_source": str(OVERLAY),
        "files": copied,
        "runtime_isolated_in_project_venv": ".venv-robot" in str(destination),
    }
    out = destination.parent / "robot_hdr_overlay_manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not manifest["runtime_isolated_in_project_venv"]:
        raise SystemExit(f"Refusing to patch a non-project-venv Genesis install: {destination}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
