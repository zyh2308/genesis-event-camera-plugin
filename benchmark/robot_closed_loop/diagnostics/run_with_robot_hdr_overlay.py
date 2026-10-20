"""Run a benchmark script with the installed robot HDR Genesis overlay.

This bridge lets a diagnostic process use a different user-level PyTorch
runtime (for example CUDA Torch from the Eventbased_WAM environment) while
loading the already-installed HDR overlay package code from ``.venv-robot``.
It does not mutate either Python environment.
"""

from __future__ import annotations

import importlib.util
import os
import runpy
import sys
from pathlib import Path


def main() -> None:
    genesis_dir = Path(os.environ["ROBOT_GENESIS_OVERLAY_PACKAGE"]).resolve()
    entry = genesis_dir / "__init__.py"
    if not entry.is_file():
        raise FileNotFoundError(f"HDR overlay Genesis package is missing: {entry}")
    spec = importlib.util.spec_from_file_location(
        "genesis", entry, submodule_search_locations=[str(genesis_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load HDR overlay Genesis package from {entry}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["genesis"] = module
    spec.loader.exec_module(module)
    benchmark_script = Path(sys.argv[1]).resolve()
    sys.argv = [str(benchmark_script), *sys.argv[2:]]
    runpy.run_path(str(benchmark_script), run_name="__main__")


if __name__ == "__main__":
    main()
