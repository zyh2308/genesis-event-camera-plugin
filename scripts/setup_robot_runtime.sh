#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if command -v uv >/dev/null 2>&1 && command -v python3.10 >/dev/null 2>&1; then
  if [[ ! -x .venv-robot/bin/python ]]; then
    uv venv --python "$(command -v python3.10)" .venv-robot
  fi
  uv pip install --python .venv-robot/bin/python 'genesis-world==1.2.2'
  # This path is for the local development workstation. Server runs should
  # inherit the already provisioned Eventbased_WAM dependencies instead.
  uv pip install --python .venv-robot/bin/python 'torch==2.8.0+cpu' --index-url https://download.pytorch.org/whl/cpu
else
  # The shared research server has Python 3.11 and a provisioned Genesis/Torch
  # stack but no uv/Python 3.10. Keep that environment untouched: create a
  # project venv with system dependencies visible, then copy its pinned Genesis
  # package into private site-packages so the HDR overlay cannot alter the
  # shared conda installation or depend on a server-side package download.
  runtime_python="${ROBOT_PYTHON:-python3}"
  if ! command -v "$runtime_python" >/dev/null 2>&1; then
    echo "Set ROBOT_PYTHON to an available Python executable" >&2
    exit 2
  fi
  if [[ ! -x .venv-robot/bin/python ]]; then
    "$runtime_python" -m venv --system-site-packages .venv-robot
  fi
  site_packages="$(.venv-robot/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  "$runtime_python" -c 'import genesis, importlib.metadata as m; assert genesis.__version__ == "1.2.2"; assert m.version("genesis-world") == "1.2.2"' || {
    echo "The activated environment must provide genesis-world==1.2.2" >&2
    exit 2
  }
  private_genesis="$(.venv-robot/bin/python -c 'import pathlib, genesis; print(pathlib.Path(genesis.__file__).resolve().parent)' 2>/dev/null || true)"
  case "$private_genesis" in
    "$repo_root"/.venv-robot/*)
      ;;
    *)
      genesis_source="$("$runtime_python" -c 'import pathlib, genesis; print(pathlib.Path(genesis.__file__).resolve().parent)')"
      genesis_metadata="$("$runtime_python" -c 'import importlib.metadata as m; print(m.distribution("genesis-world")._path)')"
      mkdir -p "$site_packages/genesis"
      "$runtime_python" - "$genesis_source" "$site_packages/genesis" <<'PY'
import shutil
import sys

shutil.copytree(
    sys.argv[1], sys.argv[2], dirs_exist_ok=True,
    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
)
PY
      cp -a "$genesis_metadata" "$site_packages/"
      ;;
  esac
  torch_version="$(.venv-robot/bin/python -c 'import torch; print(torch.__version__)' 2>/dev/null || true)"
  case "$torch_version" in
    2.8.*)
      ;;
    *)
      if [[ -n "${ROBOT_TORCH_WHEEL:-}" ]]; then
        if [[ ! -f "$ROBOT_TORCH_WHEEL" ]]; then
          echo "ROBOT_TORCH_WHEEL does not exist: $ROBOT_TORCH_WHEEL" >&2
          exit 2
        fi
        .venv-robot/bin/python -m pip install --no-deps --target "$site_packages" "$ROBOT_TORCH_WHEEL"
      else
        .venv-robot/bin/python -m pip install --no-deps --target "$site_packages" \
          --index-url https://download.pytorch.org/whl/cpu 'torch==2.8.0+cpu'
      fi
      ;;
  esac
fi
.venv-robot/bin/python scripts/apply_robot_hdr_overlay.py
PYOPENGL_PLATFORM=egl .venv-robot/bin/python - <<'PY'
import inspect
import genesis as gs
import torch
from genesis.vis.camera import Camera
assert gs.__version__ == "1.2.2"
assert torch.__version__.startswith("2.8."), torch.__version__
assert "radiance" in inspect.signature(Camera.render).parameters
assert "radiance" in inspect.getsource(Camera.render)
print("robot runtime ready:", gs.__version__, gs.__file__)
PY
