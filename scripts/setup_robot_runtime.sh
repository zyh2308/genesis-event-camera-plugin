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
  genesis_source="$("$runtime_python" -c 'import pathlib, genesis; print(pathlib.Path(genesis.__file__).resolve().parent)')"
  genesis_metadata="$("$runtime_python" -c 'import importlib.metadata as m; print(m.distribution("genesis-world")._path)')"
  mkdir -p "$site_packages/genesis"
  cp -a "$genesis_source"/. "$site_packages/genesis/"
  cp -a "$genesis_metadata" "$site_packages/"
fi
.venv-robot/bin/python scripts/apply_robot_hdr_overlay.py
PYOPENGL_PLATFORM=egl .venv-robot/bin/python - <<'PY'
import inspect
import genesis as gs
from genesis.vis.camera import Camera
assert gs.__version__ == "1.2.2"
assert "radiance" in inspect.signature(Camera.render).parameters
assert "radiance" in inspect.getsource(Camera.render)
print("robot runtime ready:", gs.__version__, gs.__file__)
PY
