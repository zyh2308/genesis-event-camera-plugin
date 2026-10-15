#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required to create the isolated robot runtime" >&2
  exit 2
fi
if ! command -v python3.10 >/dev/null 2>&1; then
  echo "Python 3.10 is required for the Genesis 1.2.2 pilot runtime" >&2
  exit 2
fi

if [[ ! -x .venv-robot/bin/python ]]; then
  uv venv --python "$(command -v python3.10)" .venv-robot
fi
uv pip install --python .venv-robot/bin/python 'genesis-world==1.2.2'
# Genesis 1.2.2 warns below Torch 2.8. This CPU wheel keeps the R0 scene off
# the shared 4 GB GPU while using the version range tested for this workflow.
uv pip install --python .venv-robot/bin/python 'torch==2.8.0+cpu' --index-url https://download.pytorch.org/whl/cpu
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
