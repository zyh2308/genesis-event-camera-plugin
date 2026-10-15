# Robot closed-loop R0 benchmark

This is downstream task code only. It does not modify `genesis_event_plugin/`,
the frozen simulator core, or the Genesis installation outside this repository's
ignored `.venv-robot/` environment.

## Runtime

```bash
bash scripts/setup_robot_runtime.sh
```

The setup pins Genesis 1.2.2 and applies the tracked HDR overlay only to
`.venv-robot`. On the local development workstation it installs Torch 2.8 CPU.
On the research server, activate `Eventbased_WAM` first; the project venv
inherits its provisioned Torch/dependencies and installs Genesis into private
site-packages. This avoids changing the shared conda environment. The scene
uses CPU physics and headless EGL rendering. Runtime and asset hashes are saved
with every run.

Server setup and run from the isolated worktree:

```bash
source ~/miniconda3/bin/activate Eventbased_WAM
cd ~/Eventbased_WAM/repositories/gec-robot
bash scripts/setup_robot_runtime.sh
CUDA_VISIBLE_DEVICES=1 PYOPENGL_PLATFORM=egl PYTHONPATH=. .venv-robot/bin/python \
  -m benchmark.robot_closed_loop.run --mode static --episodes 1 --seed 2026101600 \
  --out benchmark/robot_closed_loop/results/server_static_smoke
```

Select a currently idle GPU for EGL explicitly after checking ownership with
`nvidia-smi`; `CUDA_VISIBLE_DEVICES=1` is only a server-specific example.

## R0 commands

Use a new output directory for every run. The runner refuses to mix results in
an existing non-empty directory.

```bash
PYOPENGL_PLATFORM=egl PYTHONPATH=. .venv-robot/bin/python -m benchmark.robot_closed_loop.run \
  --mode static --episodes 20 --seed 2026101600 \
  --out benchmark/robot_closed_loop/results/r0_static_dev

PYOPENGL_PLATFORM=egl PYTHONPATH=. .venv-robot/bin/python -m benchmark.robot_closed_loop.run \
  --mode slide --split calibration --episodes 10 --speeds 0.3,0.4,0.5 --seed 2026101700 \
  --out benchmark/robot_closed_loop/results/r0_slide_calibration

PYOPENGL_PLATFORM=egl PYTHONPATH=. .venv-robot/bin/python -m benchmark.robot_closed_loop.run \
  --mode slide --split holdout --episodes 10 --speeds 0.3,0.4,0.5 --a-eff "$A_EFF" --seed 2026101800 \
  --out benchmark/robot_closed_loop/results/r0_slide_holdout

PYOPENGL_PLATFORM=egl PYTHONPATH=. .venv-robot/bin/python -m benchmark.robot_closed_loop.run \
  --mode oracle --episodes 20 --speeds 0.3,0.4,0.5 --a-eff "$A_EFF" --seed 2026101900 \
  --out benchmark/robot_closed_loop/results/r0_oracle_dev

PYOPENGL_PLATFORM=egl PYTHONPATH=. .venv-robot/bin/python -m benchmark.robot_closed_loop.run \
  --mode preflight --episodes 1 --out benchmark/robot_closed_loop/results/camera_preflight

PYOPENGL_PLATFORM=egl PYTHONPATH=. .venv-robot/bin/python -m benchmark.robot_closed_loop.run \
  --mode reference --speed 0.4 --reference-duration 0.20 --seed 2026102000 \
  --out benchmark/robot_closed_loop/results/direct_hdr_pilot
```

After slide calibration, extract its single shared `a_eff` for holdout/oracle.
Do not fit it on the same holdout trajectory or use future target states in the
controller. For the paths above:

```bash
A_EFF="$(.venv-robot/bin/python -c 'import json; print(json.load(open("benchmark/robot_closed_loop/results/r0_slide_calibration/r0_slide_calibration.json"))["a_eff_m_s2"])')"
```

Sensitivity examples use one factor per run:

```bash
... --mode sensitivity --episodes 5 --speeds 0.4 --a-eff <calibrated-a> --bias-y 0.005 ...
... --mode sensitivity --episodes 5 --speeds 0.4 --a-eff <calibrated-a> --ttc-bias-ms 20 ...
... --mode sensitivity --episodes 5 --speeds 0.4 --a-eff <calibrated-a> --extra-latency-ms 20 ...
```

R0 development uses 20 trials/speed. Only after the configuration is fixed,
run three new-seed groups of 100 trials/speed for the held-out oracle gate.
Empirical rate and Wilson 95% interval are reported; 98/100 does not mean a
95% lower confidence bound exceeds 95%.

## Evidence

Each run contains per-episode CSV, selected 10 ms traces, selected short GIFs,
`run_manifest.json`, and `SHA256SUMS`. A reference run also writes compressed
1 kHz/2 kHz event arrays, sparse HDR snapshots, and sampling/runtime comparison.
The reference uses direct scene-linear HDR and the repository's
`DvsEmulator` with a shared ideal log-threshold response. It bypasses Ours'
temporal reconstruction and V2E interpolation; it is still synthetic and the
pixel response implementation is shared.

The initial config is a candidate, not a frozen protocol. A formal freeze must
follow the static/contact checks, no-hand slide calibration plus holdout, and
GT-current-state oracle feasibility results.
