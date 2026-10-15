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
inherits its provisioned dependencies, copies the existing Genesis 1.2.2
package, and installs Torch 2.8 CPU into private site-packages. This avoids
changing the shared conda environment. Torch can come from the official PyTorch
CPU index, or be staged with `ROBOT_TORCH_WHEEL=/path/to/wheel`. The scene uses
CPU physics and headless EGL rendering. Runtime and asset hashes are saved with
every run.

Server setup and run from the isolated worktree:

```bash
source ~/miniconda3/bin/activate Eventbased_WAM
cd ~/Eventbased_WAM/repositories/gec-robot
ROBOT_TORCH_WHEEL="$HOME/Eventbased_WAM/torch-2.8.0+cpu-cp311-cp311-manylinux_2_28_x86_64.whl" bash scripts/setup_robot_runtime.sh
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

The oracle controller uses only the delayed observation and fixed motion model.
After the gripper starts closing, closure stays latched even if a later
prediction says the target has stopped before the intercept plane. Lifting
requires both a sustained gripper stall and a current propagated target estimate
inside the configured x/y gate and below its planar-speed threshold. Contact
labels are reserved for evaluation and traces; they do not trigger control.
The candidate closure lead is computed once from the first propagated current
velocity estimate using the configured linear schedule, then held for the
episode. `--ttc-bias-ms` remains an additive timing perturbation for sensitivity
runs; it does not select a speed-specific lead manually.

Slide calibration uses a target-only Genesis scene containing the same Plane
and cube material/geometry as the robot task; Panda is absent so the hand cannot
alter the measured free-slide deceleration. Its run manifest records
`panda_asset_in_scene: false`.

Sensitivity examples use one factor per run:

```bash
... --mode sensitivity --episodes 5 --speeds 0.4 --a-eff <calibrated-a> --bias-y 0.005 ...
... --mode sensitivity --episodes 5 --speeds 0.4 --a-eff <calibrated-a> --ttc-bias-ms 20 ...
... --mode sensitivity --episodes 5 --speeds 0.4 --a-eff <calibrated-a> --extra-latency-ms 20 ...
```

`--mode preflight` measures target visibility against a target-only reference
scene rendered with the same camera and target geometry. It also samples the
nominal launch-to-intercept view at the configured lateral range while holding
Panda at its legal pre-grasp home pose. Those are static camera checks, not
dynamic arm-trajectory coverage.

For earlier closure timing, use a negative `--ttc-bias-ms`; e.g. `-20` starts
closing when the predicted crossing is about 20 ms farther away than the
configured lead threshold.

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
