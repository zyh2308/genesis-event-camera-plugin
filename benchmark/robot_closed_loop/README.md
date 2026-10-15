# Robot closed-loop R0 benchmark

This is downstream task code only. It does not modify `genesis_event_plugin/`,
the frozen simulator core, or the Genesis installation outside this repository's
ignored `.venv-robot/` environment.

## Runtime

```bash
bash scripts/setup_robot_runtime.sh
```

The setup pins Genesis 1.2.2 and Torch 2.8.0 CPU, then applies the tracked HDR
overlay only to `.venv-robot`. CPU physics avoids occupying the 4 GB RTX 2050;
the HDR renderer uses the headless EGL path. Runtime and asset hashes are saved
with every run.

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
  --mode slide --split holdout --episodes 10 --speeds 0.3,0.4,0.5 --a-eff 0.0981 --seed 2026101800 \
  --out benchmark/robot_closed_loop/results/r0_slide_holdout

PYOPENGL_PLATFORM=egl PYTHONPATH=. .venv-robot/bin/python -m benchmark.robot_closed_loop.run \
  --mode oracle --episodes 20 --speeds 0.3,0.4,0.5 --a-eff 0.0981 --seed 2026101900 \
  --out benchmark/robot_closed_loop/results/r0_oracle_dev

PYOPENGL_PLATFORM=egl PYTHONPATH=. .venv-robot/bin/python -m benchmark.robot_closed_loop.run \
  --mode preflight --episodes 1 --out benchmark/robot_closed_loop/results/camera_preflight

PYOPENGL_PLATFORM=egl PYTHONPATH=. .venv-robot/bin/python -m benchmark.robot_closed_loop.run \
  --mode reference --speed 0.4 --reference-duration 0.20 --seed 2026102000 \
  --out benchmark/robot_closed_loop/results/direct_hdr_pilot
```

After slide calibration, pass its single shared `a_eff` to holdout/oracle. Do
not fit it on the same holdout trajectory or use future target states in the
controller. Sensitivity examples use one factor per run:

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
