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

Check GPU ownership with `nvidia-smi` before launching. For EGL, set both
`EGL_DEVICE_ID` to the confirmed idle EGL device and `CUDA_VISIBLE_DEVICES` for
CUDA consumers; `CUDA_VISIBLE_DEVICES` alone does not constrain EGL device
selection. The server pilot used `EGL_DEVICE_ID=2`, verified as a V100S renderer.

## Historical R0 commands

These commands describe the implementation recorded at R0 commit `e70ba12`.
For an exact replay, check out that commit; the current runner uses the R0.5
score and is documented below.

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

The R0 camera candidate was fixed at `[1.15, 0.0, 0.22]`, looking at
`[0.65, 0.0, 0.02]` with 320×240 resolution and 30° vertical FOV. The matching
scene config records the pose and its SHA256.

For earlier closure timing, use a negative `--ttc-bias-ms`; e.g. `-20` starts
closing when the predicted crossing is about 20 ms farther away than the
configured lead threshold.

R0 development uses 20 trials/speed. Only after the configuration is fixed,
run three new-seed groups of 100 trials/speed for the held-out oracle gate.
Empirical rate and Wilson 95% interval are reported; 98/100 does not mean a
95% lower confidence bound exceeds 95%.

## Evidence

Each run contains per-episode CSV, state/control traces, selected short GIFs,
`run_manifest.json`, and `SHA256SUMS`. A reference run also writes compressed
1 kHz/2 kHz event arrays, key HDR snapshots, and sampling/runtime comparison.
The reference uses direct scene-linear HDR and the repository's
`DvsEmulator` with a shared ideal log-threshold response. It bypasses Ours'
temporal reconstruction and V2E interpolation; it is synthetic and the pixel
response implementation is shared.

The initial config is a candidate, not a frozen protocol. A formal freeze must
follow the static/contact checks, no-hand slide calibration plus holdout, and
GT-current-state oracle feasibility results.

## R0.5 closeout protocol candidate

Use `scene_pilot_r0_5_candidate.json` for the unified 0.5 ms physics timeline.
The R0 files and score remain archived. New oracle runs report both the
historical R0 score and `r0_5_strict_continuous_contact_clearance_slip`.
Strict success requires 0.5 s of uninterrupted samples with the true lowest
transformed cube vertex at least 80 mm above the plane, bilateral finger
contact, no plane contact, and no more than 10 mm change in object-to-TCP
relative position from bilateral grasp acquisition. The strict score samples
every physics step; the controller still updates at its configured 100 Hz.
The old score's first stop point is reproduced and logged separately.

Recalibrate free-slide deceleration on this same 0.5 ms configuration before
oracle evaluation:

```bash
export EGL_DEVICE_ID=2
export CUDA_VISIBLE_DEVICES=2
PYOPENGL_PLATFORM=egl PYTHONPATH=. .venv-robot/bin/python -m benchmark.robot_closed_loop.run \
  --mode slide --config benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json \
  --split calibration --episodes 20 --speeds 0.3,0.4,0.5 --seed 2026101700 \
  --out benchmark/robot_closed_loop/results/r0_5_slide_calibration
```

After fitting, update the candidate's single shared deceleration value from the
calibration summary, record the new config hash, and run strict oracle episodes
with new seeds. Use 100 trials per speed for the closeout gate:

The server R0.5 calibration currently gives `a_eff=0.09803606718731761 m/s²`
from 60 no-hand trajectories (20 per speed), all 60 crossing the intercept
plane. The source calibration config hash is
`4f58c58e3b0db367066ae126db7f3f4c603bd4c273aede33777e8381db240517`; its
results `SHA256SUMS` hash is
`f7ac352ffb2fd75475dd3c85c53f185bb188c886873b1133f6c4bb9f20939059`.

```bash
PYOPENGL_PLATFORM=egl PYTHONPATH=. .venv-robot/bin/python -m benchmark.robot_closed_loop.run \
  --mode oracle --config benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json \
  --episodes 100 --speeds 0.3,0.4,0.5 --a-eff "$A_EFF" --seed 2026102600 \
  --out benchmark/robot_closed_loop/results/r0_5_oracle_strict100x3
```

The Direct-HDR command below renders every 0.5 ms from the same physical rollout,
feeds paired 1/2 kHz DVS streams, and closes the loop using the causal 2 kHz
event state adapter. The three default speeds use center and opposite lateral
edge launch positions. Segmentation is used only by an evaluator that compares
the live target mask with a matched-pose target-only rendering; that audit is
not connected to the controller or event source. The event centroid tracker is
only a causal plumbing smoke, not an estimator quality claim.

```bash
PYOPENGL_PLATFORM=egl PYTHONPATH=. .venv-robot/bin/python -m benchmark.robot_closed_loop.run \
  --mode reference --config benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json \
  --speeds 0.3,0.4,0.5 --reference-episodes 3 \
  --reference-y-values -0.055,0,0.055,-0.055,0,0.055,-0.055,0,0.055 \
  --event-controller-hz 2000 --visibility-hz 50 --seed 2026102700 \
  --out benchmark/robot_closed_loop/results/r0_5_direct_hdr_event_smoke
```

The explicit y list gives each speed the left edge, center, and right edge in
that order, so the view/estimator check does not confound speed with lateral
position.

`reference_metrics.json` reports paired event matching, 10 ms voxel agreement,
1/2 kHz event-state errors against evaluator-only GT, dynamic target visibility,
and task rollout wall time. The summary's `episode_loop_artifact_wall_s` also
includes per-episode trace/video/event/CSV/JSON generation. The runner wrapper
must separately record the full Python process wall time, including simulator
initialization and final manifest/hash writing; that process wall time is the
end-to-end reference cost. Do not extrapolate the earlier short HDR pilot.
