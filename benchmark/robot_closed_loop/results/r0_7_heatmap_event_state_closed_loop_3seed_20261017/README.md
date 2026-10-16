# R0.7 heatmap learned-event new-episode rollout

Small Direct-HDR closed-loop development check. This is not a generator comparison, a held-out performance estimate, or a frozen model result.

## Provenance

- Run code commit: `12820e556a6157c0f0d18e75ec7cc9c00826d06a`.
- Heatmap training code commit: `1b2c80ed4ef80d8a3bcb3bc8679de1298659a49b`.
- Candidate config: `benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json`, SHA256 `aa97f6f5a12af92561ef053407bb795bba2801582958b4ad1134bcd24497435a`.
- Checkpoint: `../r0_7_heatmap_event_state_development_20261017/spatial_event_state.pt`, SHA256 `64b120afde4579563d8c18f4430eeaab00c3829adaa682fa82f321fac496ea7a`.
- Source development manifest SHA256: `86cf6c0801365c0c1e5949e4e04d3178fa03f334997db6d09547f69a2b5af37c`.
- Frozen simulator core: `e79e656ac00f7847b98db9e12bf43b093f3afff1`; task and estimator code are in the robot benchmark layer.
- Physics step `0.5 ms`; causal five-bin input spans `[cutoff-25 ms, cutoff)`; event controller `2 kHz`.

The nine original episodes were split by episode, not by window. This rollout uses new seeds `2026103900/01/02`. The `0.3 m/s, y=0` condition was present in the fit split; `0.4 m/s, y=+55 mm` and `0.5 m/s, y=-55 mm` were validation cells. Therefore all three are new episodes, but only two cells are unseen during fitting. The nine source trajectories remain development data.

## Strict outcomes

| Speed / launch y | Seed | Close start | First contact / bilateral | Relative speed at contact | Lift / qualified hold | Strict result |
|---|---:|---:|---:|---:|---:|---|
| 0.30 m/s / 0 mm | 2026103900 | 0.860 s | 0.9025 / 0.9025 s | 0.193 m/s | no lift / 0 s | fail: `unstable_grasp` |
| 0.40 m/s / +55 mm | 2026103901 | 0.610 s | 0.626 / 0.626 s | 0.303 m/s | 0.694 s / 0.5 s | pass |
| 0.50 m/s / -55 mm | 2026103902 | 0.490 s | 0.504 / 0.507 s | 0.416 m/s | 0.5755 s / 0.5 s | pass |

The two strict passes had minimum true bottom clearance during qualified hold of `80.085 mm` and `80.110 mm`; maximum relative slip was `4.951 mm` and `3.091 mm` against the configured `10 mm` limit. The failed episode achieved only transient bilateral contact, with no qualified hold. Contact-to-commit has one fresh poll in the failed episode; in both successful episodes the terminal commit preceded first contact, by `16 ms` and `4 ms` respectively. Close-to-contact polls were fresh in all three runs.

## Causal state and prediction metrics

Evaluator-only rescore compares each logged event estimate and the fixed predictor against GT at the same causal cutoff. GT does not enter inference or control.

| Phase | Polls | Fresh states | Input-age P95 | XY position-error P95 | XY velocity-error P95 | Arrival-error P95 | Intercept-Y-error P95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Before contact | 190 | 190 (100%) | 10 ms | 22.9 mm | 0.212 m/s | 351.7 ms | 13.5 mm |
| Contact to lift or episode end | 223 | 89 (39.9%) | 10 ms | 178.0 mm | 0.121 m/s | 122.3 ms | 19.9 mm |
| After lift | 135 | 135 (100%) | 10 ms | 101.1 mm | 0.054 m/s | 410.5 ms | 4.5 mm |

The contact-to-lift/episode-end status counts were `89 fresh`, `25 tracking_lost`, and `109 no_event`; the low availability is concentrated in the failed `0.3 m/s` rollout. Freshness/age are availability metadata, not an accuracy guarantee. At pre-contact, per-episode XY P95 was `24.8/20.4/19.1 mm`; arrival-error P95 was `221/317/474 ms` for the three seeds. The model improves spatial position over the earlier v1 fit on its matching validation split, but arrival timing remains too inaccurate for a generator-comparison entry criterion.

## Reference-path checks and cost

- Same underlying `0.5 ms` physical trajectory was sampled at `1/2 kHz`; the two event streams use the shared `genesis_event_plugin.DvsEmulator`, threshold `0.2`, no noise, infinite bandwidth and ideal readout. This is an uncalibrated synthetic pixel response, not an EVK4 model. Ours temporal reconstruction and V2E frame interpolation are bypassed.
- Per-episode 1/2 kHz event-count differences were `11.46/5.21/5.17%`; matched 10 ms `(2,64,64)` voxel cosine was `0.981/0.991/0.991`. This is a small same-trajectory sampling check, not a full sensor validation.
- Fixed-camera target visibility at 50 Hz had 282 evaluator-only samples, median visible fraction `1.0`, minimum `0.9895`; before predicted arrival it was `1.0/0.995/0.990` for the three trajectories. Target boxes remained at least `31×37 px` in the 320×240 render.
- Each physical step made a Direct-HDR render; steady render mean/P95 was `3.08–3.20/3.51–3.56 ms`. 2 kHz DVS processing mean was `7.22–14.69 ms/sample`; model-only inference on fresh states mean/P95 was `4.74–4.92/4.88–5.37 ms`.
- Task rollout wall was `178.85/217.00/189.22 s` (`195.02 s` mean). Full Python wall was `669.24 s` (`11:09.24`), CPU user/system `658.44/6.36 s`, max RSS `1,857,144 KB`, exit `0`. The run included dual-rate diagnostic event generation, 50 Hz visibility renders, trace and video capture; this is the measured end-to-end debug path, not a production throughput estimate.

## Evidence and reproduction

- `episodes/episode_*/episode.csv`: strict scorer and contact/lift metrics.
- `episodes/episode_*/state_control_trace.jsonl`: per-step GT evaluator record and controller/event state trace.
- `episodes/episode_*/direct_hdr_events_1khz_2khz.npz`: paired raw events from the same trajectory.
- `episodes/episode_*/rollout.gif` and key Direct-HDR snapshots: representative failure and successes.
- `rescore/`: per-poll phase metrics and the evaluator-only rescore summary.
- `../runtime_logs/r0_7_heatmap_event_state_closed_loop_3seed_20261017.{launch.sh,log,pid,runtime,exit}`: exact launcher and process evidence; exit code `0`.
- `SHA256SUMS` covers the original raw run outputs. `SHA256SUMS.addendum` covers this README, evaluator-only rescore and runtime files. Verify from the repository root with `sha256sum -c` using each manifest's listed paths.

The runtime launcher records the command used. For a rerun, use the same parameters but choose a new output directory:

```bash
EGL_DEVICE_ID=3 CUDA_VISIBLE_DEVICES=3 PYOPENGL_PLATFORM=egl PYTHONPATH=. \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot/.venv-robot/bin/python \
-m benchmark.robot_closed_loop.run \
--mode reference \
--config benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json \
--speeds 0.3,0.4,0.5 --reference-episodes 1 --reference-y-values=0,0.055,-0.055 \
--event-controller-hz 2000 --reference-controller-source events \
--event-model-checkpoint benchmark/robot_closed_loop/results/r0_7_heatmap_event_state_development_20261017/spatial_event_state.pt \
--visibility-hz 50 --seed 2026103900 \
--out benchmark/robot_closed_loop/results/r0_7_heatmap_event_state_closed_loop_3seed_20261017_rerun
```

The recorded run used GPU 3 for EGL rendering, CPU Genesis physics and CPU PyTorch; other busy GPUs were left untouched.
