# Baseline versions and execution ledger

This file is part of the benchmark provenance. It records what is actually
available in the current environment; “source present” is not treated as
“successfully executed”. No baseline is relabeled when its official build is
unavailable.

## Frozen method under test

| Field | Value |
|---|---|
| Repository | `zyh2308/genesis-event-camera-plugin` |
| Frozen commit | `e79e656ac00f7847b98db9e12bf43b093f3afff1` |
| Subject | `Fix v3 correspondence and visibility semantics` |
| Method label | `Ours-v3.1-direct-real_v2` |
| Core changes during benchmark | forbidden |
| Preset | `real_v2`, frozen before this benchmark |
| Calibration source | `/home/科研/Eventbased_WAM/validation/simulator-quality/real_v2/calibration/evk4_real_noise_fit_v2.json` |
| Noise-map source | `/home/科研/Eventbased_WAM/validation/simulator-quality/real_v2/calibration/evk4_real_noise_map_v2.npz` |
| Seed | `42` |
| Frozen config | `benchmark/configs/ours.yaml` (`sha256:aa17afad421d6d3252238101fd270965783947a7047b56e46c025021df91efa4`) |

`real_v2` is the project’s six-recording EVK4 dark/white-wall calibration. It
is not retuned on the dynamic sequence. Historical `clean`, `moderate`, and
`noisy` presets are excluded from the formal baseline set because they were
not tied to this camera’s measurements.

## V2E

| Field | Value |
|---|---|
| Official source | [SensorsINI/v2e](https://github.com/SensorsINI/v2e) |
| Local source | `/home/科研/Eventbased_WAM/code/v2e_source` |
| Version | `1.5.1` from local `setup.py` |
| Git commit | unavailable; local source is a non-git snapshot |
| Executed component | official `v2ecore.emulator.EventEmulator` core |
| Existing artifact | `/home/科研/Eventbased_WAM/validation/simulator-quality/current_data_precheck/external_baselines/v2e_paper_defaults/aer_events.h5` |
| Input | `/home/科研/Eventbased_WAM/validation/simulator-quality/demos/20260913_real_demo_v1/rgb_frames.h5` |
| Input shape | 420 frames, 1280×720, 10 ms step |
| Seed | `42` |
| Algorithm modification | none; only an I/O adapter and HDF5 writer |
| Frozen config | `benchmark/configs/v2e.yaml` (`sha256:155f0ef7305bac9c52aa09e485d4c5a1feeaf448f1279a2cccf04097b290add4`) |
| Formal status | smoke artifact only; not yet a fair final comparison |

Recorded parameters:

```text
pos_thres=0.20, neg_thres=0.20, sigma_thres=0.03
cutoff_hz=300, leak_rate_hz=0.01, shot_noise_rate_hz=0.001
refractory_period_s=0.0005, leak_jitter_fraction=0.1
noise_rate_cov_decades=0.1, photoreceptor_noise=false, seed=42
```

The existing run emitted V2E’s warning that a 100-Hz input step and 300-Hz
photoreceptor cutoff can make its IIR update inaccurate. A formal V2E run must
therefore use a protocol-approved input cadence or explicitly report this
numerical limitation; this is not to be hidden by retuning only V2E.

## ESIM

| Field | Value |
|---|---|
| Official source | [uzh-rpg/rpg_esim](https://github.com/uzh-rpg/rpg_esim) |
| Local source | `/home/科研/Eventbased_WAM/code/rpg_esim` |
| Local commit | `4cf0b8952e9f58f674c3098f1b027a4b6db53427` (`added link to python bindings`) |
| Source status | present |
| ROS/catkin build | not present in the current checkout/environment |
| Executable smoke | not run / unavailable |
| Parameters | no formal parameters selected because no executable was built |
| Formal status | disabled until an official build and supported input path pass smoke test |
| Frozen config | `benchmark/configs/esim.yaml` (`sha256:34d54fb6da81b02004d4af36c69dc8a51e4d71797df6df655182d2da08148b8d`) |

It would be scientifically incorrect to call a different image-to-event script
“ESIM” merely because the official build is inconvenient.

## ICNS / IEBCS

| Field | Value |
|---|---|
| Official source | [neuromorphicsystems/IEBCS](https://github.com/neuromorphicsystems/IEBCS) |
| Local checkout | absent |
| Version/commit | unavailable |
| Executable smoke | not run |
| Formal status | disabled; no substitute implementation will be relabeled |

## PECS

PECS is kept as a reference for real–simulation comparison design and CD/GD
definitions, not as a mandatory executable baseline for Phase 1–4. References:
[ECCV paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/6110_ECCV_2024_paper.php),
[trail repository](https://github.com/lanpokn/PECS_trail_version).

## Phase 4 decision

The baseline smoke test validates the frozen Ours artifact and the existing
V2E-core artifact, and records ESIM/IEBCS as unavailable rather than filling
the table with unsupported numbers. It does not run the formal benchmark or
perform parameter search.
