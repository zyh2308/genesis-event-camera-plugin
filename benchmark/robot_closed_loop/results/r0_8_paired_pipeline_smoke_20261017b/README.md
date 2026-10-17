# R0.8 paired generator pipeline smoke

This is a two-episode pipeline smoke only. It is not a model comparison, score, or protocol freeze.

## Provenance

- Rollout and generator code commit: `3bef1684824acbe1db6de8aa215429eeee46dab0`.
- Scene config SHA256: `aa97f6f5a12af92561ef053407bb795bba2801582958b4ad1134bcd24497435a`.
- Shared-control config SHA256: `bb88542e49924acc428120ce21ade6660cb0813a2ecc791302aa5b8fdc718383`. The exact used config is preserved in `configs/shared_control_r0_8_candidate.used.json`.
- Split contract SHA256: `8070763b5b2e98ecc6be76b054d4d917094f1a077c7440933d889649ecad509f`.
- Ours and V2E source manifest SHA256: `49483b76d90e2a6b584f2c5a269e1818bebf599a76495319c99be5ef1e865ffa` and `76457f3beb77f8f6824d6294b68e3c8e89fe85cf2a698067463d158f6c415be7`.
- Result JSON SHA256: `7c8d49196cf3cf76763cd68c02947f80318715fb04691108a3f0659db43b77a6`.
- The full g1 output manifest SHA256 is `644a538864acdceac51d213946f6a2e043d7711bffe804cbb3a3cf60ae7e4581`. All transferred files matched that manifest; run `sha256sum -c SHA256SUMS.local-artifacts` for this repository subset.

The full raw cache and all rendered frames remain on g1 at:

`g1:/home/zhaoyuhan/Eventbased_WAM/repositories/genesis-event-camera-plugin-r0_8-7134f27/benchmark/robot_closed_loop/results/r0_8_paired_pipeline_smoke_20261017b/`

Verify the complete remote artifact set there with `sha256sum -c SHA256SUMS`. The cached HDF5 hashes are in `cache_sha256_manifest.json`; the two rollout caches are 11,804,632 and 11,854,085 bytes. The local result subset is 33.6 MB and contains both event training sources, their common episode labels/traces, both tiny checkpoints and fit outputs, episode summaries, and source provenance. The larger HDR/depth/segmentation/frame caches are intentionally left at the g1 address above.

## Run

```bash
EGL_DEVICE_ID=3 CUDA_VISIBLE_DEVICES=3 PYOPENGL_PLATFORM=egl \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=. \
  .venv-robot/bin/python -m benchmark.robot_closed_loop.diagnostics.run_paired_generator_smoke \
  --config benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json \
  --control-config benchmark/robot_closed_loop/configs/shared_control_r0_8_candidate.json \
  --output benchmark/robot_closed_loop/results/r0_8_paired_pipeline_smoke_20261017b \
  --speed 0.4 --y-values -0.055 0.055 --seed 2026104130 \
  --v2e-source /home/zhaoyuhan/Eventbased_WAM/code/v2e_source \
  --slomo-model /home/zhaoyuhan/Eventbased_WAM/code/v2e_source/input/SuperSloMo39.ckpt \
  --python .venv-robot/bin/python --train-epochs 3
```

The complete server run exited 0. Wall time was 769.34 s, CPU user/system 1017.46/26.78 s, and maximum RSS 2,253,644 KB. The measured stages were shared physics/robot rollout and cache 23.74 s, Ours plus official V2E generation 654.70 s, and both three-epoch fits 28.52 s. Ours cached plugin replay alone took 190.99/197.57 s per 1.11/1.12 s episode. V2E/SuperSloMo processed the second episode's 226 interpolated frames in 123.42 s (1.83 frame/s). These are pilot costs, not scale extrapolations.

## Observations and limits

- Two cached rollouts used one speed, `0.4 m/s`, with launch y `-55/+55 mm`, seeds `2026104130/2026104131`. Each was captured once at 100 Hz from the same 0.5 ms physics trajectory; both generators consumed the exact frame timestamp hashes, common episode labels, and common episode split. Each cache/hash pair is repeated in the two source provenance files.
- Both episodes reached bilateral contact and lift. The old 0.5 s post-contact stop window censored the strict 0.5 s continuous hold: only 0.255/0.256 s of qualified hold was observed. Therefore the stored `success=false` is **not a completed strict failure or success**. The scorer was unchanged. The candidate diagnostic window has since been lengthened to 1.0 s; this cached run used the earlier config and must not be used as the updated strict oracle regression.
- The source split was exactly episode 0 train (`0.4 m/s`, y `-55 mm`) / episode 1 validation (`0.4 m/s`, y `+55 mm`), with 108 validation windows per model. A generic legacy field in the archived trainer summaries incorrectly says both splits cover all speeds and lateral positions. That sentence is inapplicable to this two-episode pilot; actual episode assignment and coverage above are authoritative. The training-summary code is corrected for future runs.
- Causal five-bin validation state age was 10 ms and validity was 100% in both small splits. Ours validation XY position P95 was 143.9 mm and arrival-time P95 267 ms; V2E was 112.7 mm and 600 ms. These diagnostics show the call path and the weak fit, not generator quality or ranking.
- Event counts were Ours/V2E `941,607/178,496` and `927,704/185,268` for the two episodes. The sensor response is not matched: Ours used ideal 0.2 thresholds, no noise and infinite bandwidth; V2E used the frozen project YAML (sigma 0.03, cutoff 300 Hz, leak/shot noise, 0.5 ms refractory). Any later difference is a full-generator pipeline difference until response parameters are aligned.
- The shared R0.8 gate still has an unresolved front-edge close-only false eligibility. The centered true hold passed and empty gripper did not trigger. The R1 dynamic runs held stable through the observed censored interval, but that does not qualify the gate against the front-edge negative. Do not use this smoke as the new strict oracle regression or learned Direct-HDR R2.

Runtime log, `/usr/bin/time -v` output, launcher, and exit file are under `../runtime_logs/`.
