# R1.5 core candidate closeout validation (run B)

## Result

Run `r1_5_core_candidate_final_20261102b` completed on g1 on 2026-10-31, with runner exit code 0 and output-checksum verification successful. It used the isolated candidate source snapshot at commit `e71f2b7b7369994d714f540ff498f3902b90081c`; the formal frozen baseline and shared checkout were not changed. This was a CPU-only regression: no renderer, CUDA, physics advancement, training, or ranking ran in this job. The worker used Python 3.11.15, NumPy 2.4.6, Torch 2.8.0+cpu, one CPU thread, and peaked at 359 MB RSS. The diagnostic took 13.29 s of measured validation work and 18.90 s wall time.

The previous isolated attempt `r1_5_core_candidate_final_20261102` stopped at the last control because its source snapshot omitted the simple-contour config. That failed attempt and its logs remain intact. Run B staged the missing config under a new run ID and completed; it did not overwrite the failed output.

## Controls

| Control | Run B result | Decision |
|---|---:|---|
| Exact observed endpoints | NumPy and Torch maximum radiance error 0 | Pass |
| Same CPU projection/compositing semantics | coverage/luma maximum delta 0; support/visibility mismatch 0 px | Pass |
| Continuous surface at −1e−6 / +1e−6 px | maximum interior radiance change `2.98e−8` | Pass |
| Tiny footprint | coverage `9.54e−7`, visible but unsupported | Pass; preserved as unknown |
| Depth tie band | tested gaps `0.858` and `1.049 µm` around configured `0.954 µm` at 1 m | Pass for these probes |
| Foreground over background | interiors `0.8` / `0.02`; mixed edges match expected front-to-back values within float32 rounding | Pass |
| Analytic 10 ms log-linear brightening | 48 vs 48 events; max timing error `3.81e−10 s`; static control 0 events | Pass for the supplied analytic ramp |
| Strict mode without a real intermediate sample | raises `RenderRequiredError` with a timestamped request | Pass; no synthetic subframes presented as observed data |

The depth tie result is a narrow numerical probe, not proof for every scene scale or material. The bilinear coverage threshold remains `0.995`; it is an approximate warp footprint, not exact renderer pixel coverage.

## Cached robot clips and sampling density

The exact 1 ms direct-HDR frame sequences for episodes 004 and 005 were replayed through strict mode. Both image sequences matched their cached frames pixel-for-pixel; raw DVS outputs matched exactly, including timestamp, pixel, and polarity: 625 events (354 ON / 271 OFF) for 004 and 939 (536 / 403) for 005. Each interval output was marked available only at its right endpoint. These are archived real samples, not newly rendered frames in run B.

A single 5 ms midpoint does not preserve the 1 ms event stream: it produced 413/625 and 615/939 events, pixel-polarity count L1 differences 212/324, and event-time P95 errors 4.013/4.275 ms. The existing motion sampler requests six more real samples within the tested 10 ms interval. The supported-pixel P95 image error was small, but isolated silhouette/appearance samples still had larger errors; event totals alone would hide this.

## Shadow, MSAA, and contour controls

Candidate images were compared against the already-rendered, internally matched shadow × MSAA factorial. Candidate paths using only a sparse midpoint kept unknown fractions at 4.37–4.67%, below the fixed 15% ceiling. Supported-interior log-error P95 was at most `2.45e−4`, below the fixed 0.2 ceiling. There are still high-tail pixels near changing edges: 0–120 supported pixels per factorial cell exceeded 0.2 log error, with a maximum observed error of 2.156. These were retained and reported rather than averaged away.

| Episode | Shadow | MSAA | Direct sparse events | Warp-midpoint events | Pixel-polarity count L1 | Unknown | Supported-interior log P95 |
|---|---|---:|---:|---:|---:|---:|---:|
| 004 | on | on | 413 | 397 | 16 | 4.37% | 1.68e−5 |
| 004 | on | off | 513 | 490 | 29 | 4.42% | 1.67e−5 |
| 004 | off | on | 40 | 40 | 0 | 4.37% | 1.02e−5 |
| 004 | off | off | 97 | 97 | 0 | 4.42% | 7.27e−6 |
| 005 | on | on | 615 | 582 | 59 | 4.55% | 2.25e−4 |
| 005 | on | off | 927 | 941 | 118 | 4.67% | 2.44e−4 |
| 005 | off | on | 252 | 252 | 0 | 4.55% | 1.75e−4 |
| 005 | off | off | 613 | 631 | 42 | 4.67% | 1.57e−4 |

In the simple moving-box contour control, the supported floor covered 99.89% of pixels and had P95 log error near float32 precision. With shadows on, 38 supported pixels exceeded 0.2 log error; with shadows off, none did. This supports a real shadow/light contribution in this control, but does not attribute every live-episode residual to shadow alone. MSAA state also changed event counts, so the resolve mode remains part of source provenance.

Strict cached samples made the raw event stream match the corresponding direct sparse replay exactly with zero additional render requests, because those frames already existed. That validates the strict API and causal labeling; it is not a production render-cost measurement. The single-midpoint event discrepancies above remain real and rule out treating 5 ms endpoint spacing as universally sufficient for these dynamic clips.

## Combined renderer and causal evidence

The complementary actual-renderer run I used the same core module contents as this candidate (no `genesis_event_plugin` changes occurred between run I and commit `e71f2b7`). On g1 GPU1 it freshly rendered 18 exact 1 ms HDR interior states across the two archived episodes. Every frame matched the archived radiance exactly, and direct DVS replay matched 625/625 and 939/939 events. Render plus GL readback cost was 289 ms total, 16.04 ms per frame; the sequence was available at the 10 ms interval's right endpoint. This is offline replay and is slower than the 1 ms simulated spacing; wall-clock real-time operation is not claimed or required.

The separate live causal smoke K advanced Genesis physics normally for two 0.3 s lateral crossings with 0.5 ms physical steps. It captured 1,202 native HDR observations, assembled 300 strict plugin intervals from actual samples with maximum 0.5 ms gaps, and produced event packets at 2 ms endpoints. The event-only policy read each packet and the close command took effect on the next physics step. No target GT, GT segmentation/depth, future state, or pose overwrite entered the policy. This is a causal integration smoke, not a grasp-success or sensor-ground-truth test. Its plugin path averaged 361 ms wall time per 2 ms simulated interval and is not real-time.

## Freeze scope and remaining limits

The combined evidence supports freezing the **versioned strict sampled mode** for Genesis linear-HDR generation under the tested contract: float64 projection, unchanged `0.995` support threshold, exact endpoint identity, explicit unknown support, real timestamped HDR samples when strict mode requests them, and event availability no earlier than the bracket's right endpoint. It does not support a claim that unobserved sub-5-ms illumination transients are reconstructed correctly. The strict API default remains opt-in for compatibility; the legacy permissive/hold path is an explicitly approximate mode and is not covered by the strict fidelity claim. A downstream scientific configuration must set `strict_fidelity=true` and supply a causal real-radiance provider.

Residual risk is concentrated around changing shadows, silhouettes, and other unsampled photometric events. The 5 ms floor is only a minimum sampling constraint; the motion sampler may require denser observations. Acquiring those observations costs more wall time than their simulated interval in the measured g1 runs. No old perception checkpoint, closed-loop grasp success, or scene-general photometric accuracy was established here.

## Reproduction and hashes

Remote run root:

`/home/zhaoyuhan/Eventbased_WAM/runs/robot_scenes/genesis_event_camera_plugin/r1_5_core_candidate_final_20261102b/`

CPU validation command recorded in `logs/time_v.txt`; rerun with a new run ID/output directory:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$PWD/source" \
  /home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot/.venv-robot/bin/python \
  source/benchmark/robot_closed_loop/diagnostics/validate_r1_5_core_candidate.py \
  --cache-root /home/zhaoyuhan/Eventbased_WAM/repositories/genesis-event-camera-plugin-r1-diag-20261019/benchmark/robot_closed_loop/results/r1_failure_chain_diagnosis_20261019a/native_hdr_cutoff059 \
  --factorial-root inputs/r1_5_shadow_multisample_cached_states_20261029_g1_gpu0_previewfix_855b4b0 \
  --config source/benchmark/robot_closed_loop/configs/r1_5_core_candidate_20261030.json \
  --output output_new_run_id
```

- Candidate branch/source commit: `exp/r1_5-core-candidate-20261030` / `e71f2b7b7369994d714f540ff498f3902b90081c`.
- Candidate config SHA256: `bde0882f0a70e14dfea4d70f94557579bcb089aaa78bfca85a5c18d6bdc0b7b4`.
- Validator SHA256: `b127a168d690bd21df78999aa67237571486261d590be437c0c4206883810af8`.
- Core modules SHA256: `physics_interpolator.py` `0ac9174f5e3a0635de0dc8cb7a28c51f4d6ac0eb690bfeb1ae8486bf44a5941d`; `physics_plugin.py` `85226f8e0fa57c26512a77d76ee3444580ebe3566a4871682d1c108a1a449bba`; `physics_warp.py` `8c23c601677cf25c1df445ea65235b154f5895353ba9bc67f37da40a3ae3fe41`; `torch_physics.py` `8a409765268746fefedb15a805c4d95db857bf6a4359c899cd65e6ac4dfe5955`.
- Episode state cache SHA256: 004 `0525c18c8d89f5831ef0cb71ac9e3b6b130d60579daf374f06d5cfcee5b879d7`; 005 `91c1397642e317d6bb51150b4107c518b52e4fa9fcf5f1caf6a2a91315c7d434`.
- Shadow/MSAA source archive SHA256: `da5f37122cc9aa64958704e9cf7442760606b5db2132152a1b93d00321c5942c`.
- Run B output manifest SHA256: `8b701523a614cbb5911af264181224050ec59cef17b24a7b525e9aebe7376059`; summary SHA256: `92414b119220972256aafb4433fc66334b43dd057aba7c42910d71b9f75406f5`.
- G1 evidence bundle SHA256: `3383442b07f276940c805ce75df4e7a515395239c517bd1c84e99ecfa62d4239`.

Full raw arrays, output hashes, logs, resource preflight, and provenance are retained in this result directory. `output/SHA256SUMS` verifies every generated output.
