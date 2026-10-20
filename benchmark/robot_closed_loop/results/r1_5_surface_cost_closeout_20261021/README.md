# R1.5 surface coverage and compute-cost closeout

**Purpose.** Finish the cached-clip candidate repair by testing whether a surface-based intermediate-frame path can keep coverage, occlusion handling, event fidelity, and runtime suitable for downstream use. This is development evidence only: no formal ranking, retraining, or closed-loop rollout was run.

## Result

The candidate now has strong geometric coverage on the two archived development clips. Across 47 real logged query rows per episode, it requested exact rendering for a mean of **0.913%** of pixels in episode 004 and **0.951%** in episode 005; the largest per-row request fractions were **1.048%** and **1.095%**. It needed no full-frame fallback. Requested regions were repaired by the existing main-camera scissor renderer. The rectangular scissor crops still covered about 24.7–25.8% of full-frame pixels across the 42 reconstructed query rows, so a small pixel request does not translate to a proportionally small renderer cost.

Occlusion and surface diagnostics are good but not literally perfect. The supported-region segmentation agreement minimum was 99.9987% in episode 004 and 99.9895% in episode 005. Same-label depth P99 stayed below 0.019 mm. Across 42 frame metrics, dynamic occlusion misses were 34 and 21, false dynamic occlusions were zero, and wrong dynamic-link pixels were 0 and 101. The misses and wrong-link pixels are rare relative to the millions of supported samples, but remain recorded as residual errors.

**Event fidelity did not recover.** In the requested/rendered regions, exact timestamp/pixel/polarity matching was 608/608 events for episode 004 and 905/907 for episode 005. In the supported warped regions it was only 4/2,199 and 0/3,711. Candidate/direct total event counts were 1,774/2,807 and 3,106/4,618. The same causal five-bin query tensors had mean cosine 0.873 and 0.823, which does not compensate for the event-count and exact-match gaps. The old Ours checkpoint, used only as a diagnostic, had seven valid candidate-event queries per episode: median XY error 12.7 mm at center and 243.6 mm on +55 mm; the +55 mm maximum was 405.6 mm. This is not a retraining result.

## Runtime experiments

All paired runs used the same cached episode 004/005 source states, direct-HDR reference, 100 Hz source cadence, 47 logged query rows per episode, and 42 intermediate rows. They were offline state replays; physics was not advanced. A paired cost includes source preparation, reconstruction, requested-region HDR repair, candidate DVS, and amortized one-time scissor setup. It excludes shared scene construction, model inference, file serialization, and optional comparison audits. Numba JIT time is included when it occurs in a timed production phase. Direct cost includes all exact trace-state HDR renders plus DVS.

| Profile | Raster path | Candidate / Direct total | Ratio |
|---|---|---:|---:|
| U | CPU serial | 6.015 / 2.693 s | 2.23× |
| V | CPU tiled profile | 7.374 / 1.570 s | 4.70× |
| W | CPU profile | 9.988 / 1.797 s | 5.56× |
| X | CPU scissor profile | 10.265 / 2.069 s | 4.96× |
| Y | CPU cached indices; corrected for optional audit time | 10.915 / 1.769 s | 6.17× |
| Z | CPU Numba, 8 threads | 4.038 / 2.026 s | 1.99× |
| AA | CPU Numba, 16 threads | 4.293 / 3.151 s | 1.36× |
| AC | CPU parallel semantic-label raster | 6.107 / 2.172 s | 2.81× |
| AE | PyTorch CUDA triangle scatter | 13.540 / 2.619 s | 5.17× |

The lowest paired ratio was AA, whose direct run was also the slowest at 3.151 s. Ratios varied with shared-server load, but **no paired run beat Direct-HDR**; even the warm second episode of the best CPU profile remained 1.27× slower. Parallelizing independent labels was exactly equivalent in support and label output but was slower. The CUDA candidate is a forward triangle-fragment scatter rasterizer with a z-buffer, not an inverse-warp implementation; it was substantially slower and does not justify retaining a GPU path as the production candidate.

For the CUDA run, support and label masks matched the CPU serial raster in every audited call. Floating outputs were not bit-identical: maximum luminance difference was `1.19e-7`, and maximum depth difference was `5.96e-8 m`; the event output still did not match Direct-HDR. The PyTorch `scatter_reduce_`/`scatter_add_` smoke ran on the idle Tesla V100S. Numba CUDA was unavailable for this kernel on the installed CUDA 10.1/NVVM toolchain. A first CUDA attempt using the stock conda Genesis package exited before rendering because it lacked the installed HDR-overlay API; that failed attempt is archived separately and is not result data.

The successful AE runner loaded Genesis 1.2.2 HDR-overlay code from `.venv-robot` and PyTorch 2.5.1+cu121 from `Eventbased_WAM`. Genesis warned that this Torch version is unsupported. Nevertheless, each episode rerendered five saved source endpoints and ten saved dense Direct-HDR frames with zero HDR/depth error and zero segmentation mismatch. The warning remains an environment limitation; the exact cache rechecks are the evidence that the renderer output was unchanged in this diagnostic.

## Scope and decision

- The six general numerical controls still pass: static zero events, bidirectional micro-motion, analytic weak-change plane, smooth luminance ramps, foreground/background depth ordering, and exact requests for unknown pixels. The latest CPU control output is [`controls/cpu_controls.json`](controls/cpu_controls.json).
- The spatial sampling counterexample is repaired by continuous same-label triangles with a real depth buffer; small motion is no longer magnified into full-pixel brightness jumps. The real clips also have low unsupported-pixel requests and no full-frame fallback.
- The broader temporal image/event mismatch remains: supported-region event matching is near zero, Ours localization on +55 mm remains unstable, and the candidate is slower than direct rendering. No retraining or closed-loop test was started because the candidate did not pass these prerequisites.
- Frozen simulator core remains unchanged (`e79e656ac00f7847b98db9e12bf43b093f3afff1`). Scene, camera, physics clock, control, and strict scorer are unchanged.

There are two reasonable next directions for the research session to choose. **A:** end the warp-candidate line and retain exact Direct-HDR as the shared R1 reference source. **B:** authorize one bounded benchmark-layer experiment that rasterizes reconstructed surface triangles inside the existing GPU/OpenGL rendering path, then checks event fidelity and total cost before any retraining. The PyTorch scatter experiment shows that a generic per-call GPU tensor path is not sufficient. Continuing B would be a new implementation direction; it should not delay Direct-HDR reference use for R1 if that reference already meets the experimental need.

## Reproduction and provenance

The successful cached replay ran on g1 in the isolated checkout below; its launcher refuses to overwrite existing paths:

```bash
cd ~/Eventbased_WAM/repositories/genesis-event-camera-plugin-r1_5_multiref_scissor_20261021ae
bash benchmark/robot_closed_loop/diagnostics/launch_r1_5_multiref_scissor_candidate_20261021ae.sh
```

Run code commit: `1f5c1815490f45d00ef61f6d6f167ae31f614017`. The current CPU-default setup fix is commit `2d60e28`; it does not change AE's CUDA output. Frozen-core base: `e79e656ac00f7847b98db9e12bf43b093f3afff1`. CUDA config SHA256: `1ce44bca85f2970acaa91ea4298e9e991bf3c55fbd420025894bbe11a6fcf702`. ROI repair config SHA256: `ed10df8b944494a28c0e6accd06a8af1e891b1114721d4275ef625b1b6913f6e`.

Episode 004 hashes: trace `3f07e4e7a0846827be7a3d5decf7c6197d77ca0237bacff55f2c45a70a04349c`, source cache `0525c18c8d89f5831ef0cb71ac9e3b6b130d60579daf374f06d5cfcee5b879d7`, direct-render evidence `b3afe5b5badee3d2606d343bb750aa970887928c41b8001a5fc4e9ff32b9c321`. Episode 005 hashes: trace `f0f556842029cd54b3fc082ee0e2848dd194f3322a8a9c0769c2c889a6f6ae5d`, source cache `91c1397642e317d6bb51150b4107c518b52e4fa9fcf5f1caf6a2a91315c7d434`, direct-render evidence `f161079a24c1bf9fee838a6fbb2a0fa5d785435b116f0ea5628e6efbba497236`. Ours/V2E checkpoint hashes: `6ecea6a4520ae09b8cc99ebfd8da7a6a324363d2e57dbf73790e27ee2dbec8d2` / `a7140b63f3b5042b0dc0ed7cc3aa2974c4e9859db919f318826358883f7ce8a0`.

The successful AE output is archived at [`r1_5_multiref_scissor_candidate_20261021ae/`](../r1_5_multiref_scissor_candidate_20261021ae/), with `SHA256SUMS` file SHA256 `75aa1a4c0374b1fd1c9ee9a9d5f954cbbc4dd112057780c332b42c43d4e64cb8`. Its g1 source path is `~/Eventbased_WAM/repositories/genesis-event-camera-plugin-r1_5_multiref_scissor_20261021ae/benchmark/robot_closed_loop/results/r1_5_multiref_scissor_candidate_20261021ae/`. Runtime log SHA256 `e0768a03a59164fb7c707239955d3a9d9e12a68774feb5030e0219bd7bc71765`, exit code `0`, PID `2337284`. Other full g1 manifests are copied to [`g1_manifests/`](g1_manifests/) and the compact profile summary to [`cost_profile_inventory.json`](cost_profile_inventory.json). The original Y output manifest and corrected audit addendum are both retained; the original result files were not rewritten.

Attempt outcomes and their runtime log checksums are also listed in [`environment_attempts.json`](environment_attempts.json); failed AB/AD attempts are preserved as diagnostics and are excluded from performance results.
