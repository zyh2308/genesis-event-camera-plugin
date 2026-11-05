# Exact adaptive-sampling cost reduction — implementation plan

> Execute the authorized steps in this session; retain review checkpoints and all failed evidence.

**Goal:** Reduce unnecessary computation without changing the 377 sensor times, numerical precision, stochastic process, visibility/coverage/unknown rules, or any running training.

**Architecture:** Start from an independent copy of the qualified Stage9 GPU package. Remove unused mixing and the histogram used only as an iteration upper bound; test optional exact full-coverage pruning separately. Investigate temporal splat tiling only if it preserves every computed projection, reconstruction and diagnostic. Never replace dense radiance with interpolated sparse approximations.

**Tech stack:** Existing g1 V100S GPU2, pinned Torch2.8/CUDA environment, existing private compiler, NumPy/HDF5, immutable physical fixtures and official V2E+SSM.

## Task 1: Register and isolate

- Worktree: `/home/科研/Eventbased_WAM/repositories/gec-sampler-cost-20261105`, base `f918aa33596edc75e822914cf59430f80ac879bf`.
- Run: `/home/zhaoyuhan/Eventbased_WAM/runs/simulator_sampler_optimization_20261105e`.
- Verify GPU UUID/processes and atomic lease; use only GPU2. g3 remains prohibited. Keep all formal data/checkpoints, frozen17 files and shared environments unchanged.

## Task 2: Build independent candidates

- `geometry_reuse_candidate.py` reads immutable `gpu_geometry.py` and writes new modules/packages; assert old source hashes unchanged.
- Candidates: `deadmix`, `histfree`, `histfree+deadmix`, optional `prune-full` with conservative numerical guard.
- Preserve bounded reference behavior for nonfinite tolerance parameters; no infinite-loop regression.
- Compile each candidate before GPU execution; no runtime replacement in any production process.

## Task 3: Same-input component gate and profiling

- Capture identical actual projection/radiance/depth/surface tensors from the existing provider, using the full377 time stack.
- Compare frames and every visibility/coverage/surface/z-buffer/unknown field byte for byte under the same deterministic settings; preserve normal-mode diagnostics separately.
- Stress nonnegative HDR/lowlight, empty/no-support, depth crossing, partial coverage, nonfinite/overflow guard, RGB and source0/source1 boundaries. Assert actual nonzero mask flips where claimed.
- Separate instrumented profiling from uninstrumented CUDA-synchronized timing. Record time and memory per candidate; select only passing candidates, never by downstream real score.

## Task 4: Actual plugin/consumer gate

- Use the real candidate package entry, assert optimization active and no hidden reference fallback.
- Fixed five fixtures plus a continuous long clip: compare intermediate frames/masks, rawfloat64 events, all sensor tensor states/counters and RNG ordering.
- Reuse the actual three-loader scripts and independent fixed-checkpoint reconstruction probe from `simulator_formal_acceptance_20261105d` if raw/input differences require tracing. Deterministic exact gate must pass without quantizing timestamps or increasing tolerances.

## Task 5: Fair efficiency comparison

- Same five physical clips and hashes, three rotated warm trials, same GPU/thread budget and cached-input→sortedCPU-event boundary.
- Show the previous GPU package and candidate, both native-auto. V2E native-auto/minfactor2 stays present; auto/minfactor63 is a separate numerical-step control; fixed377 is secondary only.
- Keep the complete noise/profile settings, official SSM and actual midpoint timing; disclose first/last supported times. Use common interior support for descriptive event statistics, without stretching timestamps or treating event count as fidelity.
- Do not assert quality-matched ranking without independent quality/convergence evidence. Larger-scene training and real tests are outside this optimization run.

## Task 6: Deliver and release resources

- Record full results including failed attempts, input/source/binary hashes, exact commands, raw paths and bounded scope.
- Publish report/test code in the shared research branch; freeze a new optional candidate only after its actual entry is accepted. Keep the previous release intact.
- Confirm own GPU jobs exit and release GPU2. Existing three-line training remains untouched.
