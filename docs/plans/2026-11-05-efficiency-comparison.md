# Matched EventGenesis / v2e efficiency and release plan

**Goal:** Measure matched-input, matched-resource event-generation costs before deciding whether to optimize further or freeze a qualified release. Never change formal training or the frozen reference.

**Architecture:** Use the existing five fixed synthetic scenes and the complete Stage 9 sensor profile. Compare the frozen implementation, the GPU-resident candidate, and pinned official v2e + SuperSloMo on g1 GPU 2. Separate warm in-memory generation, official file-based use, sensor-only controls, and common rendering/I/O costs. Freeze only an independently verified candidate with documented numerical contracts.

**Tech Stack:** Existing Torch 2.8 / CUDA, pinned v2e `cac638444557a224ba6ee5c29260916f152c00bd`, frozen simulator `62da4f35ce3c5c96f04268bd1687a27657af9238`, Python, NumPy, HDF5. No shared environment changes.

## Predeclared comparison

- Cases: rich s0_00/s1_00/s2_00/s3_00 and pose train_000; 20 consecutive 10ms intervals (200ms), rich index 100 and pose index 0. Every case has sufficient endpoints. No real data or GT-based scene selection.
- Primary: warm in-memory physical/display endpoints to sorted CPU Nx4 events. v2e uses its own unmodified SuperSloMo inference, transforms, uint8 image conversion, resize and EventEmulator. Only lossless PNG write/read is replaced by equivalent in-memory uint8 arrays, proven against actual PNG decode. Both retain native numeric precision, full noise/refractory settings and output all events.
- Match the 377 substeps / 10ms of the complete Stage 9 profile: v2e fixed factor 377, native midpoint interpolation unchanged. Record actual time spacing, initial/final coverage and endpoint conventions. Do not call sampling spacing identical timestamp quantization or event accuracy.
- Additional official 25us / factor400 control, and stock file-based CLI control. Any native CLI time-rescaling or input boundary issues are disclosed; no faster but broken output counts as a win. Exclude process/model startup from warm timing; record separately in cold measurements.
- Three repetitions per case, rotated order, synchronized CUDA, exclusive GPU, same two Torch/OMP threads and one OpenBLAS thread. Common event serialization uses the same writer. Record phase times, peak memory, output count, ordering/bounds, input/source hashes, environment, clocks/power, contention.
- Sensor-only common-frame control isolates geometry/interpolation cost; differing sensor equations and FP32/FP64 are explicit. Do not require different generators to output identical events.
- No fps guess or dividing historical unmatched production times. Use simulated-seconds / wall-seconds and wall-time ratio; events/sec is supplementary because event counts differ.
- Predeclared practical advantage: geometric-mean speedup >=1.20, scene-resampled 95% CI lower bound >1, no unexplained case slower by >10%. Limited to these fixed scenes/device/profile. Failures and exclusions remain visible. Otherwise investigate measured dominant cost and perform one semantics-preserving optimization attempt, rerunning the same suite.

## Correctness and conditional freeze

1. Keep all frozen sources, physical banks, formal training processes/configs/checkpoints unchanged. Reserve only GPU2 using the existing coordination lock after live availability checks.
2. Validate memory-backed v2e against native PNG-backed inference and sensor output on identical frames/RNG; audit timestamps, not just successful exit.
3. Recheck candidate sensor state/RNG, hole/HDR/occlusion behavior, lazy diagnostics, normal NumPy API, recorder payload and restore. Preserve the known strict raw-float bitwise failure. Accept no lower-precision math, altered noise, fewer samples or hidden masks.
4. If advantage qualifies, create an additive, versioned optimized entry point with explicit supported profile and exact-reference fallback for unsupported settings. Remove process-global monkey patches from production code, package CUDA source/build provenance, and test the actual integrated entry point (including Genesis adapter/recorder smoke).
5. Freeze only after required checks pass. State any limitations of float reproducibility and online renderer access. Do not silently switch running experiments to the new release.
6. Deliver raw timing CSV/JSON, comparison report, source/hash manifest, release or optimization decision and reproducible commands. Release GPU2 at completion; no empty holding process afterwards.
