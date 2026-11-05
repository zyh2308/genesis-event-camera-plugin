# Coverage first-layer z-buffer reuse implementation plan

**Goal:** Reduce event-generation allocation, synchronization and compute cost beyond the exact archived index-reuse v1 candidate, without changing its output contract.

**Architecture:** Keep the complete archived v1 package as reference. The selected geometry delta uses its already-computed `zbuf` as read-only `nearest` on the first coverage-splat iteration, where all unresolved masks are initially true. Later iterations keep the original full nearest-depth calculation. Preserve weighted reduction order, dtype, sampling, complete noise/RNG, visibility, coverage and unknown rules. No cross-frame cache or algorithm change.

**Tech stack:** Existing PyTorch CUDA environment on authorized g1 GPU2, independent worktree/output/private sm70 build, `--fmad=false`, no fast-math, and registered deterministic checks with `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

1. **Completed: evidence and isolation.** Read the handoff, evidence index and project rules; verify the archived v1 source manifest and preserve the shared checkout, banks, labels, checkpoints, frozen cores and environments. Use global + UUID flock reservation with live preflight and a post-reservation recheck for all GPU work.
2. **Completed: first candidate rejected.** The initial projection-storage alias candidate was tested independently against v1. Its normal-mode s2_00 two-interval trials were slower (candidate 1.8492/1.8532s, v1 1.7953/1.8187s), and peak allocated/reserved memory did not fall. Preserve its exact source in `candidates/projection_storage/`; do not combine it with the final candidate or claim an efficiency benefit.
3. **Completed: identify a different repeated cost.** The first coverage-splat nearest-depth pass repeats the same all-candidate ordered `amin` operations already used to construct `zbuf`. Reuse that existing result only on the first iteration; leave subsequent layers and the bounded nonfinite-tolerance fallback unchanged.
4. **Completed: independent selected delta.** Build `candidates/zbuf_reuse/genesis_event_plugin/gpu_geometry.py` directly from archived v1, with no storage-alias change. Its build receipt verifies AST syntax, unchanged other functions and byte-identical bounded fallback. The root `release/` delta now points to this measured zbuf candidate, byte-identical to the archived candidate source.
5. **Completed: selected strict short checks.** All 35 registered component stress cases and five real plugin input classes with two intervals each passed exact checks. The actual attach/capture path must retain N=377, one sensor invocation per interval, zero fallback, and exact raw events, intermediate physical observables, state and RNG. Default-mode inherited strict failures remain separate.
6. **Completed: measured deterministic pilot.** On s2_00 200ms, three trials per package gave median v1 34.41667269496247s and selected candidate 33.875527661060914s (ratio 1.015974512). Peak allocated fell 11.9453358655 to 11.6530499458 GiB; reserved fell 13.994140625 to 13.505859375 GiB. Every trial produced 46,698 events. Keep profiling independent and do not generalize this single-fragment pilot to five scenes or native renderer end-to-end performance.
7. **In progress at this update:** s3_00 20-interval continuous exact probe and the actual three-loader checks. Do not mark either passed until execution receipts complete. The execution owner will supply their final status and the measured remaining bottlenecks.
8. **Pending final archival:** retain all trial timings, failures, exact sources, full-package source manifest, raw artifact references, limitations and confirmed lease release. The execution owner will fill the final report. Do not freeze/promote, replace production data, or modify an active script merely to simplify archival.

No formal training, optimizer change or real-test scoring is part of this delivery. Any new physical/noise/sampling contract, frozen upgrade or formal-data replacement must first be reported with evidence. Default CUDA strict qualification remains failed; no tolerance or timestamp quantization is used to turn a failure into a pass.


Final first-delivery check: selected zbuf reuse passed all 35 pressure cases,
five two-interval exact plugin fixtures, the s3_00 continuous 200ms exact
fixture, and all tested actual three-loader windows (6 flow/20 pose/100
reconstruction). GPU2 was released after children exited. Production freeze,
default CUDA strict qualification and formal datasets were not upgraded.
Full measurements and the rejected storage candidate are archived in the
review's RESULTS_ZH.md and evidence/.
