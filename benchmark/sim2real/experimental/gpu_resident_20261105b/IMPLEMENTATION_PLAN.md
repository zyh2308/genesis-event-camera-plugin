# GPU Resident Pipeline Implementation Plan

**Goal:** Connect frozen FP64 geometry and the validated complete-profile sensor kernel without intermediate full-frame CPU round trips on g1 GPU 2.

**Architecture:** Isolated adapters outside the frozen core. Preserve the frozen sampler, input bank, coverage/unknown rules, noise RNG ordering, precision, and public sparse-event serialization. Compare resident frames with reference frames, then sensor state and event equivalence. Preserve failed gates. Diagnostic tensors remain available; measure event-only capture and forced diagnostic export separately.

**Tech Stack:** Existing Torch 2.8 environment, frozen 17-file core, prior CUDA sensor extension, NumPy reference fallback, private GPU UUID lease and test queue.

1. Acquire a fresh bounded GPU 2 lease under the existing UUID/global locks after checking processes and resource budgets. Create an isolated worktree from c1d4bb8; no changes to training checkouts or shared environments.
2. Extract the already tested sensor path into a CUDA-tensor input adapter. Keep RNG draw order and FP64 state; retain exact sparse CPU ordering at the final output boundary. Clone retained final frame views to avoid keeping whole stacks alive.
3. Make geometry outputs and diagnostics stay on device. Preserve full masks/render requests. First retain exact NumPy fallback only for unsupported holes, uploading sparse fallback values; separately test a pure-GPU fallback against reference arithmetic.
4. Verify identical projected inputs, holes, masks, endpoint behavior, strict rejection, sensor state, RNG, events, and independent-process restore. Include controlled occlusions and invalid depth, plus all five existing synthetic cases. Never relabel known frozen scatter variability as strict bitwise equivalence.
5. Benchmark baseline, prior fused path and resident path in alternating order with warmup and CUDA synchronization. Report medians/ranges, peak memory, host/device transfer counts and volumes, and forced full-diagnostic export. Include continuous intervals and only a bounded concurrency check when useful.
6. Run fixed-model/voxel regressions. Archive original failures, machine results, code and external hashes; verify frozen core and existing training resource identities remain unchanged. Publish results in the shared repository and release the temporary GPU lease.

The user has explicitly approved this design and execution. No additional confirmation or new training is required. No real data scoring is part of this task.
