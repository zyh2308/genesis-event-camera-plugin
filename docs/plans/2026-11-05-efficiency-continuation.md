# Projection storage reuse implementation plan

**Goal:** Reduce event-generation allocation/copy cost beyond the exact archived index-reuse v1 candidate.

**Architecture:** Keep the immutable full v1 package as reference. Apply one geometry delta in an isolated package: XYZ projection storage aliases its final homogeneous buffer. Preserve operation order, dtype, complete noise/RNG sequence, sampling and visibility rules. No cross-frame cache or algorithm changes.

**Tech stack:** Existing PyTorch 2.8 CUDA environment on g1 GPU2, private sm70 build with --fmad=false and no fast-math.

1. Verify all handoff index hashes, candidate manifest and 17-file reference. Keep shared checkout, banks, labels, checkpoints, frozen releases and environments untouched.
2. Create independent WSL and g1 worktrees, output and private build directories. Obtain global + UUID flock reservation after live process/memory/load checks, then recheck.
3. Restore complete v1 per manifest. Apply only release/genesis_event_plugin/gpu_geometry.py. Record new source manifest.
4. Run source/check_projection_storage.py (42 exact projection cases), then source/run_checks.py --mode p0. Require actual attach/capture, N=377, sensor call once/interval, no cap/fallback, exact raw events, intermediates, state and RNG.
5. Profile actual v1 entry separately. Report three remaining costs with inclusive versus self timing distinguished. Compare the candidate independently; exclude profiler overhead from performance claims.
6. Measure at least three synchronized warm trials with alternating package order and identical physical inputs; record initialized wall time, allocated/reserved memory and actual events. Keep deterministic and normal mode separate. Only retain a candidate with measured resource or time benefit.
7. After a successful pilot, run five class short exact gates and an s3_00 200ms exact/actual-three-loader probe. Reuse unchanged sensor/fallback qualification where no new risk exists; record what was not rerun. Default CUDA strict raw/loader qualification remains failed as inherited.
8. Archive source/hash, all measurements, failures, evidence references and lease release. Do not freeze/promote or replace formal datasets. Escalate only a demonstrated need to change physical/noise/sampling or production contracts.

No formal training, optimizer change or real-test scoring is part of this delivery.
