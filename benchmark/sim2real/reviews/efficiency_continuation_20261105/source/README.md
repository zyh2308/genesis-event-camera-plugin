# Isolated measurement entry points

These scripts acquire no reservation. The caller must first complete the g1
live resource precheck, acquire the existing atomic GPU lease, recheck actual
ownership/load, and set exactly one authorized CUDA_VISIBLE_DEVICES UUID.
They never install dependencies or write into the core package.

Use the existing Torch28 interpreter and derive CUDA_HOME/CC/CXX from the
registered private-toolchain receipt. Keep MAX_JOBS=2,
TORCH_CUDA_ARCH_LIST=7.0 and a new EVENTGENESIS_GPU_BUILD_DIR outside both cores.
No fast-math or FMA setting is changed.

measure_plugin.py --core COMPLETE_ISOLATED_PACKAGE --out NEW_OUTPUT --case s2_00
--intervals 20 --trials 1 [--deterministic]

The coordinator imports no Torch and starts a new OS process per trial.
Default trials=1 lets the external driver rotate reference/candidate order.
Each child loads physical/display inputs outside timing, warms the private
kernel, then creates a fresh plugin. It preserves the archived run_ours
function's setup/generation timers and sorted native FP64 CPU event output.
It saves timings.json plus trialN/result.json, events.npy, physical_inputs.json
and warmup.json. Report initialized_pipeline_s, not process launch duration.
All source files except gpu_geometry.py must match the archived index-reuse v1
manifest byte for byte; the actual geometry hash and v1 anchor are recorded.

--deterministic sets CUBLAS_WORKSPACE_CONFIG=:4096:8 before Torch import and
enables deterministic algorithms. Default mode clears the CUBLAS setting and
disables deterministic algorithms. Keep these results in separate tables.
Neither mode's runtime checks claim pairwise equivalence or supersede the
preserved default-CUDA strict failures. Use the independent full plugin gate
and physical comparator for exactness.

Peak allocated/reserved memory is reset after warmup, and captured separately
at the setup/generation reset boundary. pipeline_peak_* is the maximum of
these two phases; persistent runtime allocations are included. The archived
worker_peak_bytes value is generation-only and is retained for provenance.
Only the post-timing close hook collects sensor tensor dtype/device/config
and optimization status; no per-frame observer is inserted in timing.

profile_plugin.py --core COMPLETE_ISOLATED_PACKAGE --out NEW_PROFILE
--case s2_00 --intervals 1 [--deterministic]

This separately warms the same entry and runs one instrumented profile with
shape/memory recording. profile.json contains operator self/inclusive CPU and
device times/memory plus projection, coverage_splat, warp, finish and sensor
call counts. trace.json is the raw Chrome trace. Inclusive phase times overlap
and must not be summed. Instrumented wall values are not performance results.

Both scripts compile only the unchanged digest/Fixture/validate/run_ours AST
definitions from the repository's archived
sampler_cost_20261105/source/compare_static_template.py. The snapshot adapts
CORE and ROOT, but does not execute the legacy OLD sys.path mutation or V2E
imports. If copied outside the repository, pass --fixture to that archived
template; its sibling ../release/manifest.json is also required. This avoids
importing another session's mutable run directory.

Outputs must be new. Input banks, labels, checkpoints, frozen packages,
shared environments and legacy evidence are read only. No formal training or
real scoring is invoked. All reported equivalence still requires separate
strict acceptance; event count is not a quality metric.
