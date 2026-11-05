# Exact geometry-reuse audit

2026-11-05. Read-only examination of the archived release. No server access,
GPU execution, shared source edits, or performance/quality acceptance is claimed.

## Projection is already a temporal stack

`gpu_interpolator.py:285–334` creates one alpha stack and calls
`project_endpoint_to_alpha_stack_torch` once per endpoint. There is no Python
377-times projection loop to remove. `gpu_geometry.py:143–215` already evaluates
all alpha cameras/objects using batched matrix operations.

The two endpoint calls redundantly compute the same camera pose stack and
camera inverses (`gpu_geometry.py:173, 207`); object pose stacks are also repeated
(`:192`). They can be precomputed once and passed explicitly. Source pixel grids,
source camera/world point conversion, masks, source object inverse and homogeneous
source points are constant within an interval and can also be prepared once.
However, these operations currently occur once or twice per whole 377-stack,
so this is unlikely to explain a 30–50x gap by itself. Changing matrix contraction
shapes or operation ordering may change final bits; benchmark only after strict
deterministic projection/frame/event regression.

## Larger immediate opportunity: remove unused histogram bound

`gpu_geometry.py:456–459` concatenates all bilinear candidate target indices,
builds a `bincount(minlength=total)` and transfers its max to the CPU solely to
bound the number of depth-layer iterations. The existing loop already breaks
as soon as all candidates are resolved (`:495–497`). Valid candidate z is finite
and positive; every iteration consumes at least one remaining nearest candidate
per active target because the depth-tie tolerances are nonnegative. Therefore
the histogram is not needed for termination or correctness.

Replace `for range(max_layers)` with `while True`, keeping the existing break,
and remove the `pixel_ids`/`bincount` allocation. No image, mask, source ordering,
layer arithmetic, noise draw, sample count or threshold is changed. At
T=377,H=260,W=346, `total=33,914,920`; the int64 histogram alone is about259MiB.
The concatenated int64 candidate indices can approach1GiB for four candidates
per source. This is the lowest-risk first candidate.

The revised builder statically preserves the entire original bounded function
under a private reference name. For NaN/Inf depth-tie tolerances, the optimized
function selects that unchanged reference: finite nonnegative tolerances are
essential to the progress argument, and NaN comparisons could otherwise make
the while-loop never consume a candidate. Negative finite values retain the
original ValueError. No runtime source execution or global function replacement
is used.

## Additional exact zero-contribution pruning

`gpu_geometry.py:489–493` computes `accepted=min(layer_weight,1-coverage)`.
Once coverage is exactly1.0, every later `accepted` is0. Those pixel candidates
can be removed from `unresolved`, while retaining initial occupancy, z-buffer,
first-layer surface ID/consistency and all original composition rules.

Never prune merely because coverage reaches0.995: residual contributions can
still change intensity and normalization. The threshold must be exactly1.0.

There is an IEEE corner case: an extreme finite HDR source may overflow an
FP32 layer sum, causing the old expression `inf*0` to produceNaN; skipping it
would change fallback behavior. The prepared optional candidate conservatively
disables pruning for NaN, negative, non-FP32 or potentially overflow-prone input.
It bounds all possible per-layer positive contributions by `4*total*max_source`
with an8x safety margin. Thus ordinary nonnegative HDR gets the pruning candidate,
while unusual inputs retain exactly the histogram-free original layer arithmetic.

Removing other pixels' scatter entries may alter default CUDA atomic scheduling,
so the mathematical argument is not enough for a bitwise release claim. Require
deterministic backend checks, actual sensor/event checks and formal loader checks.

## Prepared offline builder

`geometry_reuse_candidate.py` writes a new static module and a build hash receipt.
Modes:

1. `histfree`: no concatenation/histogram; same actual depth-layer arithmetic.
2. `prune-full`: histfree plus guarded removal of fully-covered zero-contribution
   pixels. Original coverage/surface/visibility rules remain.
3. `deadmix`: in coverage-time composition, do not compute the ordinary
   log-linear expression that is immediately overwritten by `conditional_mixed`.
   The grayscale and RGB branches both retain the old expression verbatim for
   `same_surface_log_linear`; all other composite paths and diagnostics remain.
4. `histfree+deadmix`: combine the first and third independent no-output-work
   removals.
5. `prune-full+deadmix`: combine guarded pruning with both removals.

Example (use a fresh isolated candidate destination):

```text
python geometry_reuse_candidate.py --source FROZEN/gpu_geometry.py --target NEW/gpu_geometry.py --mode histfree
```

Test requirements: compare raw geometry tensors and every diagnostic mask on
identity/motion/occlusion/unknown/invalid-depth/HDR fixtures; exact original and
candidate events and sensor state/RNG under deterministic execution; all three
formal loaders; native-auto timing and peak memory with unchanged sampler,
bandwidth, noise, thresholds, source physical cache and CPU/GPU timing boundary.

The builder was prepared without a local runnable Python environment; root must
syntax-check/build/run on the authorized server before accepting it.

## Splatter-only temporal chunk candidate (builder prepared)

Compute all377 projected `uv/z/valid` time samples exactly as the current stack
projection does. Only inside `flatten_endpoint`, after creating the flattened
M=B*T views, sliceM into consecutive chunks16 or32 and call the same coverage
splat on each chunk, including its matching optional surface-ID slice. Concatenate
the returned image/visible tensors and every diagnostic along dim0. Keep the
outer377-sample composition, alpha handling, endpoint overrides, fallback and
sensor call exactly as they are. No alpha or sensor sample is omitted.

The per-frame target domains are independent. Batch offsets change from absolute
M offsets to local chunk offsets, but that is only index relabeling. Visibility,
coverage, support, initial z-buffer, first-surface IDs and consistency must all be
returned and concatenated without loss. The same candidate order is retained
within each target. Endpoint0 chunking should complete before endpoint1 as in the
current code; do not interleave sensor state updates into geometry chunks.

Peak projection arrays remain full size, but layer scratch/candidate arrays
reduce approximately from377 frames to16/32 frames. This could reduce memory
traffic substantially, at the cost of more kernel launches and host layer-loop
checks. Measure rather than assume a speedup. The global per-stack layer loop
currently runs until allM frames resolve, so chunking may also stop easy chunks
earlier while difficult frames keep their full processing.

CUDA scatter numerical ordering can change with input shape. Even deterministic
algorithms may choose different reduction kernels for different sizes. Therefore
mathematical frame independence is insufficient for a strict bitwise claim;
compare image/diagnostic tensors including all boundaries and NaN masks, then
raw sensor events/state/RNG and formal task inputs. Overflow, NaN or signed inputs
need explicit regression because stopping an already-empty chunk can remove later
zero updates to a NaN value. Use unmodified full-stack reference fallback for
any unsupported numerical condition if exact equality cannot be established.

The offline `geometry_temporal_tiling_candidate.py` builder now implements this
isolated candidate. Give it a frozen or histfree+deadmix source and a new target:

```text
python geometry_temporal_tiling_candidate.py --source COMBO/gpu_geometry.py --target NEW16/gpu_geometry.py --chunk-size 16
```

Chunk choices are16/32. The appended outer-warp argument defaults to the chosen
chunk size; explicitly `coverage_chunk_size=None` selects the unchanged full-stack
dispatch for direct component comparisons. All other functions must be AST
identical to the supplied source. All diagnostic keys/tensors are concatenated.
Noncoverage dispatch and `need_diag` are unchanged. FP32, finite, nonnegative,
conservatively overflow-safe inputs and finite tolerances can tile; all other
numerical conditions fall back to the supplied full-stack function. The builder
writes a SHA receipt, marks quality tests unrun, and refuses existing targets.
Root must compare both chunk sizes with the same full-stack input and deterministic
flags before integrated use. Local Python is not installed; no runtime acceptance
is claimed.

## Exact paired-endpoint projection reuse (builder prepared)

`geometry_pair_reuse_candidate.py` creates two new files from supplied geometry
and interpolator sources. Geometry source can already include histogram/deadmix
or temporal-tiling candidates. It adds a private, explicit `_shared_geometry`
parameter to endpoint projection and creates a fresh cache for the original
uv0/uv1 pair. The caller discards the cache immediately after the two calls.
No persistent cache, tensor hashing, pose approximation, or alpha removal occurs.

The following original tensor results can be shared without changing their
operations or values: camera SE3 stack (original geometry line171), camera inverse
stack (206), same-size FP64 pixel grid and homogeneous ones (173–179), and each
object SE3 stack (194/200). The second call uses the exact tensors produced by
the first; object stacks are lazy so objects absent in the first segmentation
still get computed by the second call. Both endpoints retain their original
source camera, depth, world points, masks, source-object inverse, motion, camera
projection, validity and final FP32 casts.

The cache is disabled if endpoint sizes differ or any physical input requires
gradients. This preserves the original independent autograd graphs for possible
future differentiable users. The production event-generation physical inputs
do not require gradients. Individual direct projection calls default to no cache.
All other geometry functions are AST identical. The builder refuses an existing
destination directory and hashes input/output files.

```text
python geometry_pair_reuse_candidate.py --geometry-source COMBO/gpu_geometry.py --interpolator-source FROZEN/gpu_interpolator.py --target-dir NEW_PAIR
```

This is probably a smaller performance opportunity than coverage scratch because
the reused camera/object arrays contain only377 transforms. The grid is shared,
but all per-pixel projection and splat work is deliberately retained. Test exact
uv/z/valid for both endpoints with/without explicit cache, actual events/sensor
state and formal loader inputs, then measure end-to-end time. No Python or GPU
test has been run locally, and no speed/quality acceptance is claimed.
