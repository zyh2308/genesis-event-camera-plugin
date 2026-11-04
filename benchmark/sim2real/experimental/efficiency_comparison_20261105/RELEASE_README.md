# EventGenesis GPU Stage 9 release

This is a versioned, explicit GPU entry point. The frozen 17-file reference remains byte-equivalent after canonical LF normalization. Running Sim-to-Real experiments and their existing input events are not switched to this release.

```python
# Add this release directory to PYTHONPATH, not the source package subdirectory.
from genesis_event_plugin.optimized import OptimizedGenesisPhysicsEventPlugin

# Supply the archived Stage 9 noise/sampler/plugin settings and the same
# timestamped native linear-HDR/depth/segmentation and SE(3) providers.
plugin = OptimizedGenesisPhysicsEventPlugin(...)
plugin.attach(scene, camera).start_episode()
print(plugin.optimization_status)
events = plugin.capture()  # NumPy Nx4: seconds, x, y, polarity
```

The `...` above denotes application parameters; a complete executable example and the exact profile accompany the benchmark/acceptance scripts. No frame or ground-truth information is added to downstream models.

## Supported fast path

- Exact complete archived Stage 9 sensor configuration, CUDA + Torch warp, `coverage_time_log_linear`, `hold`, 320x240 or 346x260 sensors.
- No camera margins or intermediate-state provider. Other settings explicitly retain the original reference path, with the reason available in `optimization_status`; they are not silently approximated or disabled.
- CPU and the original NumPy public interpolation interface retain their reference behavior. GPU interpolation has its own device entry point.
- Intermediate images/masks and full sensor state remain on the GPU. Diagnostics are computed and available through lazy Mapping/Sequence access or `materialize_diagnostics()`.
- Missing-projection radiance fallback retains original NumPy log/exp operations. Small host transfers are deliberate numerical compatibility work.

## Numerical contract

Image/warp precision and FP64 sensor state, original noise RNG draw ordering, threshold mismatch, bandwidth, leakage, shot noise, refractory period, visibility/support/unknown rules, sample density and event sorting are retained. The CUDA build uses `--fmad=false` and does not enable fast math. Sensor overflow/unstable cases restore RNG before exact reference fallback.

Same-frame sensor state/events and recorder payload have strict equality checks. The existing FP32 scatter geometry is nondeterministic in its final bits: full raw float64 timestamp arrays are not promised bitwise identical across complete repeated runs. The separate strict raw gate remains failed, with the original evidence retained; this release must not be described as a fully bitwise-deterministic simulator.

The release does not make an unknown pixel a true observation. Endpoint pose interpolation remains the same approximation as the reference. Faster computation is not evidence of improved physical accuracy or Sim-to-Real scores.

## Private CUDA build

Install no packages into shared environments. Use a compatible existing Torch/CUDA compiler and a private `EVENTGENESIS_GPU_BUILD_DIR`. The first supported capture loads/builds `eventgenesis_precision_sensor`; compilation is excluded from warm performance timing and logged separately.

```text
CUDA_HOME=<existing-toolkit>
CC=<compatible-c-compiler>
CXX=<compatible-cxx-compiler>
MAX_JOBS=2
TORCH_CUDA_ARCH_LIST=7.0
EVENTGENESIS_GPU_BUILD_DIR=<private-build-directory>
```

The archived g1 build uses the project's existing isolated compiler; `build_release.py` records the exact reproducible path and flags. Source and binary hashes are indexed. Other GPU architectures and Torch/compiler versions require rebuilding and acceptance checks.

## Scope

Performance claims apply to the five fixed scenes, specified substep spacing and GPU reported in the comparison. Native Genesis HDR adapter/recorder smoke verifies that boundary; no claim is made that the complete renderer is zero-copy or realtime. Memory is a tradeoff: the optimized path uses substantially more GPU memory than v2e in these tests.

The core is a single-environment adapter. It does not introduce a tested multi-environment Genesis or concurrent renderer API. Static GPU geometry no longer replaces process-global functions. Existing formal training still consumes its original offline events.
