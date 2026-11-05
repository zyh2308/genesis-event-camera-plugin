# GPU event simulator primary-source review

Read date: 2026-11-05/06. Research only: no source execution, SSH, GPU, environment change, benchmark or frozen asset mutation.

Scope: the present bottleneck is that the sensor-driven 377-step schedule also requests full-image geometry reconstruction at every step. This review assesses implementation mechanisms, not a replacement sensor contract. Existing strict raw/loader failures remain failures; current index-reuse v1 qualification is not extended by this literature review.

## Primary source set and fixed versions

| Family | Paper / project | Official code version reviewed |
|---|---|---|
| Vid2E / GPU ESIM | [CVPR 2020 paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Gehrig_Video_to_Events_Recycling_Video_Datasets_for_Event_Cameras_CVPR_2020_paper.html) | [uzh-rpg/rpg_vid2e](https://github.com/uzh-rpg/rpg_vid2e/tree/ecbb11a9345bb9d31b4b691e7d82965da4401345), SHA ecbb11a9345bb9d31b4b691e7d82965da4401345 |
| OpenEB | [official GPUEventSimulator](https://github.com/prophesee-ai/openeb/blob/9003b5416676e78ba994d912087486cfa94fae73/sdk/modules/core_ml/python/pypkg/metavision_core_ml/video_to_event/gpu_simulator.py) | prophesee-ai/openeb SHA 9003b5416676e78ba994d912087486cfa94fae73 |
| Ziegler et al. | [ICRA 2023 Real-time Event Simulation with Frame-based Cameras](https://cogsys-tuebingen.github.io/realtime_event_simulator/), [paper](https://arxiv.org/pdf/2209.04634) | [cogsys-tuebingen/event_simulator](https://github.com/cogsys-tuebingen/event_simulator/tree/b85ae4c583d56022f1ed49ed597de739fefbc29d), SHA b85ae4c583d56022f1ed49ed597de739fefbc29d |
| EVIS | [July 2026 v1 paper](https://arxiv.org/html/2607.08098v1) | [spikelab-jhu/isaac-sim-event-camera-plugin](https://github.com/spikelab-jhu/isaac-sim-event-camera-plugin/tree/d6fe47923fad09c4bfa2fd2cb13f82520114cfae), SHA d6fe47923fad09c4bfa2fd2cb13f82520114cfae |
| EsaacSim | [August 2026 v1 paper](https://arxiv.org/html/2608.08522v1) | [ibugueno/esaacsim](https://github.com/ibugueno/esaacsim/tree/2e1ae1e9b64e4d046d52e98f38d5c14cb515acb9), SHA 2e1ae1e9b64e4d046d52e98f38d5c14cb515acb9 |

Versions are public repository branch heads observed during this review, not necessarily the exact paper submission versions. Current GPU ESIM and FILM additions must not be attributed indiscriminately to the original 2020 paper.

## 1. Vid2E: GPU-resident ordered temporal processing and analytic crossing

Verified code: esim_torch/src/esim_torch/esim_cuda_kernel.cu:18-57,61-123; esim_torch/src/esim_torch/esim_torch.py:51-92.

One CUDA thread owns a pixel and traverses input time in order. The first pass computes event counts and threshold reference states; prefix sum assigns output ranges; a second pass emits events. For each threshold crossing the timestamp follows a direct linear-interpolation formula, not iterative root finding. The wrapper supports streaming and carries the previous frame/time and reference between calls.

Transferable: pixel ownership, ordered temporal loops, persistent state and variable-length count/scan/fill output. This can reduce host launches and temporary frame stacks. It does not by itself reduce geometry work.

Qualification limit: this interface has positive/negative thresholds and refractory period, not our complete low-pass/leak/shot/RNG contract. Integer nanosecond representation is not proof of timestamp accuracy. The reference/time arithmetic and refractory inequality differ from our contract and cannot be copied to obtain strict equivalence.

## 2. OpenEB: separate photoreceptor filter, persistent pixel state, optional direct voxel

Verified code under sdk/modules/core_ml/python/pypkg/metavision_core_ml/video_to_event/:
- gpu_simulator.py: GPUEventSimulator, _kernel_call, get_events, event_volume_sequence, dynamic_moving_average.
- events_kernels.py:42-118: per-pixel time loop;69-85 threshold crossings;74-75 direct linear crossing time;89-113 shot/leak behavior.
- cutoff_kernels.py: _cuda_kernel_dynamic_moving_average.

GPUEventSimulator retains reference/filter state, previous images, timestamps, per-pixel thresholds and RNG state on device. get_events uses count then prefix sum then fill, replaying the same sampled random states and committing persistent state only on the fill pass. Direct voxel output avoids an intermediate AER list when that output is explicitly requested.

Its photoreceptor pass is a separate two-stage low-pass recurrence with a clamped Euler coefficient calculated once per input frame. It is not proof that our current sensor integration can safely use a larger step. Shot noise is based on probabilities over the input interval; leak lowers the reference each interval.

Transferable: separate sensor module from image production; preserve chronological local state; fuse cheap temporal arithmetic after geometry; if two passes are used, replay noise without committing twice. Direct voxel is a separate representation contract and cannot silently replace raw events/loader output.

## 3. ICRA 2023: activity selection has narrow assumptions

Verified code: src/SparseInterpolatedEventSimulator.cpp:75-129. It takes an absolute 8-bit frame difference, thresholds a mask, computes sparse optical flow only at selected points, scatters those points at uniformly interpolated times, then subtracts/thresholds consecutive synthesized frames.

Paper section III explicitly uses linear differences rather than logarithmic contrast and excludes sensor noise/artifacts. Sparse processing helps when few pixels change; whole-camera motion can erase that advantage. The difference-interpolation variant assumes constant linear velocity and unchanged perspective/appearance.

Transferable only as a research hint: use a conservative activity/visibility certificate to allocate expensive reconstruction, with a correctness fallback. Endpoint difference alone cannot prove a pixel or interval irrelevant under low-pass memory, accumulated threshold state, illumination pulses, shot noise or disocclusion.

Do not import its real-time/multiple-orders-of-magnitude claim, its simple threshold mask, or its interpolation count as evidence for our full sensor contract. This is not a reason to repeat the already-rejected tile/prune experiment.

## 4. EVIS: rendering/synthesis split, but different event semantics

Verified code: dvs_gen/dvs/processor.py:40-89, particularly84-85. Optional bandwidth/noise is called before/after the signal threshold logic.

The paper and code separate sparse rendered keyframes from motion-vector synthesized frames and process environments/cameras in batches. Intermediate geometry is still produced at the chosen synthesis cadence; batching does not remove its arithmetic. Bidirectional warp uses depth-guided softmax splatting, B-first coverage and A-only hole filling.

Crucial qualification limit: a triggering pixel latches its reference to the current sampled log value, with at most one event of a polarity per sample; events use the supplied sample time. The paper's optional finite-bandwidth recurrence uses an intensity-scaled clamped Euler coefficient; its sampled noise probabilities depend on synthesis-step duration. This is not equivalent to our threshold-by-threshold event production or full noise/occlusion contract.

Transferable: modular low-rate rendering and high-rate sensor backend, batching of independent cameras. Not a drop-in algorithm or a justification for restoring a previously rejected warp pipeline.

## 5. EsaacSim: on-demand sampling inside a temporal CUDA loop is relevant

Verified code:
- src/isaac_event_sim/sensors/frame_motion_fused.py:13-18,35-57,62-89: batch grid_sample for every intermediate frame, followed by ordered backend updates.
- frame_motion_cuda.py:38-157: one thread per pixel, loop over temporal steps, bilinear gather followed by log/threshold/reference update, without materializing all intermediate frames.
- event_backends.py:105-108 explicitly says refractory is gated once per frame rather than per sub-event.
- event_backends.py:566-571: PhysicalEventSimulator is an unimplemented placeholder.

The second pattern is relevant to avoiding intermediate time stacks. Our frozen stage9 csrc/fused_sensor.cu already has one thread per pixel and an ordered for t<T sensor recurrence; this is existing local capability, not a proposed new achievement. The remaining opportunity is that gpu_sensor.py still pre-materializes frames/log/eps/leak/on/off/noise_ts as T-by-H-by-W tensors. A candidate could investigate fusing cheap photometric reconstruction into that existing sensor kernel, with explicit arithmetic, RNG and state equivalence checks.

Critical code limitation: frame_motion_cuda.py:129/134 counts threshold multiples,141 advances the reference by count*C, but148-154 reserves and emits one event only. A multi-crossing step therefore is not equivalent to emitting count events. That kernel's arguments also omit low-pass, shot and leak states. The repository's reported rate cannot certify our full model.

The paper uses fixed uniformly spaced frame-gap samples with motion-vector backward warping and linear-image fallback. It does not establish an error-certified geometry/sensor multirate method for our occlusion rules.

## Assessment for our branch

These sources support two distinct directions; neither has been benchmarked here.

1. Existing capability: frozen stage9 already has the ordered per-pixel sensor kernel. The concrete engineering question is whether cheap photometric reconstruction and/or its intermediate state can be consumed inside that kernel without pre-materializing the T-by-H-by-W frames/log/eps/leak/on/off/noise_ts stacks in gpu_sensor.py. Geometry, all377 sample times, complete noise and chronological state behavior would initially remain. Matching sample count and RNG alone does not establish equivalence; input values, arithmetic order, RNG consumption, reference, refractory and outputs must all retain their strict gates. Materialization and sensor/geometry timings must first show that this is worth attempting.

2. The larger potential reduction is a separate geometry schedule and a sensor schedule. Only geometry reconstruction would use fewer evaluations; all377 sensor updates and complete noise would remain. A conservative validity/error certificate must detect changes in coverage, depth ordering, radiance, pixel footprint and disocclusion. Failed certificates fall back to the existing geometry evaluator. This changes the sampling/reconstruction contract unless exact equivalence is proven, so it requires a versioned research candidate and user direction before substitution.

The literature does not justify changing epsilon, cutoff, time quantization, noise rates, refractory behavior or strict tolerances to reduce N. Nor does it prove that simply interpolating between ~16-17 geometry samples preserves our raw events. No new performance number is claimed.
