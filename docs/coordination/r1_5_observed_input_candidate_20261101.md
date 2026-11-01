# R1.5 actual-HDR observation input candidate

This candidate adds `GenesisPhysicsEventPlugin.capture_observed` to the existing R1.5 core. It separates ingestion of actual time-stamped HDR observations from reconstruction between sparse camera frames. The original `capture` reconstruction path and its sampler remain available.

The public entry point requires `strict_fidelity=True`, nonempty observation provenance, a monotonically increasing observation timestamp, and an availability timestamp within the current scene time. Each gap must be no larger than the configured `max_photometric_interval_s`. Missing samples are rejected before the DVS history changes. Observation and reconstruction modes are separated at episode boundaries.

The observation array must have a floating dtype, and the plugin must declare linear input space. Integer display pixels and an sRGB declaration are rejected rather than silently treated as native HDR.

```python
plugin = GenesisPhysicsEventPlugin(
    noise_config=sensor_config,
    input_space="linear",
    strict_fidelity=True,
    max_photometric_interval_s=0.0005,
    motion_state_provider=motion_state_provider,
)
plugin.attach(scene, camera)
plugin.start_episode()
# Initial observation, then every exact native 0.5 ms physical state:
events = plugin.capture_observed(
    native_hdr,
    timestamp_s=t,
    provenance="Genesis native scene-linear HDR at exact physical state",
    available_at_s=t,
)
```

The caller preserves the raw renderer radiance. Display tone mapping belongs only in visualization. The caller also supplies actual observations; provenance identifies the renderer/cache evidence and is not a substitute for that evidence. The supported time resolution is the recorded sampling interval. Input between recorded observations is not reconstructed by this entry point.

## Development experiment

The research execution uses the nine original R1 training rollouts, whose 0.5 ms physical state replay was already verified against the archived 1 kHz state traces and 100 Hz camera states. All native HDR images are freshly rendered on g1 from those exact states. The original t=0 initialization is retained. The discarded post-launch initialization hypothesis is not part of this candidate.

The training split is `[0,1,2,3,7,8]` / validation `[4,5,6]`, `heatmap_v2`, seed 606, 40 epochs and two CPU threads. Training and the old 581-query Direct-HDR development regression have disjoint episode seeds. The regression reference is not used for fitting or checkpoint selection.

An independent continuous DVS history processes the same actual HDR samples. Raw events and every causal five-bin training window are compared before training. Identical inputs permit sharing one learning positive control with the observed-input candidate, without duplicating training. The event profile is the explicit ideal, uncalibrated development profile: threshold 0.2, no noise, no bandwidth limit, physical-log input. Existing real_v2 defaults are unchanged.

## Validation and provenance

The controlled API checks cover an increasing/decreasing brightness ramp, missing-frame rejection and recovery, causal availability, invalid input rejection, episode reset, mode separation, and the existing default capture path on a static scene. The ramp produces 108 events exactly matching the independent pixel-response control. Full trajectory results and localization metrics are archived separately with input/module/checkpoint hashes.

```bash
python benchmark/robot_closed_loop/diagnostics/check_observed_hdr_capture.py .
```

- Parent core: `90f792d166f000ef4f9e072a97633cc4ad39c8b7`.
- Only changed core file: `genesis_event_plugin/physics_plugin.py`.
- Existing frozen tags and the main branch are unchanged.
- Shared g1 output: `/home/zhaoyuhan/Eventbased_WAM/runs/robot_r1_5_research_repair_demo_20261101a/`.

This candidate establishes an explicit dense native-HDR correctness path. Its measured rendering cost and downstream localization are reported as such; sparse-warp efficiency and real-camera performance require their own evaluation.
