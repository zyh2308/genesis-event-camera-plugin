# R0.6 small Direct-HDR event-controller rollout

Three 3-second episodes were run in g1 using the actual 2 kHz causal event path, fixed predictor/controller, and strict R0.5 scorer. All episodes used 0.5 ms physics, the same fixed side camera, 100 Hz high-level controller, and no GT estimator input. The trajectories were a balanced smoke sweep over speed and lateral edge/center positions, not a success-rate estimate.

| Episode | Speed | Launch y | Contact | Lift | Strict success | Dynamic visibility minimum | 2 kHz state position P95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.30 m/s | -55 mm | no | no | no | 0.0 | 161 mm |
| 1 | 0.40 m/s | 0 mm | 0.6285 s | no | no | 1.0 | 217 mm |
| 2 | 0.50 m/s | +55 mm | no | no | no | 0.0 | 500 mm |

Only the center trajectory contacted the cube (1/3); none lifted or met strict success (0/3). In that one contact episode, the controller had a usable post-contact estimate for 274/2,374 control frames (11.54%). Its post-contact planar position P95 error was 275.5 mm and velocity P95 error 7.29 m/s; the lift gate was eligible 0/2,374 times. The accepted post-contact states themselves were fresh by age (P95 9.5 ms), which again shows that age alone does not establish accuracy. The two edge runs lost dynamic target visibility and did not contact. This failure pattern is consistent with the broad-ROI centroid returning robot/background motion, but three episodes do not isolate the causal contribution of perception and arm occlusion.

## Runtime and provenance

- g1 result directory: `~/Eventbased_WAM/repositories/genesis-event-camera-plugin-r0_6-b3dc69d/benchmark/robot_closed_loop/results/r0_6_event_controller_3sample_20261016/`.
- Actual run code commit: `30177915e0e260d6161feb48dddf087809268f73`.
- Scene config SHA256: `aa97f6f5a12af92561ef053407bb795bba2801582958b4ad1134bcd24497435a`.
- Seed block: `2026103700`–`2026103702`.
- Result `SHA256SUMS` SHA256: `90c4823bf1dfb6dec0b42ec7ee97d5241b718d6fed55c06a3f9746ba68290a1f`; every listed artifact was checked on g1 and after transfer. Raw event arrays are included in this small result batch.
- Full Python process: wall 662.89 s, user CPU 18,618.97 s, system CPU 79.01 s, maximum RSS 1,809,900 KB, exit 0. Episode rollout wall times were 196.44/182.08/193.98 s for 3 s simulated time; mean 190.83 s. Physics ran on CPU; Direct-HDR EGL used V100S GPU3. No GPU-time profiler was used.
- Steady HDR render mean was 7.47–7.51 ms/sample. DVS mean was 5.64–6.52 ms/sample at 1 kHz and 5.14–5.99 ms/sample at 2 kHz. These per-sample means omit startup, physics/control, event binning/model inference, and output costs; the full process wall above is the end-to-end cost for these three episodes. Do not scale it linearly to larger datasets.
- Launcher PID `2445286`, Python PID `2445290`; log SHA256 `12f9585d909d27184ef87083204ba1c2179ab01dd9033849bb42de1f56455b8c`; runtime SHA256 `109228a1fd87c4bcee0e10221e05ae59893c3c71d3acc141f92b3f4b3de954b8`; launcher SHA256 `49973e333315859564f7f747918147ef2ef2b2a063e4f5ba1b1af88ff66b0c73`; exit file contains `0` and SHA256 is `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`.

The event source directly renders linear HDR every 0.5 ms and feeds the shared plugin `DvsEmulator` (ideal threshold 0.2, no noise, infinite bandwidth). It bypasses Ours temporal reconstruction and V2E interpolation; the pixel response is shared and is not EVK4 calibrated. Evaluator segmentation and depth are not connected to the event source or controller. See `run_manifest.json`, per-episode `reference_metrics.json`, `state_control_trace.jsonl`, `event_state_error.csv`, and `rollout.gif` for detailed logs and representative outcomes.
