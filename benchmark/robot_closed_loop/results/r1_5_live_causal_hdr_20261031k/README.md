# R1.5 actual-physics causal HDR integration smoke (run K)

## Result

Run `r1_5_live_causal_hdr_20261031k` passed on g1, using candidate source commit `7559893ea4168b979679ee32eac03806050a9b03`, Genesis 1.2.2, the verified `gec-robot/.venv-robot` HDR environment, and GPU1 RTX 3090 EGL rasterizer. The run advanced Genesis normally for two 0.300 s lateral crossings at `dt=0.5 ms`; no target pose or velocity was overwritten inside either episode.

The evaluator-only target trace confirms the physical motion. Episode one moved in world y from `-40.000 mm` to `+0.578 mm` (40.578 mm path); episode two moved from `+40.000 mm` to `-0.578 mm` (40.578 mm path). Median planar speed was 0.1353 m/s and P95 was 0.1485 m/s. The cube remained visible throughout, with 2,862–2,994 segmented target pixels per frame. Position/velocity traces are saved for verification and were not made available to the event-only controller.

## Causal interface checks

- 1,200 ordinary physics steps, 1,202 native HDR observations, 300 plugin capture intervals/controller reads, and 450 exact historical sample requests.
- Every interval resolved with `actual_state_samples`, five photometric anchors, and maximum anchor spacing 0.5 ms. Each requested image had already been captured at its exact physical timestamp before the plugin requested it.
- Both episodes returned a nonempty event packet at simulation time 2.000 ms. The event-only policy issued a close command at that packet and Genesis applied it on the next 0.5 ms step at 2.500 ms. This checks delivery and causal action ordering only; it is not a grasp test.
- Every event timestamp was no later than its packet's right endpoint. The packet became available at the right endpoint, and no future state, target GT, GT segmentation, or depth was passed into the policy.
- The common floor mask retained a 95th-percentile log-luminance change of 0 in these samples, while localized sets of up to 1,721 and 1,950 floor pixels exceeded 0.2 relative to the initial image. This shows localized floor appearance changes during the moving-target episode; this run did not use a shadow-off causal control, so those pixels are not attributed solely to shadows.

## Cost and limits

| Component | Measured cost |
|---|---:|
| Genesis initialization | 1.593 s |
| Scene build | 57.759 s |
| 1,202 native HDR renders including readback | 16.083 s total; 13.38 ms mean per observation |
| Strict sample-provider lookup | 9.1 ms total for 450 requests |
| Plugin captures, including interpolation and DVS | 108.336 s total for 300 intervals; 361 ms mean per 2 ms simulated interval |
| Two episode loops | 128.186 s total for 0.600 s simulated time |
| Whole process before output hashing | 189.843 s |

This is an offline, step-ordered integration diagnostic; it is far slower than wall-clock real time. The 0.5 ms samples are acquired by rendering every ordinary physics step, then consumed when the strict plugin reaches each 2 ms right endpoint. The provider's 9.1 ms is only lookup overhead, not the cost of obtaining those images. The run does not claim that future images were predicted, or that the plugin can run on a real robot at this cadence.

## Reproduction and evidence

The run output directory and logs are available on the shared g1 `/home` NFS at:

`/home/zhaoyuhan/Eventbased_WAM/runs/robot_scenes/genesis_event_camera_plugin/r1_5_live_causal_hdr_20261031k/`

The source/config command and reservation/PID are recorded in `logs/preflight.txt`, `logs/resource_allocation.json`, and `output/run_manifest.json`. Re-run using a new run ID after the launcher resource gate passes:

```bash
bash source/benchmark/robot_closed_loop/diagnostics/launch_r1_5_live_causal_hdr_remote.sh \
  g1 <new-run-id> 7559893ea4168b979679ee32eac03806050a9b03 1 \
  GPU-0262d545-e9bd-7fcb-66e5-2ef81465c460
```

Main evidence files:

- `output/causal_trace.jsonl`: packet availability, event timestamps, requests, controller reads, and next-step action times.
- `output/target_physics_trace.csv`: evaluator-only position/velocity at every 0.5 ms state.
- `output/frame_metrics.csv` and `output/*_keyframes.npz`: target visibility and floor appearance metrics plus representative actual HDR images.
- `output/raw_events.npz`: raw DVS events for both episodes.
- `output/summary.json`, `output/run_manifest.json`, and `logs/`: metrics, environment/source hashes, process exit and resource record.

Remote evidence archive SHA256:

`f3d71e0b728f2606eb1b70646b7a5eb61bf7bf4d29a9d9a413a2b8518db5f851`

The archive is retained remotely as `evidence.tar.gz`; `output/SHA256SUMS` validates the raw output files and this result directory's `SHA256SUMS` validates the locally versioned evidence. Genesis emitted a Quadrants cache-lock warning from the shared cache; this run did not clear or modify that cache.
