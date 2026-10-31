# Genesis robot scene: repair before/after demo

This presentation replays two already captured Genesis robot-interception trajectories with the same camera and timestamps. Only the intermediate image reconstruction changes: the archived legacy warp is compared with the diagnostic front-to-back coverage candidate.

## What is shown

- `episode_004_center.mp4`: center trajectory.
- `episode_005_plus_55mm.mp4`: the same task with a +55 mm lateral offset.
- `overview_before_after.png`: Genesis intensity reference, old reconstruction, candidate reconstruction, and per-episode summary.
- `spatial_error_overlay.png`: red marks pixels whose absolute log-luminance error exceeds 0.2 at any of the ten saved 1 ms queries.
- `per_query_metrics.csv` and `demo_metrics.json`: counts and image-error measurements for each saved query.

The video slows the 1 ms sequence for viewing. Its frames remain the original saved samples. Each event panel runs the same ideal, noise-free DVS response independently over that source's complete 10 ms history.

## Result

| Trajectory | Direct-HDR events | Old reconstruction | Coverage candidate | Pixel-times above 0.2 log error, old → candidate |
|---|---:|---:|---:|---:|
| Center | 625 | 6,184 | 407 | 8,555 → 665 (92.5% resolved) |
| +55 mm | 939 | 4,313 | 682 | 7,851 → 1,377 (84.4% resolved) |

The old reconstruction reproduces the archived intermediate images exactly (maximum absolute difference 0). The candidate greatly reduces the visible edge/depth-competition error and event burst, while residual error remains. In particular, the candidate does not recover all Direct-HDR events; this demo demonstrates one repaired numerical failure mode, not a complete simulator repair or a production-core switch.

## Scope and limits

The images and physical states were captured previously from Genesis and are replayed offline here. This run did not launch a new renderer or physics rollout. Direct-HDR frames are exact renders at the logged sample states; they do not establish the radiance between those samples. The red error overlays are post-hoc visualization only and are not used by reconstruction or event generation. The candidate is diagnostic-only; frozen core remains unchanged. Dynamic photometric residuals and broader-scene behavior remain open.

## Reproduction

From the repository root, with the same Python environment and archived inputs:

```bash
python benchmark/robot_closed_loop/diagnostics/create_r1_5_genesis_scene_before_after_demo.py \
  --config benchmark/robot_closed_loop/configs/r1_5_genesis_scene_demo_20261101a.json \
  --cache-root benchmark/robot_closed_loop/results/r1_failure_chain_diagnosis_20261019a/native_hdr_cutoff059 \
  --output benchmark/robot_closed_loop/results/r1_5_genesis_scene_demo_20261101a_repro
```

The script refuses to overwrite existing output. Recorded execution code commit: `54ecff4f05e029c029389dca7e01c1465d95bf3e`. Configuration SHA256 and imported module hashes are in `demo_metrics.json`.

Input SHA256:

- Episode 004 source cache: `0525c18c8d89f5831ef0cb71ac9e3b6b130d60579daf374f06d5cfcee5b879d7`
- Episode 004 rendered-frame evidence: `b3afe5b5badee3d2606d343bb750aa970887928c41b8001a5fc4e9ff32b9c321`
- Episode 005 source cache: `91c1397642e317d6bb51150b4107c518b52e4fa9fcf5f1caf6a2a91315c7d434`
- Episode 005 rendered-frame evidence: `f161079a24c1bf9fee838a6fbb2a0fa5d785435b116f0ea5628e6efbba497236`

`SHA256SUMS` covers the generated report, table, stills, and videos.
