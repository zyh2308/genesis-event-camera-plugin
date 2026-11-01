# R1.5 exact 0.5 ms state replay (full 9 episode batch)

This result is the renderer-free prerequisite for generating matched strict-sampled and Direct-HDR inputs. It does **not** contain new HDR images, events, or localization results.

## Result

- Status: `PASS` for all 9 episodes.
- Replay code commit: `ad58544` (`exp/robot-closed-loop-v1`).
- Frozen core/profile selected for subsequent source generation: tag `r1.5-strict-sampled-20261031`, profile `scientific_strict_sampled_reference`, profile SHA256 `3525cad9e84982a2b835b1b7d32428b94be24508fe28faf9f0d79002dfb10818`.
- Physics timestep and saved trace interval: 0.5 ms.
- Episodes: seeds 2026104130–2026104138, speeds 0.30/0.40/0.50 m/s, starts at left, center, and right offsets.
- Every saved state at the legacy 1 kHz timestamps and every post-launch 100 Hz HDR-cache timestamp matched exactly: max absolute delta was 0 for target position/velocity/orientation/angular velocity and Panda qpos.
- Initialization + scene construction + episode replay took 181.21 s before artifact hashing. This was a CPU Genesis run; no camera rendering, event generation, training, or GPU use occurred.

## Inputs and code identity

- Scene config SHA256: `aa97f6f5a12af92561ef053407bb795bba2801582958b4ad1134bcd24497435a`.
- Shared-control config SHA256: `3d289c2bef1c562b26f9b713c2a826b813a286a1f7e21f9ead0a01c63c5e0d80`.
- Experiment config SHA256: `cb62cc6988de1682bcb5f8686e78df703134383b8f929dca86ca85571b22527c`.
- Full summary SHA256: `9b8d4275aaf3203d2cd1127dd3a5c1b07c007df43e659a1b5d8b17f43eff6b6b`.
- Run-manifest SHA256: `3e2ce1d7751d23f81e86fd05517a8e1e37e740f2cf602b7e2be0f9d4e41cd8d4`.
- Original g1 full-tree SHA256 list (renamed `REMOTE_SHA256SUMS`) SHA256: `5910ee81fb1f0e6e136cc5e6aa781931e0a429199729c4e3d3c533db2c70be6d`.
- Full per-episode raw state traces and launcher logs remain on the shared g1 filesystem at:
  `/home/zhaoyuhan/Eventbased_WAM/runs/robot_scenes/genesis_event_camera_plugin/r1_5_localization_recovery_20261101a/state_trace_full9_20261101a/`
  and the sibling `state_trace_full9_20261101a.logs/`. Their hashes are listed in the included `SHA256SUMS` and `run_manifest.json`.

The compact tracked copy contains the run summary, manifest, the original g1 full-tree hash list, and detailed summaries for episodes 004–006. Local `SHA256SUMS` covers only the compact tracked files; the raw state traces remain on g1 and are covered by `REMOTE_SHA256SUMS` and `run_manifest.json`.

## Reproduction

Use the command recorded in `docs/coordination/robot_status.md` under “完整九集状态回放命令”. The run is safe to repeat only with a new output directory; the capture script refuses to overwrite an existing result.

