# Public-data Sim-to-Real status — 2026-10-29

**Latest handoff:** [Stage 7 execution update](STAGE7_EXECUTION_UPDATE_20261029.md). This is the current status; the superseded 2026-10-26 snapshot is preserved at `reports/archive/sim2real_status_20261026.md` and its corresponding handoff.

**Latest live resource refresh:** g1 2026-10-29 06:02:30 UTC; g3 06:02:51 UTC. g1 GPU 0 has two `gec-robot` processes and GPUs 1–5 are occupied. g3 `nvidia-smi` reported an unknown device handle for GPU `0000:56:00.0`, so a complete safe allocation could not be confirmed. No new job was launched in this refresh.

| Line | Current verified state | Next actionable step |
|---|---|---|
| Reconstruction — E2VID+ / HQF | Both seed-0 synthetic-best models were evaluated on all 14 HQF sequences under the frozen protocol. EventGenesis equal-sequence LPIPS-Alex 0.570738 / SSIM 0.255283 / PSNR 11.6348 dB / MSE 0.086147; V2E 0.582220 / 0.250035 / 12.0742 dB / 0.080667. Single-seed real result only; no checkpoint selection or tuning from it. HS-ERGB remains an extension. | Archive and report the fixed result; no repeat scoring or tuning. |
| Optical flow — STE-FlowNet | Paired fixed-window GT fit: 200 updates/arm on `s5q_02_train/dt1_000`; independent `s5q_02_val` uses all 66 windows. Both reduce fixed-sample supervised objective, but independent full-valid and common event-active EPE are worse than zero flow. No useful generalization established. In-process checkpoint reload matches exactly; standalone process resume is not yet verified due one escaped-variable path bug and subsequent CUDA initialization failures. | When a GPU and CUDA process are safely allocatable, fix the verifier argument and run restore + one optimizer update only. Then create/check a more diverse independent motion scene before any longer flow training. |
| Pose — SingleShotPose / E-POSE | 18/18 data archives are present. Two final MAT windows exactly match raw HDF5 event windows (717/717 and 729/729). Exact CAD/K/pose checks identify an official shared-depth 2D keypoint label bug (~16.7 px mean projection error). No pose training, independent rollout/PnP validation, or real score was run. | Rebuild keypoint supervision from CAD + 6D pose + K, audit the corrected synthetic contract, then run paired synthetic pilot when resources are safe. |

Frozen core remains `e79e656ac00f7847b98db9e12bf43b093f3afff1`. Real data is test-only; no real score was used for model or checkpoint selection. Large artifacts remain on g1 and are indexed in `evidence/20261029-execution-refresh/evidence_index.json`.
