# Stage 4 paired flow training: live status

Updated: `2026-10-17T19:09:12Z` (g1 UTC). Both corrected full-budget resumes are active. The latest metric snapshot has EventGenesis at update 315 and V2E at 321; both have finite loss, gradient, and 70 finite gradient tensors. Latest resumable checkpoints are at update 300.

## Run identity

- Repository: `/home/zhaoyuhan/Eventbased_WAM/repositories/gec-sim2real`
- Branch: `exp/public-sim2real-v1`
- Launch and oracle commit: `1115a8f6e2f5bac82a4b7428b37ec5006342355f`
- Frozen simulator core: `e79e656ac00f7847b98db9e12bf43b093f3afff1` (unchanged)
- Campaign: `20261017-stage4-flowdt1-seed0-corrected-v5`
- Training source/config/data and optimizer protocol are unchanged from the validated pilot; retry output is isolated under `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261017-stage4-flowdt1-seed0-corrected-v5/full_training_retry1`.
- Training remains synthetic-only; flow GT is excluded from the optimizer; inference input is event-only; real-data scoring is disabled.
- Supervisor: tmux `stage4-flow-corrected-20261017-v7`, PID `3590105`. EventGenesis PID `3594965` on GPU 2; V2E PID `3594967` on GPU 4.

## Correctness and launch gate

The original full continuous-versus-resumed trajectory comparison failed and remains recorded as a failure. It observed matching sample IDs, augmentation seeds, learning rate, and checkpoint RNG fields, but CUDA training trajectories diverged from update 3. Enabling PyTorch's deterministic-algorithm guard on the actual STE-FlowNet graph confirmed that `grid_sampler_2d_backward_cuda` has no deterministic implementation in this environment. The model, objective, simulator, labels, and training protocol were not changed to force this comparison to pass.

A replacement checkpoint-boundary oracle passed on the actual fresh-process resume from update 198 to 199. It independently rebuilt the next eight sample IDs and augmentation seeds, checked all 70 Adam counters advanced 198→199, verified scheduler state and model weights changed, and recalculated the update-199 forward loss from the saved source checkpoint. All three loss components matched exactly (absolute error 0). The paired EventGenesis/V2E update-199 logs also have identical sample IDs and augmentation seeds. See [`resume_checkpoint_oracle.json`](evidence/S4/20261017-stage4-corrected-v2/resume_checkpoint_oracle.json), [`resume_full_replay_failure.json`](evidence/S4/20261017-stage4-corrected-v2/resume_full_replay_failure.json), and [`resume_cuda_nondeterminism_diagnostic.json`](evidence/S4/20261017-stage4-corrected-v2/resume_cuda_nondeterminism_diagnostic.json).

The retry ran the complete preflight again using the current code/data fingerprint. All 16 gate checks passed, fingerprint `cf7aa53677b0b2b3b010171d0a8d0c7899562ee2ce48538ee31f4854056f8145`. Five production-launch negative controls (missing receipt, stale config, missing required check, injected failed check, stale data manifest) each exited 1 before creating the run root or reaching the trainer branch. The final full-phase gate verification exited 0. Machine-readable summary: [`resume_retry1_gate_verification.json`](evidence/S4/20261017-stage4-corrected-v2/resume_retry1_gate_verification.json).

## Training observed

Both arms resumed their own validated epoch-boundary checkpoint at update 198 and wrote optimizer update 199. At 19:09:12Z the latest records were: EventGenesis update 315, loss `1100.8965`, gradient norm `393.0761`; V2E update 321, loss `984.0207`, gradient norm `882.8408`. Each update had 70 finite gradient tensors. These values are runtime health checks, not a performance comparison.

Both arms saved resumable update-200 checkpoints (580,035,686 bytes each):

- EventGenesis: `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261017-stage4-flowdt1-seed0-corrected-v5/full_training_retry1/eventgenesis/last.pt`, SHA256 `1cc06ac0e16224bd53e7aece0ee8892c64402344e4a04fc2ddf201e948e8f989`.
- V2E: `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261017-stage4-flowdt1-seed0-corrected-v5/full_training_retry1/v2e/last.pt`, SHA256 `d3241020205eb2ce1849812306b18b8791f90c6e2eb338945afbe6a2c0ff13ad`.

They have since advanced to resumable update-300 checkpoints. As of 19:09:12Z, EventGenesis `last.pt` SHA256 is `40cc0ce3adc17e35a5b3aa126af85fd3b3252a4ddab8b775f28b16e6fbb86bac`; V2E is `39746e5c9bb10742ba05b67de4a865c6e1928b95b0e301f4ecaf3d6b2da339b6` (580,035,686 bytes each). The latest exact metrics and checkpoint hashes are recorded in [`paired_training_progress_190912.json`](evidence/S4/20261017-stage4-corrected-v2/paired_training_progress_190912.json).

Early throughput projects roughly 8–10 additional hours to complete the remaining optimizer updates, before epoch-end synthetic validation overhead. This estimate uses the first 120 updates per arm and will vary with shared server load.

The previous 198-update pilot's synthetic validation remains slightly worse than zero flow for both generators (`0.28709464` EventGenesis, `0.28606818` V2E, versus `0.28420805` zero-flow EPE px/30ms). This diagnostic result is retained and does not support a generator ranking. The full run has not completed.

## Evidence and reproduction

- Gate, retry, checkpoint, and live process evidence: [`evidence/S4/20261017-stage4-corrected-v2/resume_retry1_gate_verification.json`](evidence/S4/20261017-stage4-corrected-v2/resume_retry1_gate_verification.json)
- Complete path/hash inventory, including large server-only artifacts: [`evidence/S4/20261017-stage4-corrected-v2/artifact_index.json`](evidence/S4/20261017-stage4-corrected-v2/artifact_index.json)
- Resume command: `bash benchmark/sim2real/scripts/continue_stage4_corrected_v2_after_pilot_server.sh` (the detached g1 tmux run already launched it).
- Full training logs: `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261017-stage4-flowdt1-seed0-corrected-v5/full_training_retry1/{eventgenesis,v2e}/training_metrics.jsonl`.
- Raw retry commands and stdout/stderr are under `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261017-stage4-flowdt1-seed0-corrected-v5/resume_retry1/`.

The paired training remains active under the detached tmux supervisor. No real-data ranking or sim-to-real advantage is claimed.
