# Stage 4 corrected v2 status

Current status is maintained in [`TRAINING_LIVE_STATUS.md`](TRAINING_LIVE_STATUS.md). At `2026-10-17T19:04:16Z`, the launch gate had passed 16/16 checks, all five negative controls blocked invalid launches, the actual checkpoint-boundary resume oracle passed, and paired EventGenesis/V2E 45-epoch synthetic-only resumes were active on g1 GPUs 2 and 4.

The initial strict full-trajectory replay remains preserved as a failed check because CUDA `grid_sampler_2d_backward_cuda` is nondeterministic in the installed PyTorch build. It is not relabeled as passing. The gate receipt key `resume_equivalence_epoch_boundary` covers checkpoint-boundary restoration, next-batch reconstruction, one resumed optimizer update, and a forward-only metric oracle. It does not prove strict full-trajectory equivalence, and the oracle does not independently reimplement Adam's parameter update.

Machine evidence and raw path/hash index: [`evidence/S4/20261017-stage4-corrected-v2/`](evidence/S4/20261017-stage4-corrected-v2/). Large banks, event shards, full logs, and weights remain on g1 at paths recorded in `artifact_index.json`.
