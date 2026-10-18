# E2VID+ paired seed-0 Stage 4 launch evidence

Prepared 2026-10-19. This evidence bundle proves the first 20 optimizer updates per arm, a process-exit checkpoint restore and next-update oracle, and live continuation of the original 12,000-update budgets. It does not report final reconstruction quality or any real-data score.

## Training data and protocol

- Shared physical source: `/home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage4/training_bank_v3_geometry_corrected_20261017_32x2s/shared_trajectory.h5`, SHA-256 `cca882ba912fda3b3243d49bc7e29e28186046e3b6591c424f385b0506047884`.
- Adapter root: `/home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage4/e2vid_corrected_10ms_20261019_v1/`, `adapter_manifest.json` SHA-256 `06ecadb1b4e90bc434ce30148c5d50704385cdab15a24e88ff9366f7c58edcb3`. Its manifest covers both arms across 32 rollout IDs (24 train / 8 validation), 120 disjoint 40-interval clips for training, and 8 validation sequences of 200 10 ms intervals.
- The Stage 4 temporal-consistency flow is a new direct adjacent-endpoint projection from shared physical state. The existing 30 ms optical-flow data and labels were not changed. Paired HDF5 outputs share images, times, labels, K, validity and rollout splits; the manifest includes path, size and SHA-256 for every generated file.
- Recipe remains the corrected-v2 E2VID configuration: E2VIDRecurrent, five signed bins, 40-frame sequence, 112×112 crop, effective batch 2, Adam `1e-4`/AMSGrad, 200 epochs × 60 updates = 12,000 updates. The checkpoint rule is synthetic-validation LPIPS-VGG only. Real scoring is disabled.

## Launch and preflight results

- Environment: g1, Python 3.11.15, PyTorch 2.5.1+cu121, CUDA build 12.1, Tesla V100S-PCIE-32GB. The model source bundle is pinned to `event_cnn_minimal` commit `78aa471a66593f55e349303563997297fbc816e8`; the active trainer script SHA-256 is `73cc8a1bfc2d8a4cdcf86e32ca4bdeaa0091239e93f499108aaac02cf7da2d58`.
- Shared crop/flip audit passed on four registered seeds for events, targets, physical-flow spatial transforms and validity. Each arm's recurrent-state audit passed continuation inside a clip and exact state reset at clip start.
- The full 40-frame clip executed temporal consistency. A direct `autograd.grad` probe on its first update recorded nonzero TC-to-model gradient norms `0.023957182051961377` (EventGenesis) and `0.023921836096990317` (V2E). The update log records 76 valid TC calls and separate perceptual/TC loss components. Losses and gradients were finite.
- Both arms completed the fixed 20-update segment with process exit code 0. Update-20 checkpoint SHA-256 values are `cb4f4000b435195cff1530bfc0600162ae5cbb689eb4ec39f5e57b156f6de9a0` (EventGenesis) and `776ee6b18d984fe566f36360e9d83c7460b6050a076608fe12c4ebdd9aeb484b` (V2E), 171,427,458 bytes each. Their initial state SHA-256 is `48426d94f4159592155eda873d4e71e2f06439c10eade990de5979231ed07a0f`.
- After a fresh process restore, each arm reproduced the registered next sample IDs `[55,4]`, the content digests and forward loss totals; optimizer step counters advanced from 20 to 21. The resumed trainer also compared its first next-batch sample hashes/losses against the oracle and passed. This is a boundary/next-update check, not proof of exact full-trajectory replay.

## Current live continuation

At the latest direct server snapshot, **2026-10-18 17:37:55 UTC / 2026-10-19 02:37:55 JST**, both `live_status.json` and append-only optimizer logs were at epoch 10, update 600 / 12,000. The update-600 EventGenesis checkpoint is 171,427,970 bytes, SHA-256 `8b33d0d087814e32a745c7821791079cf22770ccedea42bc20d9bde25919bcb3`; V2E is the same size, SHA-256 `5831ad77442c3b4c1a3c8bea2b510a1a44853d4e0ee486285f69c00ffe198c16`. Both tmux launchers and Python trainers were alive (pane/Python PIDs 246943/246981 and 246948/246982) on physical GPUs 2 and 4; each GPU used 3,252 MiB. GPU 4 reported 91% utilization and GPU 2 0% during the snapshot. The next periodic checkpoint is update 900. The exact growing log sizes and hashes are recorded in `evidence_index.json` and `live_observation_20261018T173755Z.json`.

Current external run root: `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261019-stage4-e2vid-seed0-corrected-v2/{eventgenesis,v2e}/`. Full logs and weights stay on g1; checkpoint and append-only log hashes at a named time are indexed in `artifact_index.json`. The compact pilot/resume logs and JSON oracle records are copied here for research handoff.

## Preserved initial loader failure

Run root `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261019-stage4-e2vid-seed0-corrected-v1/` is retained under `attempt_v1/`. Data generation and its hashes completed, but the first training launch stopped before any update because the upstream `SequenceDataset` returned zero validation sequences when the requested 200-sample sequence exactly equaled dataset length. The Stage 4 adapter now permits one validated 0–199 sequence only for that registered full-rollout boundary. The corrected v2 preflight checks the sample IDs and rollout identity. No frozen simulator core, model architecture or training budget was changed.

## Reproduction

The 20-update start and current `--resume last.pt` commands are recorded in each arm's `pilot_command.txt` and `resume_stdout.log`; their actual tmux PIDs and names are retained in `tmux_pids.txt` and the server's session record. To inspect progress without touching the live jobs:

```bash
cd /home/zhaoyuhan/Eventbased_WAM/repositories/gec-sim2real
source ~/miniconda3/etc/profile.d/conda.sh
conda activate Eventbased_WAM
python3 - <<'PY'
import json, pathlib
root=pathlib.Path('/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261019-stage4-e2vid-seed0-corrected-v2')
for arm in ('eventgenesis','v2e'):
    status=json.loads((root/arm/'live_status.json').read_text())
    print(arm, status['optimizer_update'], '/', status['optimizer_budget'], status['updated_utc'])
PY
```

Both runs are still in progress. Update 20 means the required actual pilot and resume proof were delivered; it does not mean the full reconstruction training is complete.
