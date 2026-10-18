# Stage 4 three-line training and evaluation status

**Latest direct g1 snapshot:** 2026-10-18 17:37:55 UTC / 2026-10-19 02:37:55 JST. Earlier Stage 3 and Stage 4 snapshots remain historical. Training jobs stay detached on g1; this page records the actual observation below, not an inferred completion estimate.

| Line | Latest verified state | Status | Next checkpoint / delivery |
|---|---|---|---|
| E2VID+ reconstruction, EventGenesis | Update **600 / 12,000**, epoch 10 / batch 60, latest optimizer record `2026-10-18T17:37:33.268330Z`; TC valid calls 76 and 30 finite gradient tensors. | Running on g1 GPU 2; detached tmux pane/Python PID 246943/246981; resumed from saved update 20. Update-600 checkpoint SHA-256 `8b33d0d087814e32a745c7821791079cf22770ccedea42bc20d9bde25919bcb3`, 171,427,970 bytes. | Next periodic checkpoint at update 900. Continue fixed budget and record synthetic validation. |
| E2VID+ reconstruction, V2E | Update **600 / 12,000**, epoch 10 / batch 60, latest optimizer record `2026-10-18T17:37:22.248206Z`; TC valid calls 76 and 30 finite gradient tensors. | Running on g1 GPU 4; detached tmux pane/Python PID 246948/246982; resumed from saved update 20. Update-600 checkpoint SHA-256 `5831ad77442c3b4c1a3c8bea2b510a1a44853d4e0ee486285f69c00ffe198c16`, 171,427,970 bytes. | Next periodic checkpoint at update 900. Continue fixed budget and record synthetic validation. |
| STE-FlowNet dt1, EventGenesis | 45/45 epochs, **8,910 / 8,910** updates, final status `complete`, per-arm resumed trainer exit code `0`; synthetic-best epoch 33 / EPE `0.2832538118` px/30 ms. | Complete; real MVSEC score closed. | Final-best fixed synthetic diagnostic and 45-epoch curve archive complete. Await protocol freeze before real scoring. |
| STE-FlowNet dt1, V2E | 45/45 epochs, **8,910 / 8,910** updates, final status `complete`, per-arm resumed trainer exit code `0`; synthetic-best epoch 20 / EPE `0.2828732346` px/30 ms. | Complete; real MVSEC score closed. | Same as EventGenesis. |
| SingleShotPose / E-POSE | No E-POSE training process. Dataset publisher recheck and synthetic eval-mode confidence/BN mechanism check completed. | Waiting on original event/pose sample and exact sample-matched CAD; no formal task train or real inference. | Resolve public access, then sample contract and held-out synthetic normal-eval gate. |

## Reconstruction seed-0 live run

- Run root: `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261019-stage4-e2vid-seed0-corrected-v2/{eventgenesis,v2e}/`; detached jobs use CUDA-visible physical GPUs 2 and 4. At the latest query each reserved about 3.2 GiB; GPU3 was idle after the flow diagnostic.
- Target: 200 epochs × 60 updates = **12,000 updates per generator**, with the frozen corrected-v2 reconstruction recipe and synthetic-validation LPIPS-VGG checkpoint selection.
- Data: `/home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage4/e2vid_corrected_10ms_20261019_v1/`; adapter manifest SHA-256 `06ecadb1b4e90bc434ce30148c5d50704385cdab15a24e88ff9366f7c58edcb3`. Shared source bank SHA-256 `cca882ba912fda3b3243d49bc7e29e28186046e3b6591c424f385b0506047884`.
- g1 environment: Python 3.11.15, PyTorch 2.5.1+cu121, CUDA build 12.1, Tesla V100S-PCIE-32GB. E2VID source bundle is pinned to event_cnn_minimal commit `78aa471a66593f55e349303563997297fbc816e8`; trainer SHA-256 `73cc8a1bfc2d8a4cdcf86e32ca4bdeaa0091239e93f499108aaac02cf7da2d58`.
- Seed-0 paired initial state SHA-256: `48426d94f4159592155eda873d4e71e2f06439c10eade990de5979231ed07a0f`. Update-20 checkpoints: EventGenesis `cb4f4000b435195cff1530bfc0600162ae5cbb689eb4ec39f5e57b156f6de9a0`; V2E `776ee6b18d984fe566f36360e9d83c7460b6050a076608fe12c4ebdd9aeb484b`; each 171,427,458 bytes.
- Actual resume oracle passed on each arm: the saved cursor reproduced the next two clip indices `[55,4]`, identical content hashes and the same two forward loss totals as the independent probe; optimizer counters advanced from 20 to 21. This checks boundary restore and the next update, not strict bitwise equivalence over a whole training trajectory.
- TC ran during the 40-frame clips (76 valid calls per optimizer update in the recorded case), its loss was separately logged, and a direct autograd probe measured nonzero TC-to-model gradient norms `0.0239571821` / `0.0239218361` on EventGenesis / V2E. The shared crop/flip preflight passed four fixed seeds for event tensors, images, flow-vector spatial channels and validity. Recurrent preflight passed for both models: state continued within a clip and the reset first-step output matched exactly.
- First v1 loader attempt and its failure remain in `evidence/S4/20261019-e2vid-parallel-launch/attempt_v1/`; it exited before training because the upstream dataset returned zero validation sequences at an exact-length boundary. The v2 adapter admits exactly the registered 200 samples and asserts IDs `000..199`. The original 30 ms optical-flow bank is untouched.
- Compact copied launch/resume evidence and its hash index: [`evidence/S4/20261019-e2vid-parallel-launch/`](evidence/S4/20261019-e2vid-parallel-launch/). Full and growing stdout/JSONL logs and weights stay at the g1 paths above. No real test is used.

At 17:37:55 UTC, both launcher panes and Python trainers were live. GPU 2 and GPU 4 each used 3,252 MiB of 32 GiB; GPU 4 reported 91% utilization, while GPU 2 was at 0% during the snapshot. GPUs 0/1/3/5 were effectively idle. Both latest metric rows and status JSONs agreed on update 600. The checkpoint/log sizes and SHA-256 values are recorded in the timestamped g1 artifact snapshot; the update-600 checkpoints correspond to each saved model state.

### Recheck live progress

Run from g1 in the `Eventbased_WAM` conda environment:

```bash
cd /home/zhaoyuhan/Eventbased_WAM/repositories/gec-sim2real
source ~/miniconda3/etc/profile.d/conda.sh
conda activate Eventbased_WAM
python3 - <<'PY'
import json, pathlib
root=pathlib.Path('/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261019-stage4-e2vid-seed0-corrected-v2')
for arm in ('eventgenesis','v2e'):
    status=json.loads((root/arm/'live_status.json').read_text())
    print(arm, status['optimizer_update'], '/', status['optimizer_budget'], status['updated_utc'], status['status'])
PY
```

Full-run commands are the recorded `resume_stdout.log`'s launcher invocation, equivalent to:

```bash
CUDA_VISIBLE_DEVICES=2 python -u benchmark/sim2real/scripts/train_e2vid_stage4.py \
  --model-root /home/zhaoyuhan/Eventbased_WAM/external/sources/downstream/event_cnn_minimal \
  --data-root /home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage4/e2vid_corrected_10ms_20261019_v1 \
  --config benchmark/sim2real/configs/stage4_corrected_v2_flowdt1_seed0.json \
  --output-root /home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261019-stage4-e2vid-seed0-corrected-v2/eventgenesis \
  --generator eventgenesis --device cuda:0 --seed 0 \
  --resume /home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261019-stage4-e2vid-seed0-corrected-v2/eventgenesis/last.pt

CUDA_VISIBLE_DEVICES=4 python -u benchmark/sim2real/scripts/train_e2vid_stage4.py \
  --model-root /home/zhaoyuhan/Eventbased_WAM/external/sources/downstream/event_cnn_minimal \
  --data-root /home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage4/e2vid_corrected_10ms_20261019_v1 \
  --config benchmark/sim2real/configs/stage4_corrected_v2_flowdt1_seed0.json \
  --output-root /home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261019-stage4-e2vid-seed0-corrected-v2/v2e \
  --generator v2e --device cuda:0 --seed 0 \
  --resume /home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261019-stage4-e2vid-seed0-corrected-v2/v2e/last.pt
```

Use these commands only to reproduce in a separate authorized run directory; do not duplicate the active processes.

## Optical flow final delivery

- Full 45-epoch curve (90 CSV rows, one row per arm/epoch), exit receipts, checkpoint hashes, external log hashes and reproduction command: [`evidence/S4/20261019-flow-final-archive/`](evidence/S4/20261019-flow-final-archive/).
- Final-best diagnostic JSON and retained initial failed-load attempt: [`evidence/S4/20261019-final-best-motion-diagnostic/`](evidence/S4/20261019-final-best-motion-diagnostic/). The diagnostic compares both synthetic-best weights on the same 64 validation windows and shared event-active-union mask. Overall full-valid improvement over zero flow is 0.20% / 0.43%; common active-union improvement is 14.0% / 10.5% at 0.487% support coverage. This indicates limited event-supported motion learning, not a real-data result.
- MVSEC dt1 protocol and evaluator remain candidates; `real_scores_authorized=false`. Controlled metric self-test passed; real scoring is blocked pending research freeze and exact data manifest hashes/index registration.

## Pose data and inference readiness

- E-POSE official record recheck: HTTP 202 landing and HTTP 403 API from the current WSL route. Author `wood_block` mesh is verified but not linked to an original event/pose sample; sample-level K, time, transform convention, unit, exact CAD scale and symmetry remain unaudited.
- SingleShotPose's short old budget supplied only 8 confidence-supervised updates after its 15-epoch warmup. A synthetic-only fixture subsequently showed stable ordinary `model.eval()` confidence-argmax inference after added confidence updates and train-only BN buffer recalibration, with normal save/restore. Held-out rollout and E-POSE results are still missing.
- Details and public-access evidence: [`evidence/S4/20261019-pose-eval-access/POSE_EVAL_AND_ACCESS_AUDIT.md`](evidence/S4/20261019-pose-eval-access/POSE_EVAL_AND_ACCESS_AUDIT.md).

Full three-line status and evidence interpretation: [`sim2real_status.md`](sim2real_status.md) and [`STAGE4_PARALLEL_EXECUTION_20261019.md`](STAGE4_PARALLEL_EXECUTION_20261019.md).
