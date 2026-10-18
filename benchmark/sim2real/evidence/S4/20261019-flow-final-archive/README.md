# Corrected-v2 STE-FlowNet dt1 seed-0 final archive

Completed 2026-10-19. This archive joins the retained first epoch with the 44-epoch resumed continuation; it does not retrain or change checkpoint selection.

## Training and completion evidence

- g1 run root: `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261017-stage4-flowdt1-seed0-corrected-v5/`.
- Epoch 1 is the completed paired pilot under `pilot/{eventgenesis,v2e}`: updates 1–198. Epochs 2–45 are under `full_training_retry1/{eventgenesis,v2e}`: updates 199–8,910. The compact CSV has 45 rows per arm and exactly 90 rows total.
- The archiver checks every optimizer update is present once and in order, each epoch has 198 updates, every epoch has one synthetic-validation record, final status reaches 8,910/8,910, both run arms share their initial model hash, both resumed per-arm training process receipts are exit 0, and the successful `resume_retry1` supervisor is exit 0.
- EventGenesis synthetic-best: epoch 33, EPE `0.2832538118050536` px/30 ms, 580,038,858 bytes, SHA-256 `a4e6c7e1d10a4d76562a046c364a5e17de6452d5b648fc8ccbaa3bd43c824a8e`. Final: EPE `0.2837044896456437`, 580,035,686 bytes, SHA-256 `d9efa865bdfd33e6c5f7ff0f75a2b1654f57982839efb0457495b99b8403d5ac`.
- V2E synthetic-best: epoch 20, EPE `0.2828732346143402` px/30 ms, 580,038,858 bytes, SHA-256 `11041863685b417306f534b8e12a1177b1337fc2818d255ec7b76fa6e20cd474`. Final: EPE `0.28428155346517947`, 580,035,686 bytes, SHA-256 `8fd0d7379f46827a6dc39eaa1377b8dea953e85210abae4055dea41248553d79`.
- Full metrics JSONL and all large weights remain at g1; `artifact_index.json` records exact remote paths, sizes and hashes. Final validation EPE is `0.2837044896` / `0.2842815535` (EventGenesis / V2E); zero-flow reference is `0.2842080519`. These are synthetic-only results, not a generator ranking.

## Historical gate exits and successful retry

The initial campaign wrapper exit code is `2`. Its first post-data continuation exited `1` because the strict resume-equivalence gate failed. Those are preserved gate/launcher statuses, not the final trainer-process statuses. A separate retry performed the actual checkpoint-boundary resume oracle, checked the remaining launch gates, and launched the full training: `resume_retry1/exit_code.txt` and both `full_resume_{eventgenesis,v2e}.exit_code.txt` contain `0`; both final live statuses are `complete`. The retry supervisor's completion record says both paired 45-epoch runs completed successfully. The review distinction is explicit in the `exit_code_note` field of `artifact_index.json`.

## Curve and diagnostic

- [`flow_seed0_45_epoch_curve.csv`](flow_seed0_45_epoch_curve.csv): per epoch and arm, optimizer update range, mean train/photo/smoothness loss and gradient norm, full synthetic validation EPE, zero-flow EPE and best-so-far values.
- [`../20261019-final-best-motion-diagnostic/motion_diagnostic.json`](../20261019-final-best-motion-diagnostic/motion_diagnostic.json): 64 registered validation windows, selected synthetic-best checkpoints at each arm's own best epoch, four scene strata, full-valid and common-event-active masks, and motion-threshold regions. EPE unit is endpoint displacement px per 30 ms. Real MVSEC scores were not read.

## Reproduction

From the g1 repository root, using Python 3 standard library:

```bash
python3 benchmark/sim2real/scripts/archive_stage4_flow_seed0.py \
  --campaign-root /home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261017-stage4-flowdt1-seed0-corrected-v5 \
  --output-dir benchmark/sim2real/evidence/S4/20261019-flow-final-archive
```

The archive command exited 0 on g1. The script fails closed on missing updates, missing epoch validations, bad per-arm trainer exit receipts, incomplete status, or initial-state mismatch. It refuses to overwrite a prior successful CSV/index. Earlier failed archive attempts are retained as `attempt1_*` / `attempt2_*`; they exposed an incorrect assumption that the historical top-level gate wrapper must exit 0, and then an overwrite guard after the preliminary CSV had already been written. No training file was changed.

## Evaluation isolation

Training and selection remain synthetic-only. The final-best motion diagnostic uses fixed held-out synthetic validation samples. MVSEC dt1 metrics and data-manifest rules are still a review candidate, with `real_scores_authorized=false`; no real scoring was run.
