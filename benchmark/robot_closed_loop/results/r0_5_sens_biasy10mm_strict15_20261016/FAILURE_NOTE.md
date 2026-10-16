# Sensitivity attempt ended before metrics were written

This directory preserves the first +10 mm lateral-position sensitivity attempt. It is diagnostic evidence only and contributes no episode success/failure counts.

- Run commit: 8573279936d40102efeabf00286e34b6c99e7681
- Config SHA256: dcc681fa57ed85b38871455b9ddbb736af57a5c37fab5b6bffb9356b2d86c0cd
- Seed block: 2026103500; first episode simulation completed, but result assembly raised before CSV/summary serialization.
- Exit code: 1
- Error: UnboundLocalError for legacy_true_bottom_min_at_stop when the legacy scorer never reached its historical stop condition.
- Fix: initialize the absent legacy diagnostic to null in benchmark code; no task, control, or score criterion changed. Fix commit fcf567f7485e590b075b378c4ff726079e0a287a.
- A fresh 15-episode retry uses seed block 2026103515 and a separate output directory.

See results/runtime_logs/r0_5_sens_biasy10mm_strict15_20261016.log for the complete traceback.
