# R1.5 surface-raster controls: unknown-region extension

This g1 run reran the five general candidate controls and added a direct check that a pixel unsupported by both endpoint surfaces stays unknown and requests exactly one render.

- Source code commit: `8c84e3b` (`exp/robot-closed-loop-v1`).
- The isolated g1 worktree remained at replay commit `a92f25019982affb34427631e1c6921bf9ad4aa6`; the exact control script was copied from source commit `8c84e3b` and verified before execution.
- Control script SHA256: `4ad85279ba11046edaf862174820bd59e0de911bdb54d93350655881ef513d16`.
- Candidate implementation SHA256: `74b18a926f33a70740bbdc44a33d543b9c3ee9d7e99df4b2649e0bd3f04b8dd5`.
- Candidate config SHA256: `eb7278f69c6a7dc586e3254356b22ee121cb90219244a3ad45862586811adc05`.
- g1 Python: `/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot/.venv-robot/bin/python`; CPU-only, Numba backend, `CUDA_VISIBLE_DEVICES` empty.
- All six controls passed; total wall time `1.3489 s`. The unknown pixel remained NaN and exactly one pixel was requested for direct rendering.
- Output `controls.json` SHA256 `def75a55391683b89411ed976709faa0427fe68711666a6e63621cae2c4c9f23`; `SHA256SUMS` verifies `controls.json`, `stdout.json`, and `exit_code.txt`.

The equivalent command is:

```bash
cd /home/zhaoyuhan/Eventbased_WAM/repositories/genesis-event-camera-plugin-r1_5_surface_20261021a
PY=/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot/.venv-robot/bin/python
PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TI_NUM_THREADS=4 \
"$PY" benchmark/robot_closed_loop/diagnostics/run_r1_5_surface_raster_controls.py \
  --output benchmark/robot_closed_loop/results/r1_5_surface_raster_controls_20261021b/controls.json
```
