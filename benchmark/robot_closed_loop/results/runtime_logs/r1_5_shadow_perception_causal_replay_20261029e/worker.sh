#!/usr/bin/env bash
set -u
ROOT=/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot-r1_5-20261029-f6a2b1e
LOGDIR="$ROOT/benchmark/robot_closed_loop/results/runtime_logs/r1_5_shadow_perception_causal_replay_20261029e"
OUT="$ROOT/benchmark/robot_closed_loop/results/r1_5_shadow_perception_causal_replay_20261029e"
ROBOT_PYTHON=/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot/.venv-robot/bin/python
cd "$ROOT" || exit 91
printf 'started_utc=%s\nrun_id=r1_5_shadow_perception_causal_replay_20261029e\nsource_commit=7f685817119d84b9df80120c3ebb6e85cc554c0a\ncheckout_commit=855b4b0bc7de96170b4bfee8d7002b370782f346\n' "$(date -u +%FT%TZ)" | tee "$LOGDIR/worker_started_at.txt"
set +e
DIAGNOSTIC_SCRIPT_SOURCE_COMMIT=7f685817119d84b9df80120c3ebb6e85cc554c0a PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES= EGL_DEVICE_ID=0 PYOPENGL_PLATFORM=egl \
  "$ROBOT_PYTHON" benchmark/robot_closed_loop/diagnostics/diagnose_r1_5_shadow_perception_causal_replay.py \
  --cache-root benchmark/robot_closed_loop/results/r1_failure_chain_diagnosis_20261019a/native_hdr_cutoff059 \
  --scene-config benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json \
  --experiment-config benchmark/robot_closed_loop/configs/r1_5_shadow_perception_causal_replay_20261029e.json \
  --checkpoint benchmark/robot_closed_loop/results/r1_shared_ideal_small9_20261018b/paired_r1/ours_fit/spatial_event_state.pt \
  --output benchmark/robot_closed_loop/results/r1_5_shadow_perception_causal_replay_20261029e \
  2>&1 | tee "$LOGDIR/run.log"
rc=${PIPESTATUS[0]}
echo "$rc" > "$LOGDIR/exit_code.txt"
printf 'finished_utc=%s\nexit_code=%s\n' "$(date -u +%FT%TZ)" "$rc" | tee "$LOGDIR/worker_finished_at.txt"
exit "$rc"
