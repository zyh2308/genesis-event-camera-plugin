#!/usr/bin/env bash
set -uo pipefail
ROOT=/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot-r1_5-20261029-f6a2b1e
RUN_ID=r1_5_shadow_perception_causal_replay_20261029c
OUT="$ROOT/benchmark/robot_closed_loop/results/$RUN_ID"
LOGDIR="$ROOT/benchmark/robot_closed_loop/results/runtime_logs/$RUN_ID"
ALLOC=/home/zhaoyuhan/Eventbased_WAM/runs/robot_scenes/genesis_event_camera_plugin/$RUN_ID/resource_allocation.json
ROBOT_PYTHON=/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot/.venv-robot/bin/python
SCRIPT="$ROOT/benchmark/robot_closed_loop/diagnostics/diagnose_r1_5_shadow_perception_causal_replay.py"
SCENE="$ROOT/benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json"
EXPERIMENT="$ROOT/benchmark/robot_closed_loop/configs/r1_5_shadow_perception_causal_replay_20261029c.json"
CACHE="$ROOT/benchmark/robot_closed_loop/results/r1_failure_chain_diagnosis_20261019a/native_hdr_cutoff059"
CHECKPOINT="$ROOT/benchmark/robot_closed_loop/results/r1_shared_ideal_small9_20261018b/paired_r1/ours_fit/spatial_event_state.pt"
RUN_LOG="$LOGDIR/run.log"
PYTHONPATH="$ROOT"
export PYTHONPATH CUDA_VISIBLE_DEVICES= EGL_DEVICE_ID=0 PYOPENGL_PLATFORM=egl
printf '%s\n' "$(date -Is)" > "$LOGDIR/worker_started_at.txt"
printf '%s\n' "$$" > "$LOGDIR/worker_pid.txt"
python3 - "$ALLOC" "$$" <<'PY'
import json, sys, time
from pathlib import Path
p = Path(sys.argv[1])
data = json.loads(p.read_text())
data.update({"status": "running", "worker_pid": int(sys.argv[2]), "worker_started_at_epoch": time.time()})
p.write_text(json.dumps(data, indent=2) + "\n")
PY
"$ROBOT_PYTHON" "$SCRIPT" \
  --cache-root "$CACHE" \
  --scene-config "$SCENE" \
  --experiment-config "$EXPERIMENT" \
  --checkpoint "$CHECKPOINT" \
  --output "$OUT" > "$RUN_LOG" 2>&1 &
PYTHON_PID=$!
printf '%s\n' "$PYTHON_PID" > "$LOGDIR/python_pid.txt"
python3 - "$ALLOC" "$PYTHON_PID" <<'PY'
import json, sys, time
from pathlib import Path
p = Path(sys.argv[1])
data = json.loads(p.read_text())
data.update({"python_pid": int(sys.argv[2]), "python_started_at_epoch": time.time()})
p.write_text(json.dumps(data, indent=2) + "\n")
PY
wait "$PYTHON_PID"
EXIT_CODE=$?
printf '%s\n' "$EXIT_CODE" > "$LOGDIR/exit_code.txt"
python3 - "$ALLOC" "$EXIT_CODE" <<'PY'
import json, sys, time
from pathlib import Path
p = Path(sys.argv[1])
data = json.loads(p.read_text())
data.update({"status": "complete" if int(sys.argv[2]) == 0 else "failed", "exit_code": int(sys.argv[2]), "finished_at_epoch": time.time()})
p.write_text(json.dumps(data, indent=2) + "\n")
PY
exit "$EXIT_CODE"
