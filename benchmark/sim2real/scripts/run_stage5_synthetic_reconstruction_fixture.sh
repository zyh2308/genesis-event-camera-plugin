#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 10 ]]; then
  echo "usage: $0 REPO_ROOT MODEL_ROOT EVREAL_ROOT BANK_H5 ROLLOUT_ID GENERATOR V2E_ROOT CHECKPOINT MODEL_CONFIG RUN_ROOT" >&2
  exit 2
fi

REPO_ROOT=$1
MODEL_ROOT=$2
EVREAL_ROOT=$3
BANK_H5=$4
ROLLOUT_ID=$5
GENERATOR=$6
V2E_ROOT=$7
CHECKPOINT=$8
MODEL_CONFIG=$9
RUN_ROOT=${10}
PYTHON_BIN=${PYTHON_BIN:-/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM/bin/python}

case "$GENERATOR" in
  eventgenesis|v2e) ;;
  *) echo "unsupported generator: $GENERATOR" >&2; exit 2 ;;
esac

if [[ ! -x "$PYTHON_BIN" || ! -d "$REPO_ROOT" || ! -d "$MODEL_ROOT" || ! -d "$EVREAL_ROOT" \
      || ! -f "$BANK_H5" || ! -d "$V2E_ROOT" || ! -f "$CHECKPOINT" ]]; then
  echo "one or more registered inputs are missing" >&2
  exit 2
fi
if [[ ! -f "$MODEL_CONFIG" ]]; then
  echo "registered Stage-5 model config is missing: $MODEL_CONFIG" >&2
  exit 2
fi
if [[ -e "$RUN_ROOT" ]]; then
  echo "refusing to overwrite run root: $RUN_ROOT" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT/fixture"
START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
printf '%s\n' "$START_UTC" > "$RUN_ROOT/started_utc.txt"
printf '%q ' "$PYTHON_BIN" "$REPO_ROOT/benchmark/sim2real/scripts/export_stage5_synthetic_e2vid_memmap.py" \
  --bank-h5 "$BANK_H5" --rollout-id "$ROLLOUT_ID" --generator "$GENERATOR" \
  --v2e-root "$V2E_ROOT" --output-root "$RUN_ROOT/fixture/$GENERATOR" \
  > "$RUN_ROOT/commands.txt"
printf '\n' >> "$RUN_ROOT/commands.txt"
printf '%q ' "$PYTHON_BIN" "$REPO_ROOT/benchmark/sim2real/scripts/infer_hqf_e2vid_candidate.py" \
  --evreal-root "$EVREAL_ROOT" --model-root "$MODEL_ROOT" --checkpoint "$CHECKPOINT" \
  --model-config "$MODEL_CONFIG" \
  --synthetic-fixture-sequence "$RUN_ROOT/fixture/$GENERATOR/${ROLLOUT_ID}_${GENERATOR}" \
  --output-root "$RUN_ROOT/inference/$GENERATOR" --device cpu --save-every 1 --save-float-arrays \
  >> "$RUN_ROOT/commands.txt"
printf '\n' >> "$RUN_ROOT/commands.txt"
printf '%q ' "$PYTHON_BIN" "$REPO_ROOT/benchmark/sim2real/scripts/evaluate_synthetic_reconstruction_fixture.py" \
  --fixture-root "$RUN_ROOT/fixture/$GENERATOR" \
  --sequence-root "$RUN_ROOT/fixture/$GENERATOR/${ROLLOUT_ID}_${GENERATOR}" \
  --inference-root "$RUN_ROOT/inference/$GENERATOR" --model-root "$MODEL_ROOT" \
  --output "$RUN_ROOT/metrics/$GENERATOR.json" --device cpu \
  >> "$RUN_ROOT/commands.txt"
printf '\n' >> "$RUN_ROOT/commands.txt"
printf '%q ' "$PYTHON_BIN" "$REPO_ROOT/benchmark/sim2real/scripts/verify_stage5_synthetic_reconstruction_fixture.py" \
  --run-root "$RUN_ROOT" --generator "$GENERATOR" \
  --metrics-result "$RUN_ROOT/metrics/$GENERATOR.json" \
  --output "$RUN_ROOT/verification/$GENERATOR.json" \
  >> "$RUN_ROOT/commands.txt"
printf '\n' >> "$RUN_ROOT/commands.txt"

trap 'rc=$?; date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_ROOT/finished_utc.txt"; printf "%s\n" "$rc" > "$RUN_ROOT/exit_code.txt"; exit "$rc"' EXIT
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
cd "$REPO_ROOT"
{
  date -u
  "$PYTHON_BIN" -c 'import torch,torchvision,numpy,h5py,cv2,skimage; print("python",__import__("sys").version); print("torch",torch.__version__); print("torchvision",torchvision.__version__); print("numpy",numpy.__version__); print("h5py",h5py.__version__); print("opencv",cv2.__version__); print("skimage",skimage.__version__)'
  sha256sum "$REPO_ROOT/benchmark/sim2real/scripts/export_stage5_synthetic_e2vid_memmap.py" \
    "$REPO_ROOT/benchmark/sim2real/scripts/infer_hqf_e2vid_candidate.py" \
    "$REPO_ROOT/benchmark/sim2real/scripts/evaluate_synthetic_reconstruction_fixture.py" \
    "$REPO_ROOT/benchmark/sim2real/scripts/verify_stage5_synthetic_reconstruction_fixture.py" \
    "$REPO_ROOT/benchmark/sim2real/scripts/build_e2vid_stage4_10ms_data.py" \
    "$REPO_ROOT/benchmark/sim2real/scripts/event_window_audit.py" \
    "$REPO_ROOT/benchmark/sim2real/scripts/stage4_v2e_artifacts.py" \
    "$REPO_ROOT/benchmark/sim2real/scripts/verify_pinned_source.py" \
    "$CHECKPOINT" "$BANK_H5" "$MODEL_CONFIG"
} > "$RUN_ROOT/environment_and_hashes.txt" 2>&1

"$PYTHON_BIN" -u "$REPO_ROOT/benchmark/sim2real/scripts/export_stage5_synthetic_e2vid_memmap.py" \
  --bank-h5 "$BANK_H5" --rollout-id "$ROLLOUT_ID" --generator "$GENERATOR" \
  --v2e-root "$V2E_ROOT" --output-root "$RUN_ROOT/fixture/$GENERATOR" \
  > "$RUN_ROOT/export.stdout.log" 2> "$RUN_ROOT/export.stderr.log"
"$PYTHON_BIN" -u "$REPO_ROOT/benchmark/sim2real/scripts/infer_hqf_e2vid_candidate.py" \
  --evreal-root "$EVREAL_ROOT" --model-root "$MODEL_ROOT" --checkpoint "$CHECKPOINT" \
  --model-config "$MODEL_CONFIG" \
  --synthetic-fixture-sequence "$RUN_ROOT/fixture/$GENERATOR/${ROLLOUT_ID}_${GENERATOR}" \
  --output-root "$RUN_ROOT/inference/$GENERATOR" --device cpu --save-every 1 --save-float-arrays \
  > "$RUN_ROOT/inference.stdout.log" 2> "$RUN_ROOT/inference.stderr.log"
mkdir -p "$RUN_ROOT/metrics"
"$PYTHON_BIN" -u "$REPO_ROOT/benchmark/sim2real/scripts/evaluate_synthetic_reconstruction_fixture.py" \
  --fixture-root "$RUN_ROOT/fixture/$GENERATOR" \
  --sequence-root "$RUN_ROOT/fixture/$GENERATOR/${ROLLOUT_ID}_${GENERATOR}" \
  --inference-root "$RUN_ROOT/inference/$GENERATOR" --model-root "$MODEL_ROOT" \
  --output "$RUN_ROOT/metrics/$GENERATOR.json" --device cpu \
  > "$RUN_ROOT/metrics.stdout.log" 2> "$RUN_ROOT/metrics.stderr.log"
mkdir -p "$RUN_ROOT/verification"
"$PYTHON_BIN" -u "$REPO_ROOT/benchmark/sim2real/scripts/verify_stage5_synthetic_reconstruction_fixture.py" \
  --run-root "$RUN_ROOT" --generator "$GENERATOR" \
  --metrics-result "$RUN_ROOT/metrics/$GENERATOR.json" \
  --output "$RUN_ROOT/verification/$GENERATOR.json" \
  > "$RUN_ROOT/verification.stdout.log" 2> "$RUN_ROOT/verification.stderr.log"
