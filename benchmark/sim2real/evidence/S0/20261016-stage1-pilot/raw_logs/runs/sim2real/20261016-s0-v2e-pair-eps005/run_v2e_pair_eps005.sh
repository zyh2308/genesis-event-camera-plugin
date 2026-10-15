#!/usr/bin/env bash
set -uo pipefail
RUN_DIR=/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261016-s0-v2e-pair-eps005
ROOT=/home/zhaoyuhan/Eventbased_WAM
V2E_SOURCE=$ROOT/external/sources/v2e-cac6384445/v2e-pinned
CHECKPOINT=$ROOT/external/sources/v2e-cac6384445/input/SuperSloMo39.ckpt
DATA_DIR=$ROOT/data/sim2real/stage0/shared-trajectory-20261016-v1
OUTPUT_DIR=$DATA_DIR/v2e_output_eps005
printf '%s\n' "$$" > "$RUN_DIR/pid.txt"
source /home/zhaoyuhan/miniconda3/bin/activate Eventbased_WAM || exit 90
export XDG_CACHE_HOME="$RUN_DIR/cache/xdg"
export GS_CACHE_FILE_PATH="$RUN_DIR/cache/genesis"
export TI_OFFLINE_CACHE=1
export TI_OFFLINE_CACHE_FILE_PATH="$RUN_DIR/cache/taichi"
export CUDA_VISIBLE_DEVICES=1
export PYOPENGL_PLATFORM=egl
overall=0
for episode in train_00 validation_00; do
  input_dir="$DATA_DIR/v2e_input/$episode"
  output_h5="$OUTPUT_DIR/$episode.h5"
  output_folder="$OUTPUT_DIR/$episode"
  if [[ -e "$output_h5" || -e "$output_folder" ]]; then
    printf 'refusing existing V2E output for %s\n' "$episode" > "$RUN_DIR/$episode.preflight_error.txt"
    overall=1
    continue
  fi
  command=(python "$V2E_SOURCE/v2e.py" --input "$input_dir" --input_frame_rate 100 --auto_timestamp_resolution False --timestamp_resolution 0.000025 --slomo_model "$CHECKPOINT" --output_folder "$output_folder" --unique_output_folder False --skip_video_output --no_preview --dvs_h5 "$output_h5" --dvs_aedat2 None --dvs_aedat4 None --dvs_text None --pos_thres 0.2 --neg_thres 0.2 --sigma_thres 0.03 --cutoff_hz 300 --leak_rate_hz 0.01 --leak_jitter_fraction 0.1 --noise_rate_cov_decades 0.1 --shot_noise_rate_hz 0.001 --refractory_period 0.0005 --dvs_emulator_seed 42)
  printf '%q ' "${command[@]}" > "$RUN_DIR/$episode.command.txt"
  printf '\n' >> "$RUN_DIR/$episode.command.txt"
  if "${command[@]}" > "$RUN_DIR/$episode.stdout.log" 2>&1; then rc=0; else rc=$?; fi
  printf '%s\n' "$rc" > "$RUN_DIR/$episode.exit_code.txt"
  printf 'episode=%s exit=%s\n' "$episode" "$rc"
  if [[ $rc -ne 0 ]]; then overall=1; fi
done
printf '%s\n' "$overall" > "$RUN_DIR/pair.exit_code.txt"
exit "$overall"
