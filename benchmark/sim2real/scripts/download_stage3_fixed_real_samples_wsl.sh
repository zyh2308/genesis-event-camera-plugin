#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=/home/科研/Eventbased_WAM/repositories/gec-sim2real
DATA_ROOT=/home/科研/Eventbased_WAM/data/sim2real/stage3/real
EVIDENCE_DIR="$REPO_DIR/benchmark/sim2real/evidence/S3/20261016-stage3/access_and_download"
mkdir -p "$EVIDENCE_DIR" "$DATA_ROOT/HQF" "$DATA_ROOT/MVSEC/indoor_flying"
LOG="$EVIDENCE_DIR/download_wsl.log"
RC_FILE="$EVIDENCE_DIR/download_wsl.exit_code.txt"
date -u +%FT%TZ > "$EVIDENCE_DIR/download_wsl.started_utc.txt"
trap 'rc=$?; printf "%s\n" "$rc" > "$RC_FILE"; date -u +%FT%TZ > "$EVIDENCE_DIR/download_wsl.finished_utc.txt"' EXIT

{
  printf 'python='; python3 --version
  printf 'gdown='; python3 -m gdown --version
  printf 'disk='; df -h /home
  printf 'official_HQF_folder=https://drive.google.com/drive/folders/18Xdr6pxJX0ZXTrXW9tK0hC3ZpmKDIt6_\n'
  printf 'official_MVSEC_HDF5_folder=https://drive.google.com/drive/folders/1rwyRk26wtWeRgrAx_fgPc-ubUzTFThkV\n'
} > "$EVIDENCE_DIR/download_wsl.environment.txt" 2>&1

download_file() {
  local file_id="$1"
  local output_path="$2"
  if [[ -e "$output_path" ]]; then
    echo "refusing to overwrite existing download: $output_path" >&2
    return 3
  fi
  printf 'COMMAND python3 -m gdown --id %q -O %q --continue\n' "$file_id" "$output_path" >> "$EVIDENCE_DIR/download_wsl.command.txt"
  python3 -m gdown --id "$file_id" -O "$output_path" --continue
  test -s "$output_path"
}

download_file 1WW2hz9OebgCnAmTJjFTtCYWQEzjkDQYj "$DATA_ROOT/HQF/bike_bay_hdr.bag" 2>&1 | tee -a "$LOG"
download_file 18lISHaEtIHxTETuuNHBRHPTtsVhq29l_ "$DATA_ROOT/MVSEC/indoor_flying/indoor_flying1_data.hdf5" 2>&1 | tee -a "$LOG"
download_file 1U9NChDGHdBGVSZrQJeQLSG0_i_z3NquN "$DATA_ROOT/MVSEC/indoor_flying/indoor_flying1_gt.hdf5" 2>&1 | tee -a "$LOG"
download_file 1dZNctjh7i_Myq-wACiUuBPli9FR7d96U "$DATA_ROOT/MVSEC/indoor_flying/indoor_flying_calib.zip" 2>&1 | tee -a "$LOG"

sha256sum \
  "$DATA_ROOT/HQF/bike_bay_hdr.bag" \
  "$DATA_ROOT/MVSEC/indoor_flying/indoor_flying1_data.hdf5" \
  "$DATA_ROOT/MVSEC/indoor_flying/indoor_flying1_gt.hdf5" \
  "$DATA_ROOT/MVSEC/indoor_flying/indoor_flying_calib.zip" \
  > "$EVIDENCE_DIR/download_wsl.sha256"
du -b \
  "$DATA_ROOT/HQF/bike_bay_hdr.bag" \
  "$DATA_ROOT/MVSEC/indoor_flying/indoor_flying1_data.hdf5" \
  "$DATA_ROOT/MVSEC/indoor_flying/indoor_flying1_gt.hdf5" \
  "$DATA_ROOT/MVSEC/indoor_flying/indoor_flying_calib.zip" \
  > "$EVIDENCE_DIR/download_wsl.sizes.txt"
