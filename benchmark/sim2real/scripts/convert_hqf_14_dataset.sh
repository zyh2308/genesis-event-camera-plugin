#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 BAG_ROOT MEMMAP_ROOT MANIFEST_OUT" >&2
  exit 2
fi

bag_root=$(realpath "$1")
memmap_root=$(realpath -m "$2")
manifest_out=$(realpath -m "$3")
repo_root=$(cd "$(dirname "$0")/../../.." && pwd)
python_bin=${PYTHON_BIN:-python3}
converter="$repo_root/benchmark/sim2real/scripts/convert_hqf_bag_rosbags.py"
builder="$repo_root/benchmark/sim2real/scripts/build_hqf_14_file_manifest.py"
sequences=(
  bike_bay_hdr boxes desk desk_fast desk_hand_only desk_slow
  engineering_posters high_texture_plants poster_pillar_1 poster_pillar_2
  reflective_materials slow_and_fast_desk slow_hand still_life
)

for sequence in "${sequences[@]}"; do
  bag="$bag_root/$sequence.bag"
  [[ -f "$bag" ]] || { echo "missing registered HQF bag: $bag" >&2; exit 3; }
done
if [[ -d "$memmap_root" ]] && find "$memmap_root" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  echo "refusing to overwrite nonempty HQF MemMap root: $memmap_root" >&2
  exit 4
fi
mkdir -p "$memmap_root"

for sequence in "${sequences[@]}"; do
  echo "[$(date -u +%FT%TZ)] convert $sequence"
  "$python_bin" "$converter" \
    --bag "$bag_root/$sequence.bag" \
    --output-dir "$memmap_root/$sequence"
done

"$python_bin" "$builder" \
  --bag-root "$bag_root" \
  --memmap-root "$memmap_root" \
  --output "$manifest_out" \
  --require-complete
