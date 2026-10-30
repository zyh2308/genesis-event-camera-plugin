#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/zhaoyuhan/Eventbased_WAM
RUN_ROOT=$PROJECT/runs/sim2real/20261031-stage7-replicad-multiaxis-pathpilot-v1
TMP_ROOT=/tmp/sim2real-stage7-replicad-multiaxis-pathpilot-v1
ASSET_ROOT=$PROJECT/external/sim2real_assets/replicacad_interactive_v1_5/extracted_v1_5
PY=$PROJECT/../miniconda3/envs/Eventbased_WAM_torch28/bin/python
GENESIS_ROOT=$PROJECT/runtime/sim2real/genesis_hdr_v2
EXEC=$PROJECT/execution_checkouts/sim2real-stage7-flow-resume-d0e3679-20261030
SCRIPT=$RUN_ROOT/replicad_layout_camera_motion_pilot.py
SCENES=(v3_sc0_staging_00 v3_sc1_staging_00 v3_sc2_staging_00)
exec > >(tee -a "$RUN_ROOT/driver.log") 2>&1

date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_ROOT/started_utc.txt"
LOAD1=$(awk '{print $1}' /proc/loadavg)
NCPU=$(getconf _NPROCESSORS_ONLN)
MEM=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
if ! awk -v l="$LOAD1" -v c="$NCPU" 'BEGIN {exit !(l < c * .8)}'; then
  echo "CPU gate refused: load=$LOAD1 cpus=$NCPU" >&2; exit 5
fi
if [[ "$MEM" -lt 16777216 ]]; then
  echo "memory gate refused: available_kb=$MEM" >&2; exit 5
fi
test -x "$PY" && test -d "$ASSET_ROOT" && test -f "$SCRIPT"
test -f "$GENESIS_ROOT/genesis/vis/camera.py"
mkdir -p "$TMP_ROOT"
{
  printf 'host=%s\nproject=%s\n' "$(hostname)" "$PROJECT"
  printf 'execution_checkout_commit='; git -C "$EXEC" rev-parse HEAD
  printf 'renderer_script_sha256='; sha256sum "$SCRIPT"
  printf 'registration_sha256='; sha256sum "$RUN_ROOT/multiaxis_paths_registration_v1.json"
  printf 'local_source_manifest_sha256='; sha256sum "$RUN_ROOT/local_source_manifest.json"
  cat "$RUN_ROOT/local_source_manifest.json"
  printf 'python=%s\nscene_asset_root=%s\nload1=%s\ncpu_count=%s\nmem_available_kb=%s\n' "$PY" "$ASSET_ROOT" "$LOAD1" "$NCPU" "$MEM"
  printf 'device_mode=CPU software rasterizer; CUDA_VISIBLE_DEVICES empty\n'
  printf 'path_id=camera_translate_habitat_y_040mm_v1; translation_habitat_xyz_m=0,0.04,0\n'
  nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
  df -h "$RUN_ROOT" /tmp
  sha256sum "$ASSET_ROOT/../ReplicaCAD_dataset_v1.5.zip" \
    "$ASSET_ROOT/configs/scenes/v3_sc0_staging_00.scene_instance.json" \
    "$ASSET_ROOT/configs/scenes/v3_sc1_staging_00.scene_instance.json" \
    "$ASSET_ROOT/configs/scenes/v3_sc2_staging_00.scene_instance.json" \
    "$GENESIS_ROOT/genesis/vis/camera.py" "$EXEC/genesis_event_plugin/physics_warp.py" "$PY"
} > "$RUN_ROOT/preflight_and_inputs.txt" 2>&1

for scene in "${SCENES[@]}"; do
  OUT_TMP=$TMP_ROOT/$scene/out
  CACHE=$TMP_ROOT/$scene/cache
  RUN=$RUN_ROOT/$scene
  mkdir -p "$RUN" "$CACHE/quadrants" "$CACHE/genesis"
  test ! -e "$OUT_TMP"
  mkdir -p "$OUT_TMP"  # Genesis pilot creates its requested output directory itself; remove this empty reservation below.
  rmdir "$OUT_TMP"
  printf '%s\n' "$scene" > "$RUN/scene_id.txt"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/started_utc.txt"
  set +e
  timeout 900s env CUDA_VISIBLE_DEVICES= LIBGL_ALWAYS_SOFTWARE=1 \
    MESA_LOADER_DRIVER_OVERRIDE=llvmpipe PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
    QD_OFFLINE_CACHE=1 QD_OFFLINE_CACHE_FILE_PATH="$CACHE/quadrants" \
    QD_OFFLINE_CACHE_CLEANING_POLICY=never QD_NUM_THREADS=2 \
    GS_CACHE_FILE_PATH="$CACHE/genesis" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=1 PYTHONPATH="$GENESIS_ROOT:$EXEC:$PROJECT" \
    "$PY" -u "$SCRIPT" --asset-root "$ASSET_ROOT" \
      --scene-config "$scene.scene_instance.json" --output-dir "$OUT_TMP" \
      --translation-habitat-xyz 0 0.04 0 > "$RUN/stdout.txt" 2> "$RUN/stderr.txt"
  RC=$?
  set -e
  printf '%s\n' "$RC" > "$RUN/exit_code.txt"
  if [[ -d "$OUT_TMP/camera_motion_30ms" ]]; then
    cp -a "$OUT_TMP/camera_motion_30ms" "$RUN/"
    for file in result.json layout_rgb.png layout_depth.npy layout_segmentation.npy; do
      [[ ! -f "$OUT_TMP/$file" ]] || cp "$OUT_TMP/$file" "$RUN/"
    done
  fi
  date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/finished_utc.txt"
  if [[ "$RC" -ne 0 ]]; then
    echo "scene $scene failed with exit $RC; preserving this attempt and stopping the batch" >&2
    exit "$RC"
  fi
  "$PY" - "$RUN" <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()
items = []
for path in sorted(p for p in root.rglob('*') if p.is_file()):
    items.append({'path': str(path.relative_to(root)), 'size_bytes': path.stat().st_size, 'sha256': digest(path)})
(root/'artifact_manifest.json').write_text(json.dumps({'files': items}, indent=2, sort_keys=True)+'\n')
print(json.dumps({'scene': root.name, 'files': len(items), 'bytes': sum(x['size_bytes'] for x in items)}))
PY
  rm -rf "$OUT_TMP"
done

date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_ROOT/finished_utc.txt"
printf '0\n' > "$RUN_ROOT/exit_code.txt"
echo 'all three registered Habitat-Y camera paths completed'
