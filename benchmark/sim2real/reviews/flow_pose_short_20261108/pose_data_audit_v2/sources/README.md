# CPU read-only pose data contract audit

v2 differs from the preserved v1 only by caching each NPZ array once per rollout, preventing repeated full-image decompression. No data, statistical formula, threshold, model input or scientific check changes.

No render, event replay, model inference, training, real dataset access or GPU operation is launched. Output must be a new JSON file. All source inputs remain read-only. Use an existing CPU terminal/job with CUDA_VISIBLE_DEVICES empty, two CPU threads and a bounded timeout.

## Source evidence already checked

`prepare_pose_full_train_radiance.py` asserts96 total original TRAIN trajectories and selects **all96**, unlike the earlier8-trajectory pilot. Both original normal adapter and fresh adapter use indices4..19 (sixteen10ms event windows per trajectory), GT at endpoint k+1, and the same native encoder. Thus1529 rows must not be interpreted as384 different train rollouts or a96/384 auxiliary subset. Actual artifacts still need the checks below.

Relighting reads existing cached radiance, multiplies it by0.5/0.125 and reapplies the original tone function; timestamps, camera/object geometry, depth and segmentation are external links to the original physical H5. It does not rerender Genesis. Completing existing all96 TRAIN radiance coverage is already the registered full-bank operation, not a new extension. New geometric trajectories would require new physical caches, but that is not proposed by this audit.

## Checks

- All physical TRAIN IDs versus relight selected IDs; normal NPZ IDs versus fresh base and both relit banks.
- Every actual start/end timestamp, K, CAD keys, pose and UV; normal NPZ label endpoint metadata is separately reported because the training loader ignores that metadata.
- Geometry distributions: true camera Z, center, projected scale, orientation, translational/angular/screen motion. Statistics are diagnostics, not a root-cause conclusion.
- With `--representations`, original normal raw events are SHA-checked and reencoded through the exact AST-extracted native encoder; every NPZ image is compared bitwise. Equality before the shared deterministic area128 transform implies equal downstream inputs.
- Generator manifest/profile/core/cadence and V2E timestamp/threshold/noise parameter fields are compared across normal/half/eighth. Raw physical-manifest hashes correctly differ between relit banks and must match each bank's own manifest.
- With `--input-information`, report ON/OFF/TOTAL nonzero rates, raw count quantiles (whole region and active pixels), count>8 and count>=8 rates, native encoded exact-one rate and actual full-frame area128 exact-one rate. Encoding already saturates at8; do not call only>8 the full saturation rate.
- Input statistics separate TRAIN and the four independent relit validation trajectories, true-Z bins, full sensor, projected CAD bbox and outside bbox. GT bbox is used only for statistics, never as a model crop or input mask. Quantiles are computed per region/window and then summarized across windows; aggregation is explicitly window-equal.

Expected sampling exposure is calculated from **actual** matched row counts: if normal and each relit level all have1529 windows, six epochs give normal6 visits/window, half4.5 expected visits/window and eighth1.5. The3:1 ratio is half versus eighth, not threefold half versus normal. Actual recorded augmentation/log exposure is a separate question.

## CLI

`python -B audit_pose_data_contract.py --campaign R --radiance-train FULL_TRAIN/radiance_train --radiance-validation LIGHT_ROBUSTNESS/radiance --representations --input-information --out NEW_FILE.json`

CPU fixtures verify exact histogram quantiles versus independent NumPy, distinguish saturation at8 from>8, run complete raw→NPZ/window/GT checks, deliberately alter one image and verify mismatch detection, and exercise raw/input/bbox information statistics without a model or GPU. Real artifact results are pending; no diagnosis is prewritten.
