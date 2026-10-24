# E-POSE block/r_01_gl sample semantic audit

Scope: local, read-only follow-up using the captured g1 audit summary and the
local official E-Pose CAD/export source. No SSH, download, training, inference,
or real-data scoring was performed. Raw g1 HDF5/MAT binaries are not present in
this local workspace, so event-row match counts below are attributed to the
captured g1 audit; this script independently recomputes the reported time
offset arithmetic, CAD correspondence, 3-D transform, and projection.

## Reproduce

From the repository root:

```bash
python3 benchmark/sim2real/scripts/audit_epose_block_sample_semantics_local.py
```

The script reads:

- `benchmark/sim2real/evidence/20261024-execution-refresh/g1_sample_audit_stdout.txt`
- `/home/科研/Eventbased_WAM/external_sources/E_Pose/CreateSaveDataset.m`
- `/home/科研/Eventbased_WAM/external_sources/E_Pose/xyz_data/wood block.xyz`
- `/home/科研/Eventbased_WAM/external_sources/E_Pose/obj_files/wood_block/google_16k/nontextured.ply`

Python dependencies are NumPy and OpenCV. OpenCV is used only to reproduce the
exporter's legacy undistortion step; the script separately computes the
correct pinhole projection directly from the captured camera-frame 3-D points.
Output: `epose_block_r01gl_semantic_audit_v1.json`.

## Results

- The captured g1 summary identifies the matched files as
  `.../block/r_01_gl/0001_data.mat` and
  `.../original_data/block/r_01_gl/data.h5`. It reports the MAT interval
  `(0.061367988586, 0.07137298583984375] s` relative to the first H5 event at
  `1715713676.2185817 s`. The independently recomputed absolute endpoints
  exactly equal the reported endpoints, and the MAT endpoint equals both its
  label time and last event timestamp.
- The g1 summary reports 717 events in the H5 interval and 717 in the MAT,
  with 717/717 `(x,y,p)` matches within 20 microseconds and median/max nearest
  timestamp error of zero. These row-level comparisons are not re-run locally:
  only the summary, not raw event arrays, is available here.
- `wood block.xyz` and the official `wood_block/google_16k/nontextured.ply`
  have 8,194 vertices in the same order, with a maximum coordinate difference
  of `1.0e-8 m`. Their bounding dimensions are approximately
  `[0.103917, 0.103464, 0.206003] m`. This establishes the CAD point-cloud
  correspondence for the block object.
- Transforming the CAD AABB corners by the MAT `pose` reproduces the MAT's
  camera-frame keypoint XYZ with max absolute component error
  `1.694e-5 m` (max Euclidean error `1.786e-5 m`). The official script's
  rounded object-in-base transform combined with inverse MAT `cameraPose`
  reproduces the MAT pose within `0.84 mm` translation and `0.00189` rotation
  Frobenius residual. The point-cloud scale and ROS-style translations are
  strongly consistent with metres, although no explicit unit field is present.
- Reprojecting each stored 3-D camera point with its own depth and MAT K yields
  stored-UV residual mean `16.450 px`, max `32.744 px`. Recreating the source's
  use of corner 1's depth for every corner, then applying the listed distortion
  correction, yields residual mean `0.166 px`, max `0.349 px`. The source
  lines 268–275 confirm points 2–9 all divide by `cam_point1(3)`. This strongly
  confirms that the saved 2-D keypoint labels contain the shared-depth bug.
- The source line 308 assigns each sample's `cameraPose` from
  `T_dvs_r_base(:,:,2)`, a fixed index inside the sample loop. The paired H5
  camera positions span about `[0.397, 0.345, 0.223] m` over the recording, so
  this field cannot represent all moving-camera samples. The first captured
  MAT sample is internally consistent with its `cameraPose`; whether that
  fixed field is already wrong for this first sample requires checking a
  second MAT sample or camera pose at this sample's exact timestamp.
- Official source creates MAT `events` as `[x,y,t,p]` from the raw event frame
  buffer before the object mask is applied. The mask is used for segmented
  image output. H5 is `[x,y,p,t]`; the two layouts must be converted explicitly.

## Interpretation and next data gate

The block/wood-block object and the MAT pose plus 3-D keypoint geometry are
well supported by this sample-level evidence. The exporter bug in 2-D keypoint
projection is also supported. Preserve the original files unchanged; do not use
`keypointprojections(:,1:2)` as trusted targets. Recompute projections from the
verified CAD, MAT pose, and K, and audit another timestamp to establish the
`cameraPose` behavior before using that field. A full pose training/real
evaluation interface has not been run.
