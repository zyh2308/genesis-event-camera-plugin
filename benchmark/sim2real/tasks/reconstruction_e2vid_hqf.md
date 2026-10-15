# Reconstruction route: E2VID → HQF

Status: source audit only; no code/data execution yet.

- Model: E2VID recurrent architecture using the pinned E2VID+ synthetic training implementation/config.
- Real benchmark: HQF, complete 14-sequence set; HS-ERGB is extension-only.
- Candidate event input: signed 5-bin voxel, with identical train/validation behavior across generators.
- Must verify before running: pinned trainer source completeness; public config's `num_bins=10` versus loader `5`; EVREAL E2VID+ method and HQF dataset path; no real test use during training/checkpoint selection.
- Required smoke: paired toy clips from identical Genesis trajectories and image/flow labels; loader and one short synthetic-only train/val for each generator.
- Real data remains test-only. No HQF model score is authorized during this phase.
