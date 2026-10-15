# Reconstruction route: E2VID → HQF

Status: synthetic feasibility loop passed on g1. No HQF data were downloaded or tested.

- Model: E2VID recurrent architecture using the pinned E2VID+ synthetic training implementation/config.
- Real benchmark: HQF, complete 14-sequence set; HS-ERGB is extension-only.
- Candidate event input: signed 5-bin voxel, with identical train/validation behavior across generators.
- The actual tiny pilot used five bins and a benchmark-local adapter for the pinned source's `DynamicH5Dataset` class name, CSV file manifests, tied-timestamp search, short validation sequence, and matched image normalization.
- One update and synthetic validation passed for each generator with the same seed and initial weights. The loss values and timing are smoke observations, not a generator comparison; full values and hashes are in `reports/stage1_feasibility.md` and the Stage-1 evidence bundle.
- HQF's public route is listed by the event-flow source; no HQF file was found in the inspected task data roots. This is not a global unavailability claim. Verify the complete 14-sequence package, license, and EVREAL preprocessing/evaluation manifest before formal training/testing.
- Real data remains test-only. No HQF model score is authorized during this phase.
