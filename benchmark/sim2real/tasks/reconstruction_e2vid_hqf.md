# Reconstruction route: E2VID → HQF

Status: Stage-2 paired multi-rollout adapters and training pilot are implemented; g1 run is pending. No HQF data were downloaded or tested.

- Model: E2VID recurrent architecture using the pinned E2VID+ synthetic training implementation/config.
- Real benchmark candidate: all 14 HQF sequences; HS-ERGB remains extension-only. The frozen research protocol selected HQF because it is reconstruction-specific and EVREAL has a native HQF config, whereas HS-ERGB originates from TimeLens video interpolation and needs an additional reconstruction-time/alignment definition. This is the protocol basis for review, not an engineering claim that HQF is universally superior.
- Candidate event input: signed 5-bin voxel, with identical train/validation behavior across generators.
- Stage-1 review finding R1: a two-frame sequence did not invoke the pinned temporal-consistency loss. The Stage-2 adapter uses three-frame sequences and records temporal-loss call count, per-component loss, and gradient evidence.
- Stage-1 review finding R2: labels were previously divided by observed event span, making them generator-dependent and degenerate for empty/sparse windows. Stage 2 stores direct per-endpoint physical flow labels and a common geometry validity mask, overriding the pinned loader's legacy event-span scaling. Boundary fixtures cover empty events, two-event sparse input, and tied timestamps.
- Stage 2 uses six whole physical rollouts (four train, two independent validation), pair-matched initialization and two epochs. The pending run will also check recursive state handling and checkpoint restoration. Exact results and artifact hashes will be linked in the Stage-2 report/evidence index.
- HQF's public route is listed by the event-flow source; no HQF file was found in the inspected task data roots. This is not a global unavailability claim. Verify the complete 14-sequence package, license, and EVREAL preprocessing/evaluation manifest before formal training/testing.
- Real data remains test-only. No HQF model score is authorized during this phase. The manifest pins EVREAL's HQF MemMapDataset/eval.py route and the 14-sequence inventory; actual files, license, and candidate-method wrapper still need verification before real testing.
