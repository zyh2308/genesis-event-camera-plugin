# Stage 4 corrected v2 evidence

## Current result

The original strict continuous-versus-process-resumed trajectory comparison failed with exit 1 and is retained as failed evidence. A deterministic-algorithm guard on the actual STE-FlowNet graph identified CUDA `grid_sampler_2d_backward_cuda` as an unsupported deterministic backward operation. Matching RNG state and paired data schedule therefore do not make two full CUDA trajectories a reliable exact replay oracle. No training implementation or protocol was changed to hide this.

The actual checkpoint resume from update 198 to 199 passed a state-boundary oracle. The oracle independently reconstructed the next eight sample IDs and augmentation sequences, verified the source model state, all 70 Adam step counters (198→199), unchanged scheduler position, changed model weights, finite gradients, and an independent forward-only loss from the source checkpoint. All three loss components matched the resumed log exactly. The paired EventGenesis and V2E update-199 logs used identical sample IDs and augmentation seeds.

The gate receipt key `resume_equivalence_epoch_boundary` names this checkpoint-boundary/next-step oracle. It does **not** establish strict full-trajectory equivalence, and the oracle does not independently reimplement Adam's parameter update. The strict continuous-versus-process-resumed replay remains a preserved failure; the boundary result is narrower evidence about checkpoint restoration and the next-step inputs/state.

The formal retry reran the complete preflight against current code, config, simulator, model, and synthetic-data fingerprints. All 16 checks passed. The production gate's five negative scenarios each exited 1 before creating a run root or reaching the trainer. Full-phase gate verification exited 0. The paired 45-epoch budget resumed from update 198 into new isolated directories and was active at the last check.

`resume_retry1_gate_verification.json` records the gate receipt, resume oracle, negative controls, PIDs, GPU assignment, observed update-200 checkpoints and latest metric snapshot. `artifact_index.json` gives server paths, sizes, and hashes for large artifacts and raw logs.

## Preserved pilot result

The paired one-epoch pilot completed 198 optimizer updates for each generator and passed its synthetic validation and epoch-boundary checkpoint checks. Its held-out synthetic EPE was `0.28709464` for EventGenesis and `0.28606818` for V2E versus `0.28420805` for zero flow (px/30ms). Both models were slightly worse than zero flow after one short epoch; this is only a pipeline diagnostic, not a generator ranking. Real-data scoring remains disabled.

## Provenance

- Branch: `exp/public-sim2real-v1`
- Current launch/oracle commit: `1115a8f6e2f5bac82a4b7428b37ec5006342355f`
- Frozen simulator core: `e79e656ac00f7847b98db9e12bf43b093f3afff1`
- g1 campaign root: `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261017-stage4-flowdt1-seed0-corrected-v5`
- Shared v3 corrected physical bank SHA256: `cca882ba912fda3b3243d49bc7e29e28186046e3b6591c424f385b0506047884`
- Real-data scoring: disabled
