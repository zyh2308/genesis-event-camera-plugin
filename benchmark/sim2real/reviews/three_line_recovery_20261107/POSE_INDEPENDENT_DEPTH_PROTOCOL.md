# Independent depth, matched illumination consistency — 2026-11-07

This is a synthetic development experiment, not a new simulator release or a final E-POSE result.

## Evidence and choice

The preceding physical-radiance augmentation improved EventGenesis depth at one-eighth radiance, but degraded half-radiance performance. Prediction swaps, with no GT substituted into predictions, attributed half-radiance success losses to both center and depth changes. Freezing the entire shared feature extractor did not satisfy the existing gate. A larger new backbone would confound this diagnosis and cost more training. We therefore separate the final localization path from the trainable depth path, retaining the existing CenterScale architecture in both paths.

## Fixed protocol

- Frozen simulator core: `62da4f35ce3c5c96f04268bd1687a27657af9238`; sensor/noise/occlusion unchanged.
- Both generators, seeds 0/1/2, identical data identities and augmentation draws within each seed. Starting weights have generator-specific synthetic training histories; they are identical across arms within a generator/seed, not across generators.
- Reuse the 1,529 original training windows and eight disjoint training trajectories at half/eighth linear radiance. Relighting is radiance attenuation, not a calibrated lux condition. Never use real data.
- Freeze the physical-study clean-control checkpoint as final localizer. Clone it into a trainable depth model. Keep the original frozen rotation reference identical across all arms. Final center always comes from the localizer; final depth comes from the depth model.
- Three arms: normal auxiliary control; physically relit supervised auxiliary; physically relit auxiliary plus matched-normal teacher consistency. The latter uses the SAME physical window and SAME affine transform for the two illumination views, including transformed intrinsics and labels.
- Six epochs, batch 32, 288 optimizer updates per fit, AdamW LR 0.00015, weight decay 0.01, clipping 5. Preserve the preceding normal-view teacher loss. Paired consistency weight 2 on log-size and log-depth differences. All arms compute the same teacher passes; only the loss term differs.
- Evaluate the existing 96 synthetic validation trajectories and four disjoint relit trajectories. These are inspected development data, not a fresh blind holdout. Fixed last-epoch checkpoints, no real-based selection.

## Acceptance and safeguards

Retain all original 18 per-generator/seed checks: clean ADD-S@0.1D decline at most 2 percentage points; strictly lower depth MAE at BOTH half and eighth radiance. Additionally require no low-light pose-success decline for either relit condition in any seed (12 checks). All must pass before real evaluation is considered. No relaxing gates or reporting only successful seeds. Supervised-only arm is a diagnostic ablation, not a fallback selected using real ranking.

Before GPU training: test matched transforms/labels, frozen-localizer identity, finite gradients and strict checkpoint round-trip using procedural CPU inputs. Then paired real-data-free server smoke tests. Use a fresh output root, source hashes, bounded runtime, atomic GPU ownership, automatic release. Preserve all failures. Existing reconstruction/flow jobs remain untouched.

If this gate fails, report that the hypothesis was insufficient. Do not promise a V2E win. Any later E-POSE woodblock evaluation is a development diagnostic because those sequences have already been inspected.
