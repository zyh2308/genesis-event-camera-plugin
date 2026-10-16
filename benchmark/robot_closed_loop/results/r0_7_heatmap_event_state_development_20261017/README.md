# R0.7 heatmap event-state development fit

This is a bounded architecture diagnostic on the same nine Direct-HDR GT-shadow development episodes. The episode split and event windows match the v1 spatial MLP fit; this is not a new held-out test, generator comparison, or model ranking.

## Provenance

- Training code commit: `1b2c80ed4ef80d8a3bcb3bc8679de1298659a49b`.
- Branch at training: `exp/robot-closed-loop-v1`.
- Scene config SHA256: `aa97f6f5a12af92561ef053407bb795bba2801582958b4ad1134bcd24497435a`.
- Source GT-shadow manifest SHA256: `86cf6c0801365c0c1e5949e4e04d3178fa03f334997db6d09547f69a2b5af37c`.
- Split: train episodes `0,2,3,4,7,8`; validation episodes `1,5,6`. Both splits span the three speeds and lateral offsets with disjoint speed-position cells; windows never cross episode boundaries. All nine episodes remain development material.
- Checkpoint `spatial_event_state.pt` SHA256: `64b120afde4579563d8c18f4430eeaab00c3829adaa682fa82f321fac496ea7a`.
- Original output `SHA256SUMS` SHA256: `efbebacaf24eb3c3f9c028b1e8c679cfe4933c4c36b0149f3bcb9f36df94d689`; all listed outputs verify.
- Training log SHA256 `755f0f950c6721ed631f91edb246d964de41fe1bf2ce84415765bcb6f47ec72b`; runtime SHA256 `af1492eeeb7a14aa295a846285366a32c4bc9268025cb185f0fed7806d2477f4`; exit marker SHA256 `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`.

## Model and split validation

The 35,261-parameter `event-heatmap64-aux-z-velocity-v2` model predicts a 64×64 target-center heatmap and uses softargmax plus the fixed pinhole calibration and predicted height to recover world XY. A separate head predicts height and velocity. Inference input remains only the same five causal 5 ms bins of two-polarity events. GT pose/velocity supplies training labels and evaluator metrics only.

Training used CPU, two threads, seed 606, batch 64, best epoch 53 and 942.41 s fit-loop time. The full process took 15:56.39 wall, 1168.76 s user, 703.37 s system and 3,811,124 KB maximum RSS; exit 0.

Validation has 417 windows: 195 pre-contact, 22 contact-to-lift and 200 post-lift. Phase-wise XY position P95 is `22.0 / 49.6 / 69.5 mm`; XY velocity P95 is `0.201 / 0.218 / 0.052 m/s`. All 417 windows passed the event-count/bin freshness rule at 10 ms P95 age. Same-fixed-predictor pre-contact arrival error P95 is `319.3 ms`; predicted intercept Y error P95 is `10.4 mm`.

Compared with v1 on the same split, pre-contact position P95 improved from `59.4` to `22.0 mm`, pre-contact velocity P95 from `0.242` to `0.201 m/s`, arrival P95 from `376.2` to `319.3 ms`, and intercept Y P95 from `11.6` to `10.4 mm`. Contact-to-lift and post-lift position errors were worse than v1 (`49.6` vs `24.5 mm`, and `69.5` vs `30.1 mm`, respectively); the contact-to-lift set has only 22 windows. Fresh status still does not certify localization accuracy.

The checkpoint improves the pre-contact spatial diagnostic but remains weak on arrival timing. It is used for a separate three-seed real closed-loop development rollout, not frozen for generator comparison.

## Reproduction

Use [`../runtime_logs/r0_7_heatmap_event_state_development_20261017.launch.sh`](../runtime_logs/r0_7_heatmap_event_state_development_20261017.launch.sh) on g1. It disables CUDA, uses two CPU threads, records PID/log/exit/runtime, and reads the source episodes from the preserved g1 GT-shadow path.
