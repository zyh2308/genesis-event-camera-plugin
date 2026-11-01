# Sparse reconstruction repair, 2026-11-01

Only actual 100Hz HDR endpoints plus depth, segmentation and endpoint camera/link poses may enter the generator. All intermediate radiance is reconstructed by the original capture -> PhysicsInterpolator -> projected geometry path. No dense HDR, render provider, capture_observed, reference event arrays or test labels are allowed during generation or training.

Candidate: replace next-primary instantaneous photometry and hard mid-interval unknown hold with time/coverage-weighted log-radiance fusion. Keep coverage support thresholds and unknown flags. Exact endpoints, identical sensor response and original adaptive schedule are required. This is endpoint-only approximation, not newly observed photometry.

Controls: static identity, known log-brightness ramp timing, exact endpoints, translating geometry, unknown-region honesty, and public capture call-path audit. Then generate all nine cached training/validation episodes; same heatmap_v2, seed606, 40 epoch cap, batch32, same split and checkpoint selection. Accept only if pixel median and P95 improve over old sparse Ours on the fixed common-input development bank, and confirm with frozen weights on the other three archived episodes. Failed candidates remain failed; dense-HDR positive control never counts as this repair.
