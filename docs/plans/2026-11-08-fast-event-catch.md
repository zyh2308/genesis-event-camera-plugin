# Fast event catching diagnosis and verification plan

Goal: improve fast/backlit event-assisted catches while preserving Catch It! weights, Genesis physics, scoring, camera and the frozen sparse event generator.

Current branch: exp/robot-mature-catch-20261107, baseline a89fe60; immutable runtime source 64b22894fd7d. The user authorized direct implementation and server tests. g3 access was explicitly restored by the user; the older AGENTS temporary ban does not apply after that restoration.

## Design choices

1. Preferred: use the accurate first RGB-D position plus observed event bearings and a short physical flight model to estimate motion; do not repeatedly treat event-contour radius as accurate range. Estimate onset from observed movement rather than reading release time. Preserve the baseline as fallback until enough causal observations exist. Correct event-time camera pose using robot proprioception.
2. Preserve a separate early-hold experiment: continue receiving events and permit an assumed grasp to be revoked when observed motion contradicts it. Do not use target contact truth, target pose or target velocity as control inputs.
3. Reject blind process-noise tuning as the main route: the previous same-observation covariance probe did not help. Do not alter the frozen generator or retrain the original policy.

## Execution

- Reproduce all baseline state outputs from completed raw/cached records before comparing candidate errors. Inspect range/velocity bias and the timing of presumed hold versus actual contact (GT is scoring only).
- Implement the smallest event-state adapter candidate and check geometry, causal availability and future-data exclusion using controlled trajectories; keep missing/ill-conditioned observations explicit.
- Use server CPU cached replay to select a candidate for actual rollout; all such choices are development, not held-out evaluation.
- Deploy actual materialized immutable source; check loaded module hashes and frozen core hashes. Use fresh atomic GPU allocation on g3 if g1 remains full. No idle GPU holder.
- Run the two previously failed conditions plus normal regression; separate the motion and hold interventions when possible. Preserve every outcome, original checkpoint and task scoring.
- Save per-episode data, raw events, state/control trace, immutable source, commands, metadata, SHA256 manifests and actual Genesis videos. Deliver desktop MP4/GIF and a short report; release the GPU after jobs finish.

Primary methodological reference: Su and Shen, Catching a Flying Ball with a Vision-Based Quadrotor (ISER 2016), known gravity and noisy monocular bearings. This is an adapted event-input implementation, not a claim to reproduce their estimator exactly.
https://uav.hkust.edu.hk/wordpress/wp-content/uploads/2016/09/iser2016kunyue.pdf
