# R0.6 causal event-state offline replay

This is the g1 rerun of the offline diagnosis using raw Direct-HDR event streams from three saved GT-shadow trajectories: episode 0 (0.30 m/s, y=-55 mm), episode 4 (0.40 m/s, y=0), and episode 8 (0.50 m/s, y=+55 mm). These are sensor-reference trajectories controlled by delayed current GT. Their strict task success does **not** count as event-controller success.

## Provenance

- Code commit: `30177915e0e260d6161feb48dddf087809268f73`.
- Config SHA256: `aa97f6f5a12af92561ef053407bb795bba2801582958b4ad1134bcd24497435a`.
- Source result directory on g1: `~/Eventbased_WAM/repositories/gec-robot-r0_5-39b235a/benchmark/robot_closed_loop/results/r0_5_direct_hdr_gt_shadow_20261016/`.
- Source `SHA256SUMS` SHA256: `86cf6c0801365c0c1e5949e4e04d3178fa03f334997db6d09547f69a2b5af37c`.
- Selected raw event hashes: episode 0 `1427e38cea39dde2baa581dd8b2809abe7b5c6e99422ad38ab48c0ce2074ca54`; episode 4 `354b13c1b13c1ca9e085ae2768a53fecb606ad410d9cc5d9b529393ab93132a6`; episode 8 `e5207aa6d7d678bea2ad55d28c840956e4e40987bb6ef86287c96d53ffb02dca`.
- The `run_manifest.json`, `per_episode.json`, `causal_observation_polls.csv`, and `five_bin_sampling_comparison.json` hashes are recorded in `SHA256SUMS`; all four were verified on g1 and after transfer into this repository.

## Findings

The six-bin constant-velocity regression returned 0.1272727 m/s for a synthetic 0.4 m/s trajectory because its time vector and positions used inconsistent centering. The corrected least-squares fit returns 0.4000000 m/s. Position timestamps now use the latest contributing event-bin median, while causality uses the estimator generation time. The fixed-config camera projection round-trips a 25-point workspace grid to 2.2e-13 mm maximum numerical error. This only checks the reconstructed config geometry: the old renderer camera transform was not serialized, so this is not an independent renderer or EVK4 calibration.

The explicit status contract was exercised with synthetic observations: startup reports `no_event`, insufficient supported time bins reports `tracking_lost`, a recent estimate is `fresh`, and after silence it becomes `stale`; only `fresh` yields a usable motion state. The configured 30 ms maximum age is an engineering fail-closed value, not frozen. In saved real trajectories, age alone looked healthy but the spatial estimate did not: pre-contact position P95 averaged 18.4/20.0 cm and velocity P95 3.36/3.57 m/s at 1/2 kHz (mean of three episode-level P95s). From contact to lift the corresponding P95s were 7.93/8.46 cm and 1.77/2.18 m/s. All 21/21 contact-to-lift polls per rate were marked fresh, demonstrating that freshness is not a tracking-accuracy score.

The free-slide arrival reference is the evaluator-only arrival time predicted from the no-hand calibrated deceleration model. Only 51/194 (26.3%) and 65/194 (33.5%) pre-contact polls produced a reachable arrival prediction at 1/2 kHz. Conditional on those reachable predictions, absolute arrival error MAE was 0.296/0.288 s and P95 was 0.799/0.657 s. Ground truth is read only after estimates are formed for offline error calculation; it is never passed to the event tracker, fixed predictor, or event controller.

The sampling comparison uses the causal controller representation: five 5 ms time bins, 25 ms total history, two polarities, 64x64 spatial bins, 10 ms controller cadence, and a 10 ms observation cutoff. Across 417 windows all 1/2 kHz pairs were nonempty. The window-weighted event-count difference, `|n1-n2|/max(n1,n2)`, was 8.43%; weighted voxel cosine was 0.98597. This supersedes the earlier 10 ms single-bin cosine sanity check for deciding which input should be compared downstream, but it does not establish estimator accuracy or sufficient sampling.

An evaluator-only projection of event rays onto the target-center-height plane found only 13.8–17.6% of events within 80 mm of the GT cube center. Parallax remains in this diagnostic. Together with the large state errors, this indicates that the broad-ROI event centroid is dominated by robot/background activity and the fixed-height projection. It remains a plumbing smoke, not an estimator suitable for control.

## Replay command on g1

```bash
cd /home/zhaoyuhan/Eventbased_WAM/repositories/genesis-event-camera-plugin-r0_6-b3dc69d
PYTHONPATH=. /home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot/.venv-robot/bin/python \
  benchmark/robot_closed_loop/diagnostics/replay_event_state.py \
  --source-results /home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot-r0_5-39b235a/benchmark/robot_closed_loop/results/r0_5_direct_hdr_gt_shadow_20261016 \
  --out benchmark/robot_closed_loop/results/r0_6_event_state_offline_20261016 \
  --episodes 0,4,8
```

Large raw event arrays stay at the g1 source path above and are covered by its full checksum list. The downloaded result directory contains the small analysis products and their checksum list.
