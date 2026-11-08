# CPU-only full-packet subwindow diagnostic

2026-11-08. Diagnostic scripts only. No production edits, SSH, rendering or GPU work.

## Scientific / causal decision

The candidate is reasonable to test: the old detector sees only the last 12 ms of each 40/80 ms original event packet, while discarding the other available event times for detection. After an entire packet arrives, a batch estimator can process its internal timestamps in order and return a current estimate. This uses more of the already-generated event stream without claiming earlier information arrival.

The important limits are explicit:

- Every child window keeps the original packet's `available_s`. No extra controller queries are introduced; previous controller outputs are not retroactively revised.
- The original 0.30 m association gate, 50 ms two-observation confirmation gap, fit code, gravity and velocity prior remain unchanged.
- The candidate camera pose at a child-window endpoint is an **approximation**: rotation Slerp and linear translation between its enclosing HDF camera endpoints. Both enclosing endpoints must already be available when the original packet is consumed. This is the actual sensor camera pose history, not hand pose or target GT.
- `world` position and `camera_origin` use the same child-window transform through the existing native detector and Kalman wrapper. Camera pose is evaluated at child-window end; the median event time can be slightly earlier. This remaining approximation is recorded rather than treated as exact rendered camera evidence.
- All child windows carry the actual `parent_packet_id` from the original packet audit. Multiple detections from one parent are **not independent renders or independent real-sensor packets**. Their errors can be correlated.
- Splitting and approximate within-packet camera geometry are a combined diagnostic intervention. Its result alone cannot isolate which component causes a change.
- Existing physical trajectories are replayed; estimator outputs do not change the archived actions. No catch-success improvement can be claimed from this replay.

## Three layers prevent a version mix-up

Old normal25 archives predate the tentative-association correction. In particular old c02 Ours contains 17 detections and no onset; the current source produces 22 with onset on the same raw events. Therefore a current-native estimator is not expected to equal every old archived state.

The script runs, for every supplied case:

1. **`original`**: old production association (`--archived-association public` default), fit0, original whole packets. Requires exact archived sample count and positions, state positions/velocities, and all logged sample fields. Failure stops the experiment. This establishes input/availability replay correctness.
2. **Native whole-packet baseline**: actual source hook with unchanged original packets and the chosen common fit flag.
3. **Native 10 ms subwindow candidate**: same native source/fit flag, plus chronological non-overlapping subwindows and camera-pose interpolation. Compare this only with layer 2.

The archive baseline is a prerequisite, not an extra competitor. The choice of archived association version is explicit and does not depend on GT scores or success labels. If using newer archives that actually employed the native tentative hook, pass `--archived-association native` instead.

## Files and deployment

Place these files together in an isolated CPU diagnostic directory:

- `replay_packet_subwindows.py`
- Existing `replay_static_association.py` (shared, previously used actual raw replay helper; this task did not edit it)

Use the existing server Python that has `h5py`, `numpy`, `scipy` and OpenCV. No Genesis, torch, CUDA or new HDR is loaded. Example, after filling the actual source/data locations from the existing run manifests:

```bash
python replay_packet_subwindows.py \
  --source /absolute/path/to/sealed/current/benchmark/mature_catch \
  --output /absolute/path/to/new/subwindow8_output \
  --until 1.45 --flight-fit --archived-association public \
  --run /actual/old/c01_normal25/ours --run /actual/old/c01_normal25/v2e \
  --run /actual/old/c02_normal25/ours --run /actual/old/c02_normal25/v2e \
  --run /actual/old/c03_normal25/ours --run /actual/old/c03_normal25/v2e \
  --run /actual/old/c04_normal25/ours --run /actual/old/c04_normal25/v2e
```

`--flight-fit` enables the existing fit1 for **both** native baseline and candidate. Without it both use fit0. The archived exact-parity layer always preserves original fit0. The current request is a fixed common fit1 diagnostic; do not sweep this flag while selecting a favorable result.

The actual reset audit contains an empty packet at t=0, available at 10 ms. It is retained in parent accounting with zero child windows; it requires no preceding camera endpoint. Nonempty reset packets are rejected because they have no valid interpolation bracket.

## Outputs / interpretation

`SUMMARY.json` has all 24 case-mode results after eight inputs, no filtering of failed or unfavorable cases. It reports onset, measurement count, false onset before true release, and precontact position / velocity / range / transverse errors (median and P95). GT is used only in this grading stage, never in detection or pose interpolation.

Each candidate JSON adds `subwindow_history`: parent ID, original availability, child time bounds, event counts, pose source, enclosing endpoint availability, interpolated camera transform, and returned detector sample. `samples[]` is authoritative. The generic helper's `packets[].detected` is intentionally `None` for candidate mode because multi-output samples were already appended inside the wrapper; this avoids duplicating the last sample.

`PARTIAL_SUMMARY.json` preserves completed modes if a later assertion fails. `SOURCE_HASHES.json` captures the script, helper and all relevant estimator source hashes. The tool asserts that each recorded controller state contains only measurements whose original parent packets have arrived, and that event timestamps do not exceed availability.

## Local checks actually completed

`test_replay_packet_subwindows.py`: **6/6 PASS**, local CPU, with:

1. No duplicate/dropped events at child-window boundaries.
2. Exact endpoint preservation and a known rotation/translation midpoint.
3. Rejection of a not-yet-available camera endpoint.
4. Multi-output append exactly once and unchanged packet availability.
5. Existing native contour/Kalman/RGB-D classes exercised on synthetic static and moving sphere events; static remains unconfirmed and sufficiently moving data confirms using the original rule.
6. The actual empty-reset-packet format.

These are synthetic/tool checks, **not actual cached raw parity or new closed-loop validation**. `SUBWINDOW_SYNTHETIC_TEST_STATUS.json` records that distinction. Root should run the actual eight caches before any online deployment decision. No production source file was changed.
