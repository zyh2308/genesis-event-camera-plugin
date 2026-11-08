# Subwindow replay v2: true carry-in events, unchanged availability

2026-11-08. Read-only CPU diagnostic tool. No production file, frozen core, policy, physical setting, or event-generator change.

## Why v1 stopped

The actual g1 v1 run is retained: exit 1 after 6.128 s. Ours c01 completed its three replay layers and V2E c01 passed archive/native whole-packet replay. V2E c01 then failed a strict nominal-interval check.

`V2E_BOUNDARY_MEASUREMENT.json` establishes a real source behavior, not just floating-point roundoff: a parent with nominal HDR endpoints 0.04–0.08 s contains 130 events stamped **0.0395000018 s**. Similar carry-in occurs later, around 0.5–0.696 ms before nominal interval starts. These native SloMo/emulator events belong to the new arrival packet even though their event time lies in the previous nominal renderer interval. We must preserve them and their later availability.

## v2 contract

The new script is `replay_packet_subwindows_v2.py`. The v1 script, tests, hash manifest and failed server result directory are unchanged.

1. **Event time is authoritative for window support.** Extend the partition into earlier 10 ms history windows when the packet actually contains carry-in events. For the example, t=0.0395 goes into a child ending at 0.04 s.
2. **Arrival parent remains authoritative for availability.** Every child still has parent 2's availability 0.09 s. Neither the event nor its estimate is exposed at 0.04 s. The previous controller outputs are not revised.
3. **Camera uses only available history.** A carry-in child ending at 0.04 s uses the already-known exact camera endpoint at 0.04 s. Other historical child times use Slerp / linear translation only between available camera endpoints. There is no target GT or hand-pose substitution. The detector still uses the child-end camera transform for a measurement whose median event time can be earlier; that geometry-time approximation is explicit and visible via `camera_pose_time_s` and `detected.t`.
4. **Repeated child endpoint is legal.** The old parent's last child and the new parent's carry-in child can both end at 0.04 s. They are different arrival groups with different events and availability. Unique identity is `(parent_packet_id, subwindow_index)`, not `end_s` alone.
5. **No event is dropped, duplicated, retimestamped, or moved to an earlier parent.** Each child records its original parent-relative input-index interval. Concatenated children must exactly equal the original event array in order and count.
6. **Global chronological checks replace the false nominal-start constraint.** Events within each packet must be ordered; new parent raw events must not precede the previous consumed parent event maximum. Equal timestamps are permitted and logged, without deduplicating distinct input events. A newly detected sample older than an already-processed filter/sample time causes an explicit stop. The diagnostic never rewinds or resets `filter_time` to absorb out-of-sequence input.
7. **No future support is accepted.** Events past the nominal parent endpoint by more than the existing 1e-8 numerical guard still stop for separate review. The v2 fix introduces no float32 special case and does not use a tolerance to absorb the measured 0.5 ms carry-in.

These checks are deliberately conservative. Genuine out-of-sequence arrivals would require a separately designed out-of-sequence estimator; this diagnostic stops instead of silently dropping them or changing published history.

## Separate historical onset from published state

The original `motion_onset_s` and `scoring_only_pre_release_false_onset` are preserved. They refer to an estimated historical event time.

Every mode now also reports:

- `scoring_only_first_control_moving_time_s`: first original control-query time at which replay returns an available, initialized moving state.
- `scoring_only_pre_release_moving_control`: whether that query actually precedes the true release (GT is used only for scoring this afterward).
- `scoring_only_first_control_moving_delay_from_release_s`.

Example: onset estimate 0.659 s, actual release 0.670 s, first moving control query 0.720 s means a slightly early estimated past onset, **not** a control state that declared movement before release. These are counterfactual estimator outputs at archived controller query times; no new physical control rollout is performed here.

## Replay layers and command

The same three-layer design remains:

1. Original archived public/static association + fit0, with exact archive parity required.
2. Current native whole-packet baseline, common fit1.
3. Current native subwindow candidate, same common fit1.

Compare layer 2 with layer 3. Layer 1 proves the raw-event reader, packet availability and archived baseline can be reproduced. The old c02 association repair is not confused with the new subwindow change.

Deploy `replay_packet_subwindows_v2.py` beside the unchanged `replay_static_association.py` in a **new** server run directory. Use the same sealed source, same old eight normal25 run paths, `--until 1.45 --flight-fit --archived-association public`. Replace only the executable script name from the original command. Do not overwrite `subwindows_cached8`; use a fresh v2 directory.

Candidate per-case JSON adds `carry_in_window_count`, `carry_in_event_count`, repeated endpoint count and `subwindow_history`. The per-window records include parent ID, child index, original input index range, true event/detection times, original availability, camera-support times/availability, prior parents with the same child endpoint, and equal measurement timestamp flags. A genuine order violation is saved as `*_TEMPORAL_SUPPORT_FAILURE.json` before the run stops, and prior completed modes remain in `PARTIAL_SUMMARY.json`.

## Verification actually completed locally

`test_replay_packet_subwindows_v2.py`: **11/11 PASS**, CPU only. Adds to the original six checks:

- True 0.5 ms carry-in with late availability and repeated child endpoint.
- Atomic equal timestamps across parents remain distinct input events and are logged.
- Raw events genuinely older than the previous arrival's consumed events stop without filter reset.
- A sample older than the already-processed filter time stops without rewinding.
- Retrospective early onset is graded separately from a genuinely early published moving query.

The existing native detector, RGB-D initialization and motion filter are exercised on synthetic static/moving event contours. No Genesis, torch, CUDA, HDF rendering or physics is loaded for these tests. **Actual eight-cache v2 parity and accuracy are still for root to run on g1; the synthetic PASS is not that result.**

Script SHA256: `76644b03188ef8cde09602de67cd142c0d2676fba2028ae5e6bd661ba5b9ec96`. Related source/test/evidence hashes are in `SUBWINDOW_V2_SHA256SUMS`.
