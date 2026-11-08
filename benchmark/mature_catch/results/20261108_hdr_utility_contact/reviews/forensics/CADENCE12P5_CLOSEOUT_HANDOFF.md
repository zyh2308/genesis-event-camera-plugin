# Final 12.5 Hz batch: minimal closeout plus independent timing supplement

2026-11-08. Local preparation only. No SSH, W production changes, thresholds, policy, core or already-deployed postprocessor changes.

## Deploy the main closeout now

`postprocess_contact_cadence12p5.py`

SHA256: `2ffab436fe0df543e9d44950a35d3ba9f5ce815d18fb6916030e1df15388d6e8`

This is the validated pair postprocessor with only a new description and three cadence changes: require one common case-protocol endpoint rate of12.5 Hz, verify each actor's `event_endpoint_rate_hz`, and put that actual common rate into `comparison.fixed_parameters.endpoint_hz`. The same `audit_contact_observation_windows.py` is unchanged. All raw/score/window checks and retention remain. No timing reconstruction is required to finish this batch.

Read-only inspection confirmed existing `analyze_cadence.py` and `audit_cadence_final.py` accept this cadence: source intervals are checked using1/case endpoint rate, and V2E minimum upsampling is checked as1000//12.5=80. Their success criterion is unchanged. There is no25 Hz restriction in those generic checks.

## Optional timing supplement, after actors are available

`extract_contact_cadence_timing.py`

SHA256: `c549786cf0e93323cc207b0759aa65ba2ebd6005b0097f41e47de2be5e37a9e3`

Run `python extract_contact_cadence_timing.py --root <actual12p5root>`. It writes `TIMING_ACQUISITION_SUMMARY.json` with its own `.SHA256` sidecar. It does not modify primary raw arrays, scores, existing manifests or closeout. If incorporated into the final full-file inventory later, refresh that inventory explicitly; the supplement's sidecar already hashes the new JSON.

First-priority fields come directly from saved actual trace/window artifacts:

- Whether a moving state was ever published and adopted by the policy.
- Every parent's endpoint, actual availability/processing query, raw event count, total/nonempty/detected child-window counts and pending status.
- `first_control_published_moving_state_s` and `first_control_policy_adopted_moving_state_s`. The latter additionally requires a policy object input and no held gate. Neither is the historical event onset timestamp.

Supplemental confirmation details reconstruct only the sealed RGBD-base static-to-motion rule from saved public detections and the saved single-frame initialization/calibration. This uses no target GT, release ground truth, new events, association, full trajectory fitting, physics, controller rollout or renderer. It stops after the first confirmation and records the actual sample times, child/parent IDs, original availability and adjacent gaps. If reconstruction or mapping cannot be established, details are `UNKNOWN` with a reason; basic actual query and packet statistics remain available. Missing/unfinished actors are marked unavailable in this supplemental file rather than holding the primary results open.

The report can establish that multiple detections within one original80 ms arrival packet met the existing50 ms adjacent-confirmation rule. It must still report the enclosing packet's actual arrival and first control adoption. It does not establish earlier delivery, independent camera renders or that an onset at, for example,65 ms caused a control update at65 ms.

## Checks completed

- Both scripts compile.
- Two lightweight CPU supplement tests pass: unknown confirmation detail preserves actual published/adopted query times; known same-parent10 ms confirmation spacing preserves the late90 ms arrival and120 ms first query rather than confusing them with a65 ms event onset.
- The old deployed pair and four-method-confirmation scripts and the strict window auditor were not edited.

These files prepare final closeout. They contain no new12.5 Hz physical outcomes or group-advantage claim.
