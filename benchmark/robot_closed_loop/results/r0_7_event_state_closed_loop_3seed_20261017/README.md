# R0.7 learned event-state closed-loop development rollout

This is a three-episode development rollout for the fixed Direct-HDR reference domain. It is not a success-rate estimate, held-out test, generator comparison, or EVK4 calibration. The two strict successes show task contact/lift/hold feasibility for those episodes; the third identifies a high-speed failure mode.

## Provenance

- Server: g1, worktree `/home/zhaoyuhan/Eventbased_WAM/repositories/genesis-event-camera-plugin-r0_7-learned`.
- Run code commit: `e158dc06ee00ba44815d1f2a86898278f136e84d` on `exp/robot-closed-loop-v1`.
- Frozen simulator core: `e79e656ac00f7847b98db9e12bf43b093f3afff1`.
- Config SHA256: `aa97f6f5a12af92561ef053407bb795bba2801582958b4ad1134bcd24497435a`.
- Checkpoint SHA256: `49b91d4d89d320d2267a332e0dd8b15de692a7fed37fd8828a2e67c0f79cb4dc`.
- Run manifest SHA256: `8df981626cec229073d588921ba5562e3b021c5099c3b513e7abbe1ef345151e`.
- Original raw-output `SHA256SUMS` SHA256: `e52c827b0d89470db00dba9ecab3de4ced823f455dca9eec9e3c7a6b4c40d3aa`; all listed raw outputs verify with `sha256sum -c SHA256SUMS`.
- The original summary's `mode` string says `r0_6` due a label typo. The run manifest records the learned checkpoint and exact event-controller arguments; this was an R0.7 learned-event run. A correction note is included without altering the raw summary.

## Episode outcomes

| Episode | Speed / launch y | Seed | First / bilateral contact (s) | Close / commit (s) | Contact x offset / y mismatch (mm) | Relative contact speed (m/s) | Lift / strict hold | Result |
|---|---|---:|---|---|---:|---:|---|---|
| 0 | 0.30 m/s / 0 mm | 2026103800 | 0.904 / 0.907 | 0.890 / none | +19.9 / -1.1 | 0.183 | 1.1095 s / 0.5 s | strict success |
| 1 | 0.40 m/s / +55 mm | 2026103801 | 0.676 / 0.676 | 0.660 / none | +5.1 / +0.1 | 0.299 | 0.7435 s / 0.5 s | strict success |
| 2 | 0.50 m/s / -55 mm | 2026103802 | 0.4875 / 0.4875 | 0.470 / 0.480 | +25.9 / +1.1 | 0.365 | none / 0 s | `unstable_grasp` |

Strict success was 2/3: 1/1 at 0.30, 1/1 at 0.40, and 0/1 at 0.50 m/s. Do not infer population reliability from these three episodes. The failed episode had 22 bilateral-contact samples but no lift, no qualified hold samples, and 0.9438 m maximum relative slip; the minimum true bottom clearance was `-0.310 mm`.

## Causal state and arrival metrics

[`rescore/closed_loop_rescore.json`](rescore/closed_loop_rescore.json) and [`rescore/controller_poll_metrics.csv`](rescore/controller_poll_metrics.csv) are a deterministic evaluator-only rescore of the saved controller polls. State error is aligned at the event state's causal timestamp. The arrival and intercept comparison sends both that event estimate and aligned GT at the same cutoff through the same fixed deceleration predictor using the logged input age; it does not query future GT arrival.

| Phase | Polls | Fresh fraction | Input age P95 | XY position error P95 | XY velocity error P95 | Arrival error P95 | Intercept Y error P95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Before first contact | 193 | 100% | 10 ms | 61.8 mm | 0.213 m/s | 324 ms | 18.9 mm |
| Contact to lift, or episode end if no lift | 278 | 100% | 10 ms | 910.0 mm | 0.772 m/s | 705 ms* | 85.8 mm* |
| After lift, successful episodes only | 135 | 100% | 10 ms | 28.9 mm | 0.051 m/s | 406 ms* | 23.6 mm* |

`*` Arrival/intercept errors exist only for polls where both states produced a reachable prediction (5 and 5 polls in these phases); they are too sparse for a stable percentile. The contact-to-end aggregate combines two short successful contact-to-lift windows with the failed 0.50 m/s episode; see per-episode phase metrics in the JSON. Successful episodes' contact-to-lift XY P95 was 37.9 mm. The failed episode's contact-to-end XY P95 was 913 mm although its status remained fresh at all 251 polls. Freshness currently measures event support and age, not localization confidence.

Immediately before closure, the last fresh estimates were:

- Episode 0, poll `0.8805 s`: XY error 3.6 mm; event/GT same-cutoff arrival `66.9/120.3 ms`; predicted intercept-y difference 2.2 mm.
- Episode 1, poll `0.6505 s`: XY error 5.0 mm; event/GT same-cutoff arrival `62.7/41.4 ms`; predicted intercept-y difference 3.6 mm.
- Episode 2, poll `0.4605 s`: XY error 5.4 mm, velocity error 0.182 m/s; event/GT same-cutoff arrival `134.4/85.1 ms`. At `0.4705 s`, the event estimate said 58 ms versus 75 ms for aligned GT. The object first contacted 25.9 mm beyond the intercept plane and only briefly maintained bilateral contact. This points to high-speed velocity/arrival and closure timing as the failure area; it does not by itself establish a need to change the shared lift gate.

## Cost and source boundary

- Task rollout wall time: `198.515 / 205.208 / 337.892 s`; mean `247.205 s`.
- Episode-loop artifact generation: `776.405 s` total.
- Full Python process: `836.69 s` wall, `823.90 s` user, `6.07 s` system, maximum RSS `1,836,820 KB`; exit 0.
- Per-episode Direct-HDR dynamic visibility minimum: `1.000 / 0.986 / 1.000`.
- Each physics step rendered HDR, ran the shared `DvsEmulator`, and retained 1/2 kHz raw event streams. The response was ideal threshold 0.2, no noise, infinite bandwidth. Ours temporal reconstruction and V2E interpolation were bypassed. This is a synthetic shared pixel response, not an EVK4-calibrated sensor result.
- Runtime log SHA256 `d366440b6942f0469946ddd7227bfc58d61a8ca83d2e0caf717afe09436b7cfb`; `/usr/bin/time -v` runtime SHA256 `b3ba75c68a52c88e960fbe537c5067aa20b42b4c180efe23107ae29f883999e7`; exit marker SHA256 `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`.

The per-episode folders contain the strict episode CSV, full 0.5 ms state/control trace, raw events, HDR snapshots, and rollout GIF. Episode 0 and 1 include bilateral-contact, lift-start, and strict-hold images; episode 2 includes the failed contact and video.

## Reproduction

The recorded g1 launcher is [`../runtime_logs/r0_7_event_state_closed_loop_3seed_20261017.launch.sh`](../runtime_logs/r0_7_event_state_closed_loop_3seed_20261017.launch.sh). The training and closed-loop code commits differ intentionally: the checkpoint was trained at commit `64d5de9`; the event-controller rollout used `e158dc0`. Reuse the exact configuration and checkpoint hashes above when comparing downstream states.
