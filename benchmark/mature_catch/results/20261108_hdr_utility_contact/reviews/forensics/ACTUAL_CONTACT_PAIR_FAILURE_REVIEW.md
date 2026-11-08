# Actual contact-paired 25 Hz failure review

2026-11-08. Local read-only analysis; no SSH, rendering, physics, training or production edits.

## Main finding

这批结果确实是 **Ours 2/6、V2E 5/6、只初始化 1/6**。Ours 四个失败都碰到了球，但没有完成留球。现有证据不支持把主要问题说成“事件断流”“球一直没有动起来”或“接触前过早宣布抓住”。

更值得优先检查的是：**Ours 在接球前一段时间持续存在约 5–7 cm 的三维位置偏差，最后一刻虽然追准了，但机械手可能已经按此前的错误位置运动。** 这能解释为什么末帧的像素误差、速度误差很好，最后仍然失败。这是时序上的相关证据；目前的 portable 文件没有完整动作与手球相对姿态，不能据此直接断言原因。

## Data checked

- `delivery_local/contact_paired25/PAIRED_PORTABLE.json`: complete=true, violations=[], 18 episodes retained. The same six launch configurations are identical across Ours/V2E/init-only. All records share the policy checkpoint and runner hash.
- `delivery_local/contact_paired25/CONTACT_PAIRED_COMPARISON.json`: per-episode physical outcomes agree with portable records.
- `delivery_local/contact_paired25/FINAL_AUDIT.json`: archived audit reports PASS for 18 episodes, packet contract and retention of all successes/failures. This review does not independently rehash remote raw arrays.
- No event-flight-fit is enabled in this 18-episode batch. The new shared contact settings are benchmark settings; the frozen event core and policy remain outside this analysis.

## All six matched cases

“接触前窗口”取首次接触控制步前的最后六次查询，约 200 ms；只用在该控制动作之前已经存在的状态。单位为 mm / m/s。下面的中位数不是整集平均，也不含接触后的估计。

| Case / seed suffix | Ours / V2E outcome | Actual contact control steps O/V | Precontact-window position median O/V | Precontact-window velocity median O/V | Last precontact position O/V | Last precontact velocity O/V |
|---|---|---:|---:|---:|---:|---:|
| v01 / 31 | success / success | 29 / 29 | 59.3 / 42.0 | 0.300 / 0.334 | 54.7 / 67.8 | 0.191 / 0.339 |
| v02 / 32 | failure / success | 2 / 27 | 60.6 / 34.2 | 0.256 / 0.261 | 53.3 / 50.7 | 0.139 / 0.274 |
| v03 / 33 | failure / success | 2 / 29 | 53.6 / 52.1 | 0.289 / 0.324 | 10.4 / 113.7 | 0.050 / 0.359 |
| v04 / 34 | failure / failure | 1 / 3 | 54.6 / 37.0 | 0.212 / 0.242 | 43.8 / 56.1 | 0.098 / 0.263 |
| v05 / 35 | success / success | 29 / 29 | 57.4 / 47.1 | 0.288 / 0.302 | 41.4 / 108.9 | 0.137 / 0.358 |
| v06 / 36 | failure / success | 2 / 30 | 69.9 / 36.1 | 0.328 / 0.326 | 50.4 / 50.4 | 0.159 / 0.320 |

The last precontact query is `first_touch - 0.04 s`: it is the input to the first action interval that later registers contact. The contact timestamps are control-step endpoints, not exact collision instants. For v05 the Ours/V2E contact times differ by 40 ms, so their relative-to-contact windows are not identical absolute times.

## What each Ours failure shows

| Case | Observable failure chain | What can be concluded now |
|---|---|---|
| v02 | 1.24–1.44 s position errors: 55, 60, 61, 66, 64, 53 mm. V2E: 23, 33, 23, 35, 42, 51 mm. Both first touch at 1.48 s; Ours only two contact steps and ends at 1.60 s. Both enter assumed-grasp at 1.56 s. | Sustained earlier 3D bias is a stronger lead than the last estimate. Entering the grasp state did not ensure real retention. No evidence of a precontact false grasp trigger. |
| v03 | Ours error falls from 61/60/74 mm to 47/39/10 mm before contact; final velocity error is only 0.050 m/s. First touch at 1.32 s; failure terminates at 1.356 s while the final recorded step still has contact. No assumed-grasp is reached. V2E survives despite 114 mm last position error. | This is the most important case for action/termination inspection. A good final estimate does not guarantee the hand approaches correctly. The portable file cannot say which native termination condition fired. Do not label this simply as “ball dropped” based on contact count. |
| v04 | Earlier Ours position is 51–61 mm, then 44 mm; V2E is 27–46 mm, then 56 mm. Both fail; Ours contacts once and ends 80 ms after first-touch endpoint. | This is a shared hard case, not evidence for an Ours-only startup or freshness defect. Need hand pose and contact geometry to distinguish a glancing hit from another termination. |
| v06 | Ours position errors: 73, 74, 70, 69, 63, 50 mm. V2E: 28, 43, 33, 31, 40, 50 mm. Both first touch at 1.36 s and assume grasp at 1.44 s; Ours ends at 1.48 s with only two contact steps. | Similar last errors and grasp-state timing, different previous trajectory and outcome. Inspect earlier hand motion and the first contact-to-attachment interval. |

## Startup, freshness and assumed grasp

- All 12 event episodes have an available state at every reported flight query (18 or 19 queries each). Ours and V2E motion onset differ by at most about 1.2 ms within each pair; all start after a comparable 61–89 ms release-to-onset delay. The previous “never leaves static initialization” failure is absent here.
- The four failed Ours episodes use a last precontact event measurement around **40 ms old**. Their matched V2E measurements are about **81–161 ms old**. Ours therefore does not have an obvious larger state-age disadvantage.
- `fresh_flight_polls=0` comes from the strict freshness cutoff at this 25 Hz schedule (query ages around or slightly above 40 ms). It must not be interpreted as “there are no usable observations”.
- No event episode enters assumed-grasp before the first actual contact interval. Ours v02/v06 enter it 80 ms after the contact-step endpoint; v03/v04 never enter it. There is no current evidence that a precontact false-positive hold is the cause.
- A postcontact false attachment or an attachment that fails to follow the true ball is still possible. The portable file stops perception scoring at contact and cannot test that. V2E v05 succeeds without entering assumed-grasp, so that state is not a universal requirement for success.
- Last precontact pixel errors in the four failed Ours episodes are only **7.0 / 4.4 / 5.8 / 7.2 px**, all lower than their V2E counterpart. The gap between good image location and centimetre-scale 3D error makes distance/trajectory estimation a reasonable target. A ray/range decomposition needs camera poses and vector states; pixel errors alone do not prove the error is entirely depth.

## Existing causal flight-fit cache evidence

These are **eight cached old development trajectories**, not the new six physical seeds and not new catches. The fit uses the public RGB-D initialization, observed event bearings/times and gravity; GT is used only for scoring in the replay. Root is already running the same fit on both generators in a fixed new 12-episode physical comparison.

| Cached case / method | Position median native → fit (mm) | Velocity median native → fit (m/s) |
|---|---:|---:|
| c01 Ours | 42.7 → 29.2 | 0.178 → 0.123 |
| c01 V2E | 30.4 → 31.9 | 0.165 → 0.138 |
| c02 Ours | 56.9 → 33.8 | 0.330 → 0.127 |
| c02 V2E | 40.2 → 30.3 | 0.278 → 0.132 |
| c03 Ours | 36.7 → 25.8 | 0.153 → 0.114 |
| c03 V2E | 51.1 → 42.5 | 0.350 → 0.106 |
| c04 Ours | 63.2 → 37.5 | 0.293 → 0.141 |
| c04 V2E | 46.1 → 39.7 | 0.248 → 0.102 |

Velocity median improves in 8/8; position median improves in 7/8. Retain the small V2E c01 position regression (+1.5 mm). Absolute-range median improves in only 6/8, including a V2E c03 worsening despite its total 3D error improvement. Do not claim all aspects improve or extrapolate these cache metrics to closed-loop success.

## Next step, with one change only

**Continue the already-started shared fit1 comparison; do not simultaneously adjust drag, hold gates, friction, policy, camera or core.** The meaningful question is whether removing the sustained precontact position/velocity bias changes the hand's approach and real retention on these same six cases. Keep the failed fit episodes as well.

The minimal next evidence is a paired time plot for v02/v03/v06: estimated-vs-GT 3D position/velocity, actual hand-to-ball vector, action and contact over the last 0.24 s before first touch through the next 0.20 s. In plain terms: **看清球的位置是否真的更早变准，手是否因此到对地方，以及碰上之后为什么留不住。**

If fit improves free-flight error but not catching, stop claiming that another localization improvement alone fixes utility. Inspect the approach/contact state and the original termination condition, using the following existing files rather than running a multi-parameter sweep.

## Exact existing files needed from root

For v02/v03/v04/v06, both `ours` and `v2e`, under the `output` paths stored in the portable records:

1. `episode_000/trace.json` (or the actual per-episode folder used by the runner). Required fields already emitted by `run_original.py`: `t_before`, `t`, `terminated`, `truncated`, `stage`, `held_before_observation`, `hand_contact`, `ee_distance`, `action`, `qpos`, `qvel`, `object_position`, `hand_position`, `event_state`, `policy_object_input`, `scoring_only_GT_position_at_control`, `predicted_event_uv`, and pixel score.
2. `migration_audit/sensor_episode_000` state/detection history, grasp events and packet audit. Needed: measurement/available timestamps, estimated position/velocity, state source, accepted/rejected candidates, hand transform and attachment decision. Fetch the actual filenames listed in that directory; do not invent missing logs.
3. Camera packet transforms/intrinsics from existing HDF5 or packet metadata, to decompose 3D error into range versus transverse direction at the same causal timestamp.
4. The exact original environment termination expression/source hash loaded by the run. The present trace logs the boolean; if the reason is not logged, evaluate each existing native predicate offline where its inputs were recorded. For v03, also retrieve the final physics/contact audit if it already exists because termination occurs inside a 40 ms control interval.

GT and contact are evaluation labels throughout; they must not be fed into the event estimator or the frozen policy.

## Reproducibility

Generated by `forensics/analyze_actual_contact_pair.py` from the above local delivery JSON. Machine-readable outputs are `ACTUAL_CONTACT_PAIR_FAILURES.json`, `ACTUAL_CONTACT_PAIR_FAILURES.csv` and `FLIGHTFIT_CACHED8_MEDIANS.json`. Full source/output hashes are in `ACTUAL_CONTACT_PAIR_REVIEW_SHA256SUMS`.
