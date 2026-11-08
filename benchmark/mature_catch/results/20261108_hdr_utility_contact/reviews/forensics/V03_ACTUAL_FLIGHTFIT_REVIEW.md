# v03 actual before/after flight-fit review

2026-11-08. Four real Genesis traces, read-only local analysis. No SSH, new simulation, GPU, or source changes.

## Finding

**v03 的改进确实经过了真实动作和接触，并非只让离线误差好看。** Ours 使用统一 flight-fit 后，接触前较早一段时间的位置与速度更准确，机器人实际手部位置随之改变，第一次接触时球与末端参考点的距离由 172.6 mm 降至 93.8 mm，最终持续真实接触到任务时限。原来的失败不能称为“已经掉球”：它在接触控制步中提前终止。

同一个改法也让 v03 的 V2E 从成功变成失败。它的速度估计更准，但接近的横向位置更偏，说明 **接取取决于整个接近过程和接触几何，不能只优化速度误差或看最后一次定位。** 全部结果需要保留。这一案例不支持“统一 fit 对所有来源都提升”，也不能据此宣称 Ours 总体优于 V2E；当前六例总体仍是 Ours 4/6、V2E 5/6（总体数字由主会话完整交付提供，此文件只重算 v03）。

## Recomputed from the four traces

All four runs first enter native `grasping` at the control-step endpoint 1.28 s and first record hand contact at 1.32 s. Here `grasping` is the original environment's approach stage, **not** the event estimator's assumed attachment.

For the precontact accuracy comparison, use the same six absolute query times **1.04, 1.08, 1.12, 1.16, 1.20, 1.24 s** in all four traces. Every corresponding action step ends by 1.28 s, before the first contact interval.

| Quantity | Ours baseline | Ours + fit | V2E baseline | V2E + fit |
|---|---:|---:|---:|---:|
| Outcome | failure | success | success | failure |
| Common precontact 3D position error median | 57.3 mm | **25.7 mm** | 32.3 mm | 42.5 mm |
| Common precontact 3D velocity error median | 0.347 m/s | **0.114 m/s** | 0.324 m/s | **0.134 m/s** |
| True hand-to-ball distance at 1.28 s, before contact | 191.7 mm | 177.8 mm | 135.6 mm | 155.5 mm |
| True hand-to-ball distance at first contact-step end 1.32 s | 172.6 mm | **93.8 mm** | 92.3 mm | 106.7 mm |
| Ball-minus-hand world X at 1.32 s | +3.6 mm | −1.1 mm | −31.6 mm | **−69.2 mm** |
| First assumed-attachment control time | none | 1.36 s | 1.36 s | none |
| Number of control intervals containing actual hand contact | 2 | 31 | 29 | 1 |
| Final simulation time | 1.356 s | 2.52 s | 2.52 s | 1.322 s |
| Final action-step elapsed time | **36 ms** | 40 ms | 40 ms | **2 ms** |
| Final terminated / truncated | true / false | false / true | false / true | true / false |
| Final interval had actual hand contact | true | true | true | false |

Distances use the actual `link6` reference position, not nearest collision-surface distance or proof of palm contact. World X is a world-coordinate component, not a camera-axis or full palm-alignment metric.

## The change happens before contact

- Both Ours and V2E actions first diverge between baseline and fit at **0.76 s**; actual hand positions first diverge at **0.80 s**, more than half a second before the first contact-step endpoint.
- Ours hand positions differ by **49.1 mm at 1.28 s**. Baseline/fit ball positions at corresponding endpoints remain equal to numerical precision through 1.28 s (maximum numerical difference around 1e-16 m in the listed approach window). Thus the approach difference exists before any divergent ball collision response.
- Ours at 1.24 s has a true ball-minus-hand vector of approximately **(+9, +162, +102) mm** at the subsequent 1.28 s endpoint; with fit it is **(−16, +126, +124) mm**. This is a changed approach geometry, not simply a smaller scalar image error. At 1.32 s the physical ball paths also differ because contact has occurred, so the postcontact ball difference must not be presented as a different launch condition.
- V2E hand positions differ by **43.4 mm at 1.28 s**. At 1.32 s its world-X hand/ball offset grows from −31.6 to −69.2 mm. The fit run's last event measurement at the 1.28 s query is about 201 ms old, versus about 81 ms in baseline, while its first contact still occurs in the same interval. This is an observable tracking/approach difference, not proof that age alone caused the failure.
- The Ours last precontact query at 1.28 s is actually slightly **less** accurate with fit in position (10.4 → 14.3 mm) and velocity (0.050 → 0.123 m/s), yet catches successfully. That directly illustrates why the previous focus on a last-query number was insufficient: the earlier history improved substantially.

## Native termination narrowed to collision-related predicates

The source excerpts in `NATIVE_TERMINATION_PREDICATES.json` show:

1. `step_touch` resets to false every policy step, then becomes true if any of that step's 20 physics ticks has actual hand/object contact. Therefore `hand_contact=true` does not imply the ball remains in hand at the end of that interval.
2. Base contact or ball contact with a non-hand object can set `terminated` inside the physics loop and immediately break it.
3. After the loop, a grasping-stage end-effector distance crossing the fixed threshold can also terminate the episode, but it does not shorten that loop.

The two failed v03 traces stop after **18/20 ticks (36 ms)** and **1/20 tick (2 ms)**. Under the supplied original loop, these are evidence for an **in-loop collision-related termination**, rather than a termination caused only by the post-loop distance rule.

The distance values provide an independent exclusion of the latter rule: the old Ours enters grasping at 191.745 mm, so the common native entry threshold must exceed this value. The final distances are 171.735 mm (old Ours) and 107.716 mm (fit V2E), both below that already-accepted entry distance. They do not satisfy the exit predicate `ee_distance >= same_threshold`.

**Exact collision identity is not recoverable from these traces.** Neither per-tick base-contact IDs nor object-contact IDs were recorded. We can distinguish collision-related termination from the post-loop distance-only hypothesis, but cannot distinguish base collision from ball/non-hand collision. Do not label the old Ours as “dropped the ball while holding” or claim which robot surface was struck.

## Interpretation and next action

This v03 intervention supports a useful causal chain at the run level: changing only the shared estimator changes the early state history, then policy actions and actual hand approach, then the contact response and outcome. It does not isolate whether position, velocity, uncertainty, or a particular action component was individually responsible. That would require an additional intervention; none is recommended during this already-fixed comparison.

For the current work, **retain the same fit configuration and inspect the remaining failed cases with this same early-history / contact-geometry framework**. Do not simultaneously change drag, attachment gates, policy or core. If exact termination cause is needed in later already-planned runs, log the native base/object contact IDs and first terminating physics tick as evaluation-only diagnostics. Do not use them as sensor input.

The before/after plot `V03_ACTUAL_FLIGHTFIT_COMPARISON.png` / `.pdf` shows all four runs, including the V2E regression. `V03_ACTUAL_TRACE_COMPARISON.json` and `V03_ACTUAL_TRACE_ROWS.csv` retain all computed rows; `analyze_v03_actual_trace.py` and `plot_v03_actual_trace.py` regenerate the analysis and figure. Source/output hashes are in `V03_ACTUAL_REVIEW_SHA256SUMS`.
