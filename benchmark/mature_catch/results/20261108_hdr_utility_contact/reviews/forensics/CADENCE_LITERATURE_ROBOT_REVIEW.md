# Event-ball tracking / catching: cadence and initialization evidence

Checked 2026-11-08. Primary papers, author project pages, and official author repositories only. Read-only web/local research; no SSH or production changes. AMiner skill instructions were consulted; its API requires an authenticated token, so this review uses directly verifiable primary sources instead.

## Answer for the research decision

**有成熟成功先例支持“用实际时间戳积累观测，再确认运动、拟合弹道、预测接取点”。没有找到“50 ms 必须改为 100 ms”这一具体数值的通用成功先例。**

最值得借鉴的不是某个毫秒数，而是三条原则：

1. 至少积累能识别运动的观测，并按它们的真实时间间隔估计速度。
2. 区分事件积累窗口、有效检测间隔、轨迹确认间隔、控制发布频率和信息可用时间。
3. 关联/丢失门槛应允许当前输入调度下正常相邻的观测，同时继续拒绝不一致运动。

因此，我们在 12.5 Hz 输入下检验一个共同的 cadence-aware 确认间隔有学术依据；**100 ms 是由本系统正常约 80 ms 到包间隔加有限余量推导的开发候选，不是论文照搬的标准值**。它的正确性仍须由双方共同使用后的真实数据和负例证明。

## 1. Closest successful example: Event-based Agile Object Catching with a Quadrupedal Robot

Forrai, Miki, Gehrig, Hutter, Scaramuzza, **ICRA 2023**.

- [Author paper, Sections III and IV-A](https://arxiv.org/html/2303.17479v1#S3) / [official laboratory PDF](https://rpg.ifi.uzh.ch/docs/ICRA23_Forrai.pdf).
- [Official code](https://github.com/uzh-rpg/event-based_object_catching_anymal).
- [Paper-linked real robot video](https://youtu.be/FpsVB8EO54M).

**Successful task:** real ANYmal catches fast balls; the paper reports 83% catching success and speeds up to 15 m/s. Its pipeline runs at up to 100 Hz. This is a real perception-to-action result, although the actuator is a quadruped with a net rather than our hand.

**Exact support:** Section III describes methods that “require two consecutive 3D position measurements”. Section IV-A uses known target size for monocular depth, known gravity, RANSAC over pairs of measured 3D points, then an inlier fit and a median filter on predicted impact points. It explicitly explains that faster balls yield fewer observations and less accurate trajectory fits. This directly supports timestamped multi-observation ballistic estimation; it does not prescribe a 100 ms confirmation gap. [Paper](https://arxiv.org/html/2303.17479v1#S4.SS1)

**Official code checked at commit `dd40bb8a362d3f327f396736d9bcff6dc1e16cb5`:**

| Source location | Verified behavior | Relevance |
|---|---|---|
| [parameters.yaml lines 2, 20–21](https://github.com/uzh-rpg/event-based_object_catching_anymal/blob/dd40bb8a362d3f327f396736d9bcff6dc1e16cb5/rpg_dynamic_obstacle_detection/parameters/parameters.yaml#L2) | Configured detection update frequency 200 Hz; event history window 35 ms and minimum new-event time advance 1 ms. | Window length differs from update frequency; it is not a 35 ms hard reset between consecutive target detections. These are public configuration values, not proof that every hardware trial ran at 200 Hz. |
| [dvs_handler.cpp lines 180–185](https://github.com/uzh-rpg/event-based_object_catching_anymal/blob/dd40bb8a362d3f327f396736d9bcff6dc1e16cb5/rpg_dynamic_obstacle_detection/src/dvs_handler.cpp#L180) | Converts the frame-window and minimum-advance parameters from ms to seconds. | Confirms units. |
| [parabola_ransac_node.py lines 55–69, 130–141](https://github.com/uzh-rpg/event-based_object_catching_anymal/blob/dd40bb8a362d3f327f396736d9bcff6dc1e16cb5/rpg_ransac_parabola/src/parabola_ransac_node.py#L130) | Keeps up to 50 accepted positions; does not compute a fitted output with fewer than two points; nominal ball-state publishing rate 100 Hz. | Supports a measurement-count condition and filtering a history instead of treating one event packet as reliable velocity. |

**Do not copy public code blindly.** This revision has a misleading unused old-window comment/value and an apparent timestamp conversion issue (`nsecs / 10e9`). We verified source text, not execution of this ROS repository. The paper's demonstrated method is the precedent; our already-tested causal implementation should retain correct timestamp handling. The YAML `check_time: 0.10` is not evidence for our motion-confirmation gap and is not used as such here.

## 2. Strong temporal-observation precedent: EV-Catcher

Wang, Cladera Ojeda et al., **IEEE RA-L 2022**, DOI `10.1109/LRA.2022.3188400`.

- [Author paper, Section V / Figure 5](https://arxiv.org/html/2304.07200v1#S5).
- [Official author project page with catching demonstration](https://ziyunclaudewang.github.io/ev-catcher/).

**Successful task:** an event-driven linear actuator catches fast ping-pong balls; the paper reports 81% over 120 shots, up to 13 m/s, on a Jetson NX. It learns impact location and time-to-collision; it is not an EKF implementation.

**Exact temporal support:** Figure 5 states: “We collect events every 10 ms, for a total of 120 ms.” Section IV-A describes a batch of 12 Binary Event History Images before inference. During that initial collection the actuator remains stationary, then performs a one-shot catching motion. A second option repeatedly predicts TTC and reserves motion time using a 210 ms TTC threshold. [Paper](https://arxiv.org/html/2304.07200v1#S4)

**What it supports:** waiting for a finite observation history can be part of a successful event-catching design. **What it does not support:** 120 ms here is cumulative observation duration, not the maximum allowable gap between two measurements, a loss-of-track timeout, or proof that two 80 ms-delayed packets suffice. I did not verify a public runnable author code release for this method; use its paper and author demo as evidence, not an unrelated GitHub implementation.

## 3. Direct support for asynchronous tracking gates: FAST-Dynamic-Vision

He, Li, Wu et al., **IROS 2021**.

- [Author paper, Sections IV-C1 and IV-C3](https://arxiv.org/html/2103.05903v1#S4.SS3).
- [Official repository with real throwing-ball trajectory demo and video links](https://github.com/ZJU-FAST-Lab/FAST-Dynamic-Vision).

This is a real event/depth tracking and drone-avoidance system, **not a robotic catching success-rate precedent**. Its relevance is the estimator: asynchronous event/depth observations, EKF through intermittent detections, and a gravity-constrained trajectory fit with independent measurement timestamps. Section IV-C1 says correspondence considers both elapsed time and position deviation; IV-C3 explicitly addresses unequal sensor sampling rates. [Paper](https://arxiv.org/html/2103.05903v1#S4.SS3)

**Official source at commit `a2a77f0daca4fbebde642d90e8a869875f60b5f0`:** [tracker.cpp](https://github.com/ZJU-FAST-Lab/FAST-Dynamic-Vision/blob/a2a77f0daca4fbebde642d90e8a869875f60b5f0/src/event-detector/src/tracker.cpp#L22) sets the association time threshold to **0.3 seconds**, then compares ROS timestamp differences against that value in `IsNewObj` (lines 247–265). It separately accumulates same-object observations before sending a trigger (lines 157–160). The nearby comments incorrectly say 30 ms / 10 ms; the executable value is 300 ms, so those comments must not be quoted as parameter evidence.

**What it supports:** explicit time-and-position correspondence gates and multi-observation confirmation are accepted practice. It does not make 300 ms appropriate for our moving camera, hand policy, ball size, or packet schedule; nor does continuous depth fusion establish our single-frame RGB-D setup by itself.

## Mapping these precedents to our actual 50 → 100 ms question

| Timing concept | Published examples | Our system / implication |
|---|---|---|
| Event sensor output | Real asynchronous events arrive throughout motion. | Our events originate from sparse renderer intervals and are exposed only after the causal endpoint packet is available. High-resolution event timestamps do not remove this packet delay. |
| Event integration window | ANYmal public config 35 ms; EV-Catcher a sequence of 10 ms updates over 120 ms. | Our 12 ms tail used by the contour detector is a separate choice from the input endpoint period. Increasing a confirmation gap does not change the detector's integration window. |
| Minimum evidence for motion | ANYmal two-shot minimum, then robust history fitting; EV-Catcher 12 BEHI sequence. | Keep two consistent motion observations. Enlarging the allowed interval must not silently turn this into one-observation initialization. |
| Normal independent observation spacing | Published hardware processes measurements at high frequency. | At 12.5 Hz the nominal endpoint spacing is 80 ms; a 50 ms hard confirmation limit can structurally prevent two adjacent packets from confirming motion, even with correct detections. Actual detection timestamps and arrival timestamps must be measured separately. |
| Timeouts / correspondence | FAST explicitly uses time and position; its own value is 300 ms. | Choose the smallest common interval that admits expected packet timing plus measured jitter, while retaining geometric consistency and causal availability. A fixed 100 ms candidate is defensible as our adaptation, not a universal literature value. |

### Recommended evidence before adopting the candidate

Use the same candidate for Ours and V2E and retain the baseline. In existing cached trajectories first check: adjacent valid detection timestamps really exceed 50 ms; both packets were available by the confirmation time; and the second detection is consistent with the first tentative trajectory. Score GT only afterward. Report onset delay, false motion before release, accepted/rejected detections, precontact position/velocity errors, and downstream catches for **all** predefined cases. At higher input rates where 50 ms already works, verify the change does not unnecessarily alter results.

Changing only this common detector/estimator timing contract is an interface adaptation. It is not evidence of an event-simulator improvement or a generator advantage. A new held-out set is required if a successful scenario will later become a formal comparison claim.

## Evidence archive

`CADENCE_LITERATURE_CODE_EVIDENCE.json` records retrieved official repository commits, blob SHAs, paths and selected line locations. No publication claim is based on a search-result snippet alone; the linked primary paper sections and official source content were opened. Remote video links are provided for viewing, not downloaded or represented as our experiments.
