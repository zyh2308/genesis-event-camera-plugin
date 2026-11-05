# Event-model mechanisms for the 377-step bottleneck

Research checked: 2026-11-05/06. Primary papers and official source only. No SSH, GPU/numerical experiments, simulator/environment/bank/core changes.

## Recommendation

Research a separately versioned multirate variant: expensive optical/geometry reconstruction gets an adaptive clock; the existing 377-point sensor/RNG clock initially stays intact. Interpolate sensor inputs only within accepted optical segments, with visibility/unknown fallbacks. This changes intermediate radiance and output, so it is a new numerical variant, not a bitwise-equivalent engineering optimization.

Distinguish geometry/visibility evaluation, filtered/noisy sensor evolution, and event timestamp resolution. The literature separates these clocks. No inspected source proves a drop-in solver preserving this plugin's full current noise/occlusion contract.

## Verified repository pins

GitHub API queried 2026-11-05/06:
- ESIM master: 4cf0b8952e9f58f674c3098f1b027a4b6db53427 (2021-02-19).
- IEBCS main: 8fdab5c3d078435d091fa8a94ad657e1fedf0c59 (2025-04-28).
- DVS-Voltmeter main: 9d86353ecfaaa1af82b9eb66b162f0bf80cd4861 (2022-12-29).
- Event-adaptive path tracing publish: 19268b816e4d6bec92f08b6cc90887962a29aa30 (2026-10-31).
- v2e comparison pin from existing audit: cac638444557a224ba6ee5c29260916f152c00bd.

Line numbers below use pinned GitHub connector text. Browser raw views dropped blank lines and gave smaller line numbers.

## 1. ESIM, CoRL 2018

Paper (read): https://proceedings.mlr.press/v87/rebecq18a/rebecq18a.pdf
Project: https://rpg.ifi.uzh.ch/esim.html
Core: https://github.com/uzh-rpg/rpg_esim/blob/4cf0b8952e9f58f674c3098f1b027a4b6db53427/event_camera_simulator/esim/src/event_simulator.cpp

Section 3.3 schedules renders using predicted brightness change, dt=lambda_b*C/max|dL/dt|, or maximum image motion, dt=lambda_v/max||V||; experiments use lambda=0.5. The brightness predictor uses brightness constancy/local linearization and is not a universal guarantee. Section 3.1 reconstructs piecewise-linear log brightness and solves event crossings inside each optical interval.

Pinned imageCallback walks threshold levels (67-99) and directly interpolates crossing time (74). Refractory is checked at 77-91. The 2018 paper says refractory was absent; this must not be repeated about later code. The core's contrast-threshold perturbation is not our intensity-dependent IIR plus leak/shot process.

Transfer: optical sampling frequency need not equal event timestamp resolution.
Risk: ideal linear crossings do not preserve a filtered stochastic sensor.

## 2. ICNS / IEBCS, Frontiers Neuroscience 2021

Paper (read): https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2021.702765/full
Core: https://github.com/neuromorphicsystems/IEBCS/blob/8fdab5c3d078435d091fa8a94ad657e1fedf0c59/src/dvs_sensor.py
README: https://github.com/neuromorphicsystems/IEBCS/blob/8fdab5c3d078435d091fa8a94ad657e1fedf0c59/README.md

Section 2.3.1 starts with H(s)=1/[(1+s*tau_l)(1+s*tau_c)]. Since its two-exponential response is not generally analytically invertible, the authors deliberately approximate it with a delayed first-order response, allowing variable dt and efficient event timing. This is a model choice, not an algebraic identity for arbitrary two-pole dynamics.

Pinned get_latency_tau (281-297) computes -tau*log(1-amp), adds latency/jitter, clips to 0..10000 and casts to uint64 microseconds. update uses exponential interval propagation (342-346, 439-442), then repeats events/refractory resets for crossing pixels (355-437). Brightness-dependent tau is held per update (320-321). check_noise/check_noise_hist (186-258) schedule next noise but check at most one due event per polarity/pixel/update. Thus large-dt full-noise equivalence is not demonstrated. Current README describes a reduced Python reference; historical Blender/C++ needs historical revisions.

Transfer: analytical propagation plus event roots can avoid a global Euler clock.
Risk: input hold, tau, resets, noise/jitter and timestamp units differ. Integer-microsecond clipping is unsuitable for our strict gates.

## 3. DVS-Voltmeter, ECCV 2022

Paper (read): https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670571.pdf
Core: https://github.com/Lynn0306/DVS-Voltmeter/blob/9d86353ecfaaa1af82b9eb66b162f0bf80cd4861/src/simulator.py
Sampling: https://github.com/Lynn0306/DVS-Voltmeter/blob/9d86353ecfaaa1af82b9eb66b162f0bf80cd4861/src/simulator_utils.py

Equations 10-11 use Brownian motion with drift. Algorithm 1 samples polarity and first-passage time, then continues from events and preserves interval-end residuals. Nonzero drift uses inverse-Gaussian sampling; zero drift uses Levy sampling.

Pinned generate_events constructs drift/noise from frame difference and mean illumination (simulator.py 92-106). event_generation (utils 222-315) samples next time at 267, selects events before interval end at 273, and recursively continues only active pixels (294-306). Output timestamps are rounded at simulator.py 108.

Transfer: event-driven stochastic timing and active-pixel queues have a concrete implementation.
Risk: this fitted Brownian voltage law and residual rule are a different sensor/noise model, and still need adequate optical inputs. Rounding cannot make our strict FP64 comparison pass legitimately.

## 4. Manabe et al., TVCG 2026: time refinement and compaction

Author page: https://tatsylab.github.io/projects/manabe2026eapt/
DOI: https://doi.org/10.1109/TVCG.2026.3726999
Official pin: https://github.com/ichi-raven/event-adaptive-path-tracing/tree/19268b816e4d6bec92f08b6cc90887962a29aa30
Earlier author preprint (read separately): https://arxiv.org/html/2408.07996v1

The 2026 publisher article failed to open (HTTP 418). The following is code-level evidence, not a claimed full reading of that paper. The separate 2024 preprint addresses adaptive Monte Carlo path samples and hypothesis testing; do not confuse path samples with sensor-time integration.

Search core: https://github.com/ichi-raven/event-adaptive-path-tracing/blob/19268b816e4d6bec92f08b6cc90887962a29aa30/shaders/Slang/Integrators/MotionIntegrator.slang

detectEventEfficient (765-918) searches per-pixel time ranges with a stack, evaluates interval endpoints, caches endpoints (808-812, 875-877), and bisects candidates (905-908). A statistical test can reject a range (841-849). It stops at kEpsilon=1/1024 (19); output is quantized to 16-bit normalized time, with maximum 16 events/pixel/interval (14-18). Inference: endpoint-only rejection cannot by itself certify no hidden interior transient; our disocclusion/ON-OFF pulses require conservative bounds or fallback.

GPU dispatch: https://github.com/ichi-raven/event-adaptive-path-tracing/blob/19268b816e4d6bec92f08b6cc90887962a29aa30/shaders/Slang/Integrators/EfficientEventIntegrator.slang
Compaction: https://github.com/ichi-raven/event-adaptive-path-tracing/blob/19268b816e4d6bec92f08b6cc90887962a29aa30/shaders/Slang/Integrators/EfficientEventIntegratorCompute.slang

The ray-generation shader makes an event-candidate mask and early-sample cache, then traces compacted indices and updates each pixel's next search time/candidate flag. A prefix-sum compute pipeline builds the work list.

Transfer: endpoint cache, per-pixel refinement and compact GPU queues.
Risks: statistical rejection, finite epsilon, cap, packing and threshold model differ from our contract. Documented Vulkan hardware-ray-tracing/NV motion-blur requirements prevent a direct V100S renderer transplant. Do not import its timing/performance claims.

## 5. v2e, CVPRW 2021: comparison boundary

Paper: https://tub-rip.github.io/eventvision2021/papers/2021CVPRW_V2E_From_Video_Frames_to_Realistic_DVS_Events.pdf
Source: https://github.com/SensorsINI/v2e/blob/cac638444557a224ba6ee5c29260916f152c00bd/v2ecore/emulator_utils.py

The existing local audit established that automatic interpolation follows image motion, while low-pass processing uses a discrete explicit update, warns for a large per-step factor, and clamps it. Its small frame count does not demonstrate the same 300 Hz / eps=0.05 integration accuracy. It is evidence for architectural separation, not a numerical equivalence argument.

## Proposed evidence sequence; no implementation performed

1. Keep the qualified deterministic baseline and default-CUDA strict failures. Profile geometry evaluations, sensor updates, optical interpolations, active pixels and event counts separately.
2. Variant A: retain the full 377 sensor/RNG clock; use sparse optical anchors and interpolation only in accepted segments. Refine/fallback around motion, depth ordering, coverage, disocclusion, contrast reversals and uncertain pixels. Declare changed radiance as a new numerical variant.
3. Compare dense legacy output and independent dense optical/continuous references. Report strict failures honestly and separately measure timestamp/count/polarity/spatial errors, visibility/unknown behavior, complete noise statistics, runtime, peak memory, and real-loader inputs. Include subframe pulses whose endpoints agree.
4. Variant B only after A is understood: analytical sensor evolution and roots. Time-varying input, brightness-dependent tau, resets, leak/shot order and RNG coupling need explicit treatment. Changing noise clocks is another sensor/noise contract.
5. Spatial compaction needs conservative motion/depth envelopes plus fallback: forward splats/occlusion couple neighbors; a quiet event mask does not justify skipping geometry or noise.

These sources support separating clocks, not a guaranteed reduction to any particular sample count or promised speedup. Formal three-task training/real scoring, banks/labels/checkpoints/frozen core/shared environment remain outside scope. Generator acceleration does not directly accelerate optimizers reading existing offline event banks.
