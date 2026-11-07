# Camera7 held-out Catch It generalization

**Goal:** Evaluate the locked 7-degree camera and unchanged event estimator, pretrained policy and frozen sparse simulator on 18 new episodes, retaining all outcomes and demos.

**Architecture:** Reuse immutable runtime source 397d4549d685 and shared environments. Two atomic GPU leases execute disjoint seed blocks on g3, with 18 matched noisy-GT policy reference episodes. Only orchestration and reporting are new. No training or simulator modification.

**Tech Stack:** Genesis 1.2.2 physics/rendering, existing CUDA sparse-event worker, original Catch It PPO weights, Python orchestration and FFmpeg.

1. Commit the fixed protocol and source/checkpoint/core hashes before launching. Six consecutive seeds 2026110801--06; original native launch sampler; fixed asset seed1702.
2. Run normal and high-speed episodes plus backlit counterparts matched to each high-speed case. Speeds refer to horizontal launch magnitude; record complete 3D velocities. No seed replacement or tuning after outcomes.
3. Run original noisy-GT observation policy references with the same seed and condition. Verify launch/gravity/release parity, immutable hashes, causal event availability and physical contacts. A failing reference does not prove task infeasibility.
4. Summarize every episode, per-condition success, contact and pre-contact perception error; preserve initialization failures and early terminations. Shared seeds create dependency across conditions.
5. Deliver all native videos and labeled overview/representative demos, complete CSV, hashes, reproduction commands and an evidence-based camera assessment.

The detailed protocol is benchmark/mature_catch/GENERALIZATION_CAMERA7_PROTOCOL.json. Camera7 was selected on development seed1702. This new set is a development-held-out generalization screen, not a generator ranking or real-hardware event calibration. Any later adaptation makes these cases development data.
