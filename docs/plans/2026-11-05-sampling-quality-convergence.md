# Sampling-quality convergence implementation plan

**Goal:** Obtain measured error–sampling–runtime curves, and a bounded recommendation for the minimum sufficient temporal sampling under the registered Stage9 physical profile. Do not treat377 as ground truth or claim a universal highest-quality N.

**Architecture:** An isolated read-only consumer of the qualified frozen GPU package and existing synthetic physical banks. Begin with independent analytic sensor signals; validate short geometry/occlusion cases against independent rendered subframes; examine full-profile noise statistically. Use actual production sensor arithmetic in the tested implementation. Preserve old releases, all existing training/data, and real-test isolation.

**Tech stack:** Existing WSL repository/worktree and g1 research Python/Torch environment. g1 GPU2 only after fresh process/lease checks; two CPU threads. No g3, package/environment changes or training restarts.

## 1. Isolate and register

- Base shared branch: exp/public-sim2real-v1 at7dd19f5ad30e426501e2db21d4d2ee4f7bdd944f.
- New worktree: /home/科研/Eventbased_WAM/repositories/gec-sampling-quality-20261105.
- New run: /home/zhaoyuhan/Eventbased_WAM/runs/simulator_sampling_quality_20261105f.
- Existing qualified package: /home/zhaoyuhan/Eventbased_WAM/runs/simulator_sampler_optimization_20261105e/candidate_release; verify its source inventory and use a read-only snapshot.
- Atomically reserve only confirmed-idle g1 GPU UUID2. Retain source/config hashes, timing, job PID, startup ticks, logs and exit status. Close own tasks and release the lease at completion.

## 2. Lock the initial measurement contract

- Registered physical parameters remain300Hz cutoff,0.2 nominal ON/OFF threshold,0.5ms refractory; fixed initial spatial maps. Store all resolved parameters.
- Nominal N values per10ms:32,63,126,189,252,377,754,1508. Independent reference initially3016-equivalent or adaptive high-accuracy integration; recheck tighter reference. These are planned grids, not assumed good settings.
- Use common absolute evaluation times and identical signal origin, initial state, radiometry and phase offsets. No per-stream temporal/minmax normalization to hide errors; no timestamp quantization.
- Report log-state RMSE/C andp99/C; event one-to-one same-pixel/polarity matching precision/recall/F1 within50us; matched timestamp p50/p95/p99 and complete-reference success fractions; missed/extra events and count bias. Zero-event cases separately report false events.
- Pilot engineering thresholds (not a field standard): state RMSE/C<=0.025, p99/C<=0.05, event F1>=0.99 at50us, timestamp p95<=20us, count error<=1%. All protected strata must pass. Show failures and full curves; do not change these gates after measurement. Near-threshold adversarial cases are separately labeled sensitivity diagnostics rather than silently averaged away.
- A reference must change by less than20% of each relevant absolute budget under further refinement. If it does not, label reference unresolved and do not rank N from it.
- Compare against377 as a compatibility baseline in addition to independent-reference errors. Passing budget is not bitwise equivalence or a universal realism claim.

## 3. Analytic sensor experiment first

- Independent analytic step/constant-input RC solution with threshold roots, plus sinusoidal/pulse/near-threshold signals using independent tight ODE integration as needed. Temporal noise off only in clearly named diagnostic cases; fixed threshold maps and documented refractory semantics.
- Shift input edges relative to sample boundaries. Include bright/dim cases, ON/OFF, empty/no-event controls, reversals and short pulses.
- Before scoring, verify oracle on tractable single-crossing cases and verify actual sensor public-event/reset/refractory semantics. Do not invent a reference rule different from the registered implementation.
- Record actual N, dt maximum, epsilon/clamp warnings, source path, device, precision and fallback. High-N tests must avoid512/300 caps and profile-based silent slow fallback.

## 4. Geometry and occlusion experiment

- Reuse current stage9 scene/trajectory builders, not a new unrelated toy training bank. Generate a small independent dense physical-reference clip under a unique run directory, with original frozen renderer/core unmodified.
- Candidate sees fixed low-rate endpoints; independent rendered subframes are evaluation only. Cover camera/object motion, texture, occlusion/disocclusion and thin geometry; report reference-defined active/boundary/background strata separately.
- A densified endpoint interpolation alone is only a numerical reference, never true middle-frame rendering. Report this distinction explicitly if any phase is blocked.
- Narrow candidate sweep after analytic results; retain377 and at least one higher-resolution reference. Use short clips and continuous sensor state across memory-safe chunks.

## 5. Full-noise statistics and consumer sanity

- Full registered noise formula/parameters and same spatial maps. Multiple independent temporal replicates with recorded total pixel-time; compare ON/OFF rates, inter-event intervals, fixed-window count variance and refractory violations.
- Do not treat identical integer seeds as coupled stochastic paths under different N. Do not treat infinite-N output of the step-dependent noise implementation as physical noise truth. Report confidence intervals and insufficient-count limitations; no nonsignificance-as-equivalence claim.
- Frozen downstream checkpoints may be used only for synthetic sensitivity confirmation after direct-signal tests. No retraining or real-data scores for N selection.

## 6. Deliver

- Save raw events/states where useful, detailed metric JSON/CSV, standalone scientific convergence plots and a Chinese report with passing/failing conditions, error floors and sampling recommendation scope.
- Publish code/lightweight reports to the shared research branch without touching unknown files. Index large artifacts by exact server path/size/SHA256.
- Preserve all failed attempts. A partial analytic result is labeled partial until geometry/noise checks complete; do not manufacture a final N.
