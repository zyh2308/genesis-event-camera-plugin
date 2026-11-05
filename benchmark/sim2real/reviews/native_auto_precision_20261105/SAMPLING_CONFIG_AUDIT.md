# Native automatic sampling audit — 2026-11-05

Read-only code audit; no server connection, shared repository changes, model/profile changes, or training performed by this agent.

## EventGenesis: 377 is a native automatic result

The archived `stage9_sensor_profile.json` explicitly sets only `sampler_kwargs.max_samples=512`; other sampling settings come from the frozen dataclass. `noise_model.py:109` retains `sensor_eps_target=0.05`. `physics_plugin.py:164–169` automatically derives `tau=1/(2*pi*cutoff_hz)` and supplies it to the sampler when the user did not supply a separate time constant. With cutoff 300 Hz, tau is approximately 0.530516 ms.

Both `physics_interpolator.py:355` and `gpu_interpolator.py:222` call the same frozen `PhysicsAdaptiveSampler.compute`. Its final request is the maximum of:

- minimum samples, default 2;
- contrast: ceil(max estimated log-brightness change / (0.5 * 0.2));
- visibility boundary displacement: ceil(max boundary motion / 1 px);
- source-material-point relative depth: ceil(max relative change / 0.1);
- camera displacement and independent articulation displacement: ceil(max motion / 1 px), each separately;
- radiance residual: ceil(p99 residual / 0.1);
- sensor integration: ceil((dt/tau)/0.05).

Segmentation/coverage and unknown support feed visibility/disocclusion diagnostics. A nonzero unsupported fraction also requests at least min_samples+1 and recommends rendering; it does not magically reconstruct unseen surfaces. The final count is capped at 512 by the existing profile, with `capped`, `uncapped_samples`, and `recommend_render` preserved.

For dt=10 ms, the sensor bound is ceil(0.010 * 2*pi*300 / 0.05)=377. Thus even a low-motion interval can natively require 377 steps. Other bounds can dominate on a sufficiently difficult interval. There is no fixed-377 override in the benchmark's EventGenesis call: `compare.py` constructs the frozen sampler and records the returned `last_sampling.samples`.

The old controlled comparison did impose 377 on V2E (`new_slomo(auto_upsample=False, upsampling_factor=377)`). That is different from the native automatic mechanism above.

## V2E: native auto and its independent lowpass warning

Official `slomo.py:319–351` takes the ceiling of maximum forward/backward flow magnitude across the batch, honors an optional minimum factor, and enforces factor>=2. Do not replace it with a custom EventGenesis sampling controller.

Primary source: https://raw.githubusercontent.com/SensorsINI/v2e/master/v2ecore/slomo.py

The official lowpass code independently uses:

```
inten01 = (DN + 20) / 275
tau = 1 / (2*pi*cutoff_hz)
eps = inten01 * delta_time / tau
warn if max(eps) > 0.3
clamp eps at 1
```

Primary source: https://raw.githubusercontent.com/SensorsINI/v2e/master/v2ecore/emulator_utils.py

At 300 Hz and a 5 ms intermediate interval, nominal white-input eps is about 9.425. Native auto chooses from optical motion, not this numerical guard. The actual trial should record intensity/eps or preserve warnings; absence of an exception does not imply accurate lowpass integration. Do not infer full event quality solely from this theoretical bound.

The published README describes tau at least three times the input interval. The actual source's warning limit 0.3 is slightly stricter. For a supplemental guarded-auto point in the 8-bit DN range, a conservative minimum factor at 100 Hz is ceil(0.010/(0.3*tau))=63. This point remains V2E native auto with a public minimum-factor parameter, but is **not** the unmodified native-auto point and is **not** proven quality-matched to EventGenesis.

Primary documentation: https://github.com/SensorsINI/v2e#automatic-vs-manual-dvs-timestamp-resolution and https://github.com/SensorsINI/v2e#photoreceptor-lowpass-filtering

These online sources were inspected on master; the root executor should additionally verify the archived pinned source before execution. The pinned version is `cac638444557a224ba6ee5c29260916f152c00bd`; public web retrieval of that precise ref was unavailable in this audit.

## Recommended executable comparison

Do not change the frozen profile, Ours sensor_eps_target, temporal integration, precision, noise, unknown/occlusion policy, or physical trajectory. Keep both methods' cutoff=300 Hz and the archived threshold/noise/refractory/seed settings, retaining native sensor equations.

Report separate operating points:

1. **Both native auto, primary new measurement:** Ours constructs the existing optimized plugin and frozen sampler without overriding U. V2E uses official SuperSloMo `auto_upsample=True, upsampling_factor=2`, with the same existing full sensor profile. Label this accurately as native temporal auto under the registered sensor profile, not completely default V2E parameters.
2. **V2E native auto with official numerical safeguard, supplemental:** `auto_upsample=True, upsampling_factor=63`, subject to source verification. Keep Ours unchanged. This avoids blindly imposing Ours's 0.05 target; measure quality/convergence before making a quality-matched statement.
3. Retain the prior `auto_upsample=False, factor=377` evidence as a fixed-density control, never as the only headline comparison.

Use the same five archived physical clips first, not a cherry-picked new workload. Reuse exact raw inputs and hashes, target resolution, device, thread counts, native precision, batch8, no preview/video, equivalent in-memory PNG bridge, sorted CPU event endpoint, and explicit CUDA timing synchronization. Three independent episode repeats per case, rotating method order, warm model separately and reset sensor/RNG per trial. Preserve all raw output event arrays; use matched writer only for the separate serialization measurement.

Record per trial:

- Ours actual per-interval samples and all bound fields, capped/uncapped/recommend_render, fast-path status and fallbacks;
- V2E actual per-batch factor, full output frame timestamps, max/median interval, total frames, actual minfactor, native lowpass warnings and maximum eps where observable;
- exact physical interval and first/last sample/event times; no posthoc timestamp stretching;
- wall time, event count, original dtype, output SHA256, peak allocated/reserved/process memory, real GPU UUID, and source/profile hashes;
- sampling policies are different; **same cutoff and source trajectory do not imply same discretization error or event quality**.

The official helper `get_interpolated_timestamps()` contains an auto-mode TODO. Preserve the audited method that derives midpoint times from returned relative sample starts and per-frame intervals, rather than applying a fixed-factor helper or stretching the output to an arbitrary end time. Validate monotonicity and per-frame correspondence.

For a later steady-state quality experiment, a common physical warmup prefix can remove first-frame initialization asymmetry, but must be preregistered and measured as a separate experiment; do not silently rewrite the existing trial boundary.

## Interpretation

If native auto leaves Ours at 377 and V2E near 2, report that result, including any faster V2E point and numerical warnings. It reveals a real difference in each simulator's sampling policy, not a reason to reduce Ours quality settings or force V2E to match Ours. Before any accuracy claim, use independent reference/convergence checks. The same sparse physical cache and the dense temporal interpolation workload are distinct concepts.
