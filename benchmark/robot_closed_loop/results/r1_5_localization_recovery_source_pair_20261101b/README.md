# R1.5 localization recovery: strict/Direct-HDR renderer preflight

**Outcome:** this was a runner and input-availability diagnostic, not a completed source comparison. Episode 004 stopped after its first 1 ms bracket because the frozen strict sampler requested a real frame at `0.000166667 s`, which is absent from the verified 0.5 ms state trace. Episode 005 was not started. No localization model was evaluated and no training was run.

## What ran

- Branch `exp/robot-closed-loop-v1`; execution runner commit `03c5b0a548bf5c71e49cf7f340a8ebfe48ba42b5`.
- Frozen plugin tag `r1.5-strict-sampled-20261031`, commit `90f792d166f000ef4f9e072a97633cc4ad39c8b7`.
- Scientific strict profile SHA256: `3525cad9e84982a2b835b1b7d32428b94be24508fe28faf9f0d79002dfb10818`.
- g1, user-assigned GPU 0 RTX 3090 UUID `GPU-fd9f1bd4-83c4-cd74-0c88-241db6ec7e23`; Genesis physics backend CPU, native GL radiance renderer enabled through the existing `gec-robot/.venv-robot` environment (Genesis 1.2.2).
- Episode 004: 0.30 m/s, start y `+55 mm`. It rendered the initial frame and two saved-state frames, then stopped fail-closed. The verified source trace is sampled every `0.5 ms`.

## Result and interpretation

The strict sampler chose six samples for the 1 ms bracket and requested interior radiance at `0.166667`, `0.333333`, `0.500000`, `0.666667`, and `0.833333 ms`. Only the midpoint is present in the saved trace. The first missing request produced `STRICT_PROVIDER_NEEDS_UNSAVED_ACTUAL_STATE`; the provider did not synthesize or mislabel a state. The output contains 249 Direct-HDR events from the first millisecond, zero strict events, zero completed strict intervals, and no valid common five-bin comparison. The cache consistency flag is false because the frame check stopped at `t=0`; it is not evidence that the renderer disagrees with the cache.

This identifies a sampling-contract gap before the image/event comparison: a 0.5 ms state log cannot satisfy this strict profile's adaptive 1/6 ms request schedule. It does **not** explain the earlier localization failure and does not establish whether the strict generator or Direct-HDR yields better localization.

Scene construction took `50.474 s`; three saved-state HDR renders took `0.0524 s`; Direct DVS took `0.0298 s`; the strict attempt took `0.7485 s`; total episode process time was `53.432 s`. These are partial-run costs only and must not be extrapolated to a full episode. GPU 0 was idle at release (`08:03:00Z`); the reservation was released with the sampling gap recorded.

## Attempts and code fix

1. The base `Eventbased_WAM` environment does not accept `Camera.render(radiance=True)`, so it could only produce display output and was rejected for this experiment.
2. The existing `.venv-robot` environment exposes native scene-linear float HDR, but the first corrected launch exposed a runner bug: Genesis `Scene.cur_t` is read-only. Commit `03c5b0a` adds a benchmark-only replay-clock proxy; it does not advance physics or modify the frozen core.
3. With that fixed, the run reached the frozen strict provider and exposed the missing exact state at `0.166667 ms` described above.

## Reproduction and artifacts

The exact g1 launch scripts, logs, partial raw event arrays, per-episode summary, original state trace, and checksums are archived under [`strict_direct_pair_r1_5_localization_recovery_20261101b_gpu0_20261101T073805Z`](strict_direct_pair_r1_5_localization_recovery_20261101b_gpu0_20261101T073805Z/). The recorded episode command is in `attempt_03_robot_env_hdr/launch_pair.sh`; use a **new unique output directory** for any rerun because episode outputs are intentionally fail-closed. Final archive manifest SHA256: `a19c3a30c1d85692e88cc380f51448a36aeb5f65b45144c89e4c20a8d7bc5e0c`.

Key checksums:

- episode summary: `96d25ab8061673c5c234fa01b71ae854f7ad5ce8b499c152bcb3ebd1debb41e9`
- partial Direct-HDR events: `994356e6b74d4c23c3f39f4a9aac78771de358c68370a52cdc928126a64c444d`
- empty strict event array: `0497b86a786a9158f18aa60170d027bce4095ca2867ba68c47a9552a8db100d2`
- updated runner: `457e94c1705ac116f8016819b9baefc25b24a3632f2b2399111406ac8e5aee2b`

## Next decision needed

Read-only source inspection resolved whether the current integrator has hidden substeps: the scene builder passes `SimOptions(dt=0.0005, ...)` without a `substeps` override; Genesis `SimOptions.substeps` defaults to `1`, and `Simulator.step()` advances exactly that configured count. There are no 0.167 ms solver states to expose from this rollout.

Relevant runtime source hashes on g1:

- `genesis/options/solvers.py`: `ccea060d5769b02776954cd5f6b6c7a07922c234cecf19a3473e013139263090`
- `genesis/engine/simulator.py`: `b385b363dfb56512ae189d3ad86fd9bd75acc7209e2cf4b9c934ec76e98073da`
- `genesis/engine/scene.py`: `ec1e062c8e59a6fa973bcca7cc75618514a6b672ebb69cb7a30f62430139a48c`
- benchmark `robot_env.py`: `8a941724f9041a5281a3e989f9374e94308c2551e21ff54a247c8fd453722cb4`

For a valid full strict comparison, the remaining choices are:

1. Keep the current strict profile and physics clock, but accept that exact strict samples cannot be produced from these episodes. This leaves only a non-strict 0.5 ms development comparison, which is not the frozen-profile acceptance run.
2. Generate a new matched physical rollout with a `1/6000 s` step (or finer), then render both event sources and regenerate labels from that same new rollout. This preserves source fairness but changes the numerical physics clock and is not an exact replay of episodes 004/005.

Do not substitute pose interpolation and call it strict actual-state sampling. After the sampling contract and clock choice are resolved, render both short episodes, compare raw events and five-bin inputs, then evaluate the frozen checkpoints. No training decision is supported by this partial run.
