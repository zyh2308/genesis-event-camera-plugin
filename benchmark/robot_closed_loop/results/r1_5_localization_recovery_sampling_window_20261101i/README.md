# R1.5 sampler-count probe, episode 004

This is a **sampler-count diagnostic only**. It is not a strict-event reference
and it contains no localization score.

## Finding

On the failing episode 004 (seed `2026104134`, 0.30 m/s, initial lateral
offset `+55 mm`), all 28 one-millisecond intervals in `[0.574, 0.602] s`
requested seven adaptive samples. The physical trace used by this probe was
spaced at `1/6000 s`; it cannot provide the required `1/7000 s` intermediate
timestamps as exact saved states. The strict path was deliberately disabled in
this probe, so `strict_sampled_events.npz` records fallback interpolation and
must not be treated as strict reference data. The Direct-HDR side uses only
the 1 ms endpoint renders. Neither event archive was scored by a model.

The run therefore identifies a sampling-grid incompatibility that prevented
the previous paired localization run from reaching inference. It does **not**
show that the event simulator's position estimates improved or regressed.

## Runtime and provenance

- Host/run: g1, `r1_5_sampler_count_window_20261101i_gpu0`; episode 004 completed.
- Assigned resource: GPU 0 RTX 3090, UUID
  `GPU-fd9f1bd4-83c4-cd74-0c88-241db6ec7e23`; the Genesis simulation backend
  fell back to CPU. The user allocation ended at `2026-11-01 08:57:50 UTC`.
- Code commit: `cef79612405180e1a560fb289b5fe136623c52ff`; runner SHA256
  `e5c74d9351a2a85920b62e8de6da414fb5bde88ab1b325953f0de68d1d1d60c8`.
- Frozen core: `r1.5-strict-sampled-20261031`, commit
  `90f792d166f000ef4f9e072a97633cc4ad39c8b7`; profile SHA256
  `3525cad9e84982a2b835b1b7d32428b94be24508fe28faf9f0d79002dfb10818`.
- Scene config SHA256
  `fff41de71fa322421a463c4b64dab93d065b0022b3943f6c09e9c4e39c769d69`;
  experiment config SHA256
  `ccd725890f9c20471cdd8d1def4be7a4f8f462302ba46ba5300d47e5f2134484`;
  control config SHA256
  `3d289c2bef1c562b26f9b713c2a826b813a286a1f7e21f9ead0a01c63c5e0d80`.
- Probe interval: 28 ms. 29 HDR frames rendered; scene build 53.04 s, render
  0.52 s, strict-plugin probe 25.27 s, Direct-DVS 0.35 s, total 80.99 s.
- The episode 005 process began scene initialization but waited on a shared
  Quadrants cache lock while compiling. It was interrupted when the assigned
  slot ended; no episode 005 frames or events were produced. No cache was
  cleaned and no other process was touched.

## Files and reproduction

The episode 004 files and per-file hashes are in `episode_004/`. The 7.2 MB
physical trace is retained here for direct replay. The remote run also retains
the original launcher and episode 005 log at:

`/home/zhaoyuhan/Eventbased_WAM/runs/robot_scenes/genesis_event_camera_plugin/r1_5_sampler_count_window_20261101i_gpu0/`

Re-run the sampler-only probe with the committed launcher and an assigned
GPU reservation:

```bash
bash benchmark/robot_closed_loop/diagnostics/launch_r1_5_sampler_count_probe.sh \
  /home/zhaoyuhan/Eventbased_WAM/runs/robot_scenes/genesis_event_camera_plugin/<new-run-id> \
  GPU-fd9f1bd4-83c4-cd74-0c88-241db6ec7e23 \
  /home/zhaoyuhan/Eventbased_WAM/.gpu_reservations/<matching-reservation>.json \
  /home/zhaoyuhan/Eventbased_WAM/runs/robot_scenes/genesis_event_camera_plugin/r1_5_localization_fineclock_pair_20261101d_gpu0/fineclock_traces \
  benchmark/robot_closed_loop/configs/scene_r1_5_localization_fineclock_20261101c.json \
  benchmark/robot_closed_loop/configs/r1_5_localization_recovery_fineclock_20261101c.json \
  4
```

For U=6 and U=7 strict timestamps, the next exact physical-state grid is
`1/42000 s`: each 1 ms interval is 42 steps, and the `1/6` and `1/7` sample
fractions land on integer physics steps. The paired renderer now renders
1 ms Direct-HDR endpoints and renders strict intermediate images on demand
from exact saved states. It does not synthesize poses. This code change has
been committed as `241124048ca0ce93dbecfb6c319f96704bcb4b2b`; it passed syntax,
shell, and grid-arithmetic checks. A newly allocated GPU run is still needed
to verify the complete strict/Direct event and localization path.

The paired run launcher and experiment horizon were then committed as
`32e88262e2d50df0daf94d417d7348d92e3c9f70`. Its experiment config SHA256 is
`662f858b2783224c4de3c31a684a0ee308ec1a794f7c7baccaf4ec8d9870ffbd`; the
0.602 s cutoff includes the final 0.6005 s localization query while avoiding
unscored tail rendering.
