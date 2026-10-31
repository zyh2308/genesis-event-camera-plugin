# R1.5 strict intermediate Direct-HDR rerender (run I)

## Result

Run `r1_5_strict_direct_rerender_20261031i` completed on g1, GPU1 (RTX 3090, UUID `GPU-0262d545-e9bd-7fcb-66e5-2ef81465c460`), with source commit `26de5e410ecc550a9dc7f61b25aa2926d6399e0a`. It replays saved states only; Genesis physics was not advanced. The HDR rasterizer rendered the nine exact 1 ms interior states in each of episodes 004 and 005 over `[0.5805, 0.5905]` s.

At all 18 new timestamps, rerendered scene-linear HDR matched the archived direct-HDR pixels exactly (maximum absolute RGB difference 0); segmentation mismatch was zero. Replaying DVS over the 1 ms images produced the same raw events as the archived sequences, element for element:

| Episode | Events | ON/OFF | Pixel-polarity count L1 | Exact time/pixel/polarity matches |
|---|---:|---:|---:|---:|
| 004 | 625 | 354 / 271 | 0 | 625 / 625 |
| 005 | 939 | 536 / 403 | 0 | 939 / 939 |

This confirms that the verified runtime can reproduce these archived intermediate observations and that the event path consumes them consistently. The frames and states come from the same archive and renderer setup, so this is a same-state implementation check; it does not establish general physical truth or accuracy for unsampled times.

## Cost and timing boundary

The run used Genesis 1.2.2 from `/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot/.venv-robot/bin/python`, EGL device 1, rasterizer backend, and `Camera.render(radiance=True)`. The imported `camera.py` SHA256 was `78c18ffada07a98657ddf11326e833fc928431ecead674434aadb25dc023b714`; all renderer module paths and hashes are in `output/summary.json`.

- Input load: 0.896 s; Genesis initialization: 1.821 s; scene build: 57.952 s; warmup render: 1.447 s (excluded from steady-state frame cost).
- 18 strict extra renders: camera call including GL readback 0.289 s total (16.04 ms mean); returned-array materialization 0.0066 s; exact-state restore 0.0385 s.
- DVS over archived reference and rerendered sequences: 0.793 s total (per-episode reference/rerender times are in the summary).
- Main process time before artifact hashing: 69.279 s; `/usr/bin/time` wall time: 76.08 s; peak RSS 1.57 GB.

The output sequence is only available after the right endpoint at simulation time 0.5905 s, and the manifest explicitly records that online closed-loop ordering was not tested here. The per-frame wall time is slower than the 1 ms simulated sampling interval. This is a measured offline acquisition cost, not a live-latency claim. The project closeout does not require wall-clock real-time execution.

## Reproduction

The g1 run used the source snapshot recorded in `source/source_provenance.json` and the verified shared read-only cache:

```bash
bash source/benchmark/robot_closed_loop/diagnostics/launch_r1_5_strict_hdr_rerender_remote.sh \
  g1 r1_5_strict_direct_rerender_20261031i \
  26de5e410ecc550a9dc7f61b25aa2926d6399e0a 1 \
  GPU-0262d545-e9bd-7fcb-66e5-2ef81465c460 full
```

The launcher intentionally refuses to overwrite this run's output. To reproduce, use a new run ID and a config whose `run_id` matches, after its resource gate succeeds. The equivalent measurement command is recorded in `logs/preflight.txt`; cache paths and all input hashes are in `output/run_manifest.json`.

The failed prior attempt H is retained separately on g1. It exited before rendering because a boolean NumPy mask had been applied to a Python trace-row list. That implementation bug is fixed in the source commit above; attempt H is not part of these results.

## Artifact hashes

`output/SHA256SUMS` verifies the seven runner outputs. The top-level `SHA256SUMS` covers all retained run artifacts and this report. The remote evidence archive SHA256 is `b206d9198982a36db1bd430ec69db3531d73c1efc8834bc1884262fdee0a6137`.
