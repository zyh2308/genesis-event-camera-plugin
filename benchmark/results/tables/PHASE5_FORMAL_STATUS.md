# Phase 5 — Frozen Formal Quality Benchmark Status

## STOP: no formal benchmark result

Phase 5 execution was authorized by
`benchmark/FORMAL_RUN_APPROVAL_PHASE5.yaml`, but no formal benchmark result
was produced.  The run was stopped under the resource-safety gate before a
complete LR/RL pair existed, and no formal metric was computed.

The frozen protocol, pair manifest, core, replay, parameters, cadence, seed,
and metric definitions were not changed.

## Provenance

- current repository commit: `15479d97154274796043d5e5cc42a336f57b2dd9`
- frozen core: `e79e656ac00f7847b98db9e12bf43b093f3afff1`
- runtime/check commit: `5bb3d079b64118de4d76570ef98ca37841c5acab`
- Phase 4.7 readiness commit: `70db6c7117a4aa4d585d3a1882668eea82a18212`
- protocol v3 SHA-256: `c6d0926147e3fb08b15892c1f9ee368c5d7380ad3326cca0b8d962ae1f78d763`
- pair manifest SHA-256: `86c9596bb3c6e79788e1ae5aa627319ade886a64c88571f4ccc5e7ede6fbc175`
- formal metric execution: **NO**
- ranking/significance analysis: **NO**

## Real-source loader gate

The declared real files were copied without content changes into the formal
run workspace and fully decoded with the official vendor ECF filter copied to
the same user-owned workspace.  No first-event alignment was used.

| direction | complete CD/events | first `t_rel` | last `t_rel` | leading silence | trailing silence |
|---|---:|---:|---:|---:|---:|
| LR | 23,675,387 | 6,272 us | 1,756,258 us | 6.272 ms | 0 ms |
| RL | 23,766,581 | 5,824 us | 1,775,800 us | 5.824 ms | 0 ms |

The ECF library used by the loader was `libH5Zecf.so`; its run-local path was
`.../external_formal_results/formal_v3_20260929T095851Z_15479d9/ecf_plugin`.

## Genesis/Ours attempts

### Attempt 1 — default installed Genesis

- run: `formal_v3_20260929T095143Z_a0db475`
- result: **STOP**
- reason: default installed `genesis 1.2.2` exposed no `radiance=` argument;
  the native HDR gate therefore failed.
- no display RGB fallback was used.

### Attempt 2 — declared HDR overlay

- run: `formal_v3_20260929T095851Z_15479d9`
- overlay: `/home/zhaoyuhan/Eventbased_WAM/code/genesis_renderer_hdr_v2`
- environment: `PYOPENGL_PLATFORM=egl`, `GENESIS_BACKEND=cpu`,
  `GENESIS_EVENT_DEVICE=cpu`
- fixed replay: LR, 1280×720, 0.01 s cadence, 177 endpoint frames through
  1.76 s
- result: **STOP before completion**
- evidence: the overlay accepted the HDR render request and began the real
  CPU replay, but after roughly five minutes only 3 display frames had been
  archived.  Extrapolated completion time was several hours per direction,
  while all GPUs were occupied by other users' jobs.  The process was stopped
  by an exact tmux target after the resource gate was reached.
- the partial archive is retained remotely; it is not a formal result.

Because the LR formal replay did not complete, the following were not
validly produced: complete Ours event stream, complete segmentation/HDR
formal archive, RL replay, official V2E replay, paired outputs, and the four
primary metrics.

## Required next condition

Resume only in an authorized environment with sufficient free compute to run
both fixed 1280×720 Genesis directions using the declared HDR overlay.  Do not
change resolution, cadence, precision, replay duration, event parameters,
metric definitions, or the frozen core to work around this resource stop.

