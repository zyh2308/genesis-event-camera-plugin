# Phase 4.6 status: formal execution protocol completion

**Date:** 2026-09-29  
**Formal benchmark executed:** **NO**  
**Frozen core:** `e79e656ac00f7847b98db9e12bf43b093f3afff1`  
**Phase 4.5 harness:** `46ce49b941d36d2c145bd9e369dc8ef53362b8cc`

## Completed before any formal metric

| Check | Result | Evidence |
|---|---|---|
| Immutable v1 protocol | PASS | `protocol_v1.yaml` SHA-256 remains `b1b4519446bbc821220b3502a0eb3811bf8c0baf8bd457eca022a0d082ebdd97` |
| v2 protocol and hash | PASS | `protocol_v2.yaml`, SHA-256 `c6382e3cd94ce02c2f92d8d12a8d5604affb8a4fd9a17b0694c7609120ab28df` |
| Independent preparation approval | PASS | `FORMAL_RUN_APPROVAL.yaml`; formal execution remains false |
| Pair manifest | PASS | LR/RL one-round pairs, SHA-256 `e62e7cc0fb18730c53213277ceea20f7598b7f22cca102b0695201d857b2c9b9` |
| Equal-duration windows | PASS | fixed `0.005 s`, normalized fractions `0.25, 0.50, 0.75` |
| Frozen-core integrity | PASS | `git diff e79e656 -- genesis_event_plugin/` is empty |
| Config hashes | PASS | `ours.yaml`, `v2e.yaml`, `esim.yaml` match `CONFIG_HASHES.json` |
| V2E cadence contract | PASS | effective cadence `10,000 Hz`, `dt=0.0001 s`, official `maxeps=0.3`, `eps=0.1885` |
| Previous V2E warning explanation | PASS | 100 Hz with 300 Hz cutoff gives `eps=18.8496`, above the official guard |
| Metric/schema smoke | PASS | `metric_smoke_test.py` |
| Window extractor smoke | PASS | `window_smoke_test.py` |
| V2E cadence smoke | PASS | `v2e_cadence_smoke.py` |
| Baseline provenance smoke | PASS with unavailable components recorded | Existing Ours/V2E artifacts validated; ESIM/IEBCS remain unavailable |

## Blocking condition

The actual Genesis HDR renderer/adapter smoke could not run in the current
environment because the active Python environment has no importable `genesis`
package:

```text
ModuleNotFoundError: No module named 'genesis'
```

Therefore the formal replay remains `definition_only_pending_formal_scene_generation`.
No display-RGB fallback was used, no HDR v3.1 formal Ours stream was claimed,
and no formal comparison metric was computed. The protocol explicitly requires
the HDR overlay and will stop rather than silently downgrade the input domain.

## Formal execution boundary

The only currently frozen comparison is a future fixed-configuration,
coarse-distributional system-level comparison:

```text
Real EVK4 LR/RL reference
        │
        ├── same fixed 5 ms windows
        │
        └── Genesis replay definition
              ├── Ours-v3.1: HDR/radiance + depth + segmentation + poses
              └── V2E: same replay image sequence, effective 10 kHz cadence
```

The real EVK4 acquisition still has no encoder trace or hardware trigger, so
this must not be described as a strict synchronized fidelity benchmark. CD, GD,
per-bin temporal error, composite scores, and rankings remain disabled.
