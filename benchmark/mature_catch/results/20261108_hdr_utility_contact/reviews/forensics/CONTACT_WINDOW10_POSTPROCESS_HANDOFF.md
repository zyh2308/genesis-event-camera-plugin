# Window10 paired-run closeout audit

2026-11-08. Only local postprocessing files were edited. No SSH and no production/W changes.

Deploy these two files together:

- `postprocess_contact_pair.py`: SHA256 `4890935f91b5ef6b56bd70a2335b0fff92c16b5a282c621d774d705f99159c0c`
- `audit_contact_observation_windows.py`: SHA256 `d2d78bdce09f1c4abd726cccd8df7f2996eef8d264181a19847cc8e3e0f5337a`

## Compatibility and output

The existing physical outcome scoring, analyzer, general raw-event audit, planned-case retention and hash generation are unchanged. The expected window is read from `EVALUATION_PROTOCOL.shared_estimator_overrides.CATCHIT_EVENT_OBSERVATION_WINDOW_MS`; missing means zero. Actor `protocol.event_observation_window_ms` must match it. At zero, the new helper is not imported and no window artifact is required.

At ten, the helper audits every completed actor and writes `OBSERVATION_WINDOW_AUDIT.json`; its hash is included in `DATA_COMPLETION_RECEIPT.json`, and the normal final manifest includes it. `CONTACT_PAIRED_COMPARISON.fixed_parameters.observation_window_ms` is explicit (0 for old/default runs, 10 for this run), as requested for figure provenance. An old hardcoded “all18actors” receipt description now correctly says “all actors”.

## Checks

- Protocol / sensor window setting and the loaded-module receipts for window helper, tracker and sensor match the sealed runtime hashes.
- All consumed parents map uniquely to the real packet-audit index, endpoint, availability, event count and actual raw timestamp support.
- Online processing is measured and occurs at the first original control query at which the parent is available.
- Every child retains the parent availability and identity; original raw-index intervals partition the parent exactly once, with contiguous 10 ms time windows, including actual carry-in history. Repeated child endpoints across arrival parents are retained and cross-checked.
- Counts, append count, carry totals, exact partition status, equal timestamp accounting and global event/measurement order match the raw arrays and stored samples.
- Each camera transform is independently reconstructed from the actual HDF camera endpoints (exact endpoint or Slerp/linear translation), with the same declared child-end convention and only already-available camera support. Camera origin matches that transform. No target GT is used.
- Detected sample times/support lie within the actual child events, and `event_samples.json` contains exactly the audited appends. Controller trace event support matches available detections and does not use future events.

## Pending packets are legitimate

The consumer requirement is based on **the last actual `trace.t_before` control query**, not physical terminal time and not the last captured frame. A packet that arrives after the last query can remain pending even when physics continues past its arrival time before terminating. The audit reports pending parent/event counts; it does not require all generated packets to be consumed. Similarly, after estimated grasp stops further capture, all later control polls need not have new packets.

The first real actor schema supplied by root (37 captured / 37 consumed, final control2.48 s, physical2.52 s) matches the recorded parent timing and child schema. This was a schema/time review, not a claim to have audited its remote raw data.

## Verification completed

Local `test_contact_observation_audit.py`: **6/6 PASS**. The valid fixture is produced by the actual production audit module and includes true carry-in, a repeated endpoint and an intentionally pending packet available0.13 s before physical end0.15 s but after the final control query0.12 s. Corruptions of protocol, availability, consumed-parent set, event-index ownership, camera geometry/support, processing time and sample append count are rejected. Both deployment files compile.

Root can audit one completed server actor immediately, before the 12-run group completes:

```python
from pathlib import Path
import json
from audit_contact_observation_windows import audit_run
run = Path('/actual/completed/actor/output')
spec = json.loads(Path('/actual/paired/root/EVALUATION_PROTOCOL.json').read_text())
protocol = json.loads((run/'protocol.json').read_text())
print(json.dumps(audit_run(run, protocol, spec), indent=2))
```

The helper needs the existing server numpy/scipy/h5py environment; it invokes no renderer, physics, estimator, target labels or GPU. Full actual-actor audit remains for root to execute using the remote arrays already present.
