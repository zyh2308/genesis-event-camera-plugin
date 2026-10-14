# Stage 5 Q2 — v4 event interface

This directory contains the Stage 5 Q2 interface, schema, and validation
artifacts. It is not a frozen `protocol_v4` and it contains no formal result.

The intended data path is:

```text
raw artifact + sidecar -> QualityAdapter -> canonical [x,y,p,t_s] -> future metric
```

The adapters reuse existing loaders and perform only explicit layout, unit,
and polarity conversion. They do not align streams, search temporal offsets,
normalize timestamps, fit parameters, or modify simulator output.

The sidecar contract is strict: the adapter checks that the declared event
count and observed first/last timestamps match the loaded artifact. The
declared interval is sequence-local in the raw timestamp unit; an absolute
acquisition origin, when needed by a loader, is separate metadata.

`tests/pecs_metric_selftest.py` exercises only legacy PECS-style metric
sanity behavior. It is a self-test, not a benchmark and not a claim about
simulator quality.
