# Complete but unexecuted paired Census short-study preparation

Everything here is isolated. No server connection, GPU phase0, optimizer update, smoke,
study, lambda calibration, synthetic validation or real scoring is started by preparing
or importing this package. `cpu_preflight.json` covers local synthetic CPU fixtures only;
it does not substitute for a real four-arm smoke gate.

Canonical methods: EventGenesis/V2E, seed0. Arms: `native_control`, `census`. All four start
from their separately frozen formal **last8910** and use the exact same original control
200-batch schedule and shared affine seeds. No motion sampling is used. Study endpoints
are fixed at additional200 / total9110, with all528 validation windows and original gates.

The historical helper files are byte-identical and imported for restore/evaluate/save.
Their storage checkpoint schema retains its historical helper name; the new main, command
contract, completion receipts, arm names and objective descriptions are canonical and do
not pretend Census is a motion-sampler arm. Original model/loss/source/CP bytes are checked
before and after. The14-file source tree is pinned. CPU admission independently records
the exact82 named/trainable parameter inventory; gradient counts are a different concept:
dt1 has70 connected finite-gradient tensors, unused parameters contribute zeros to the
complete norm. The full signature is retained and compared on GPU.

The new Census loss follows UFlow's soft transform/Hamming and `abs_robust_loss(.01,.4)`
formula. UnFlow has a related transform, but its tail Charbonnier is not claimed identical.
Pixel SUM and original four equal scales are retained; smoothness is unchanged with
coefficient10. Predicted-support/patch-border masks are detached; no GT masks or backward
network are introduced. This is a new declared downstream objective package, not an exact
published STE training reproduction. Patch erosion and Census are changed together, so a
future improvement cannot be uniquely attributed to just one component.

**v2 input domains:** raw NPZ gray_prev/next are checked strictly finite[0,1], including
actual dtype/min/max. Known native affine/OpenCV/predicted-warp interpolation can legally
produce1+one floating ULP from a white input. Only those explicitly flagged internal
tensors receive an8*dtypeeps domain-check margin. Their values and gradients are not
clamped, normalized or rewritten. Default external inputs remain strict;1.001 is rejected
even internally. The old strict prototype is preserved unchanged as `soft_census.py` for
the CPU regression that demonstrates the failure. Actual original affine-white endpoint
fixtures then pass v2 without changing their values.

**phase0:** the first8 original control TRAIN batches for each method, no optimizer step.
Use the full requires_grad model parameter set, allow unused gradients as zeros, verify70
non-None finite photo gradients. Freeze a single common
`lambda=1/median(norm(g_census)/norm(g_native_photo))` over16 observations. This explicitly
supersedes the earlier standalone prototype's unused RMS/flow-output calibration example.
It never uses GT, validation or real scores to choose lambda. GT is only retained in
pairing hashes; no GT-gradient oracle is reported here, avoiding native augmented-GT
interpolation/validity-boundary ambiguity. Record predicted Census support fractions;
decreasing scalar loss alone is not evidence of better flow.

Suggested future CPU/GPU CLI, **only after the root approves deployment and supplies its
safe actual resource queue**:

```
python -B sources/run_census_paired_short.py cpu --contract sources/contract.json --out ROOT/cpu
python -B sources/run_census_paired_short.py phase0 --contract sources/contract.json --out ROOT/phase0
python -B sources/run_census_paired_short.py smoke --contract sources/contract.json --out ROOT/smoke
python -B sources/run_census_paired_short.py study --contract sources/contract.json --out ROOT/study
```

The contract fixes absolute ROOT and sources location; actual CLI paths must resolve to
those registered paths. Each output must be new. `phase0/smoke/study` require actual CPU
admission. Study requires all four real one-update smokes and restored model/Adam evidence.
The queue must set CUDA_VISIBLE_DEVICES to the inherited GPU3 UUID, protect the existing
lease, apply an external hard timeout and clean only its own child group. This bundle does
not acquire or occupy resources itself.

The default recomputes full528 warm parity because existing reused snapshots have not
been independently bound by this new contract. Reuse is implemented only through an
explicit before/parity/receipt record binding the exact source CP and model-state SHA,
recipe/data/source identities and full528 inventory. If these are uncertain, leave the
default null and recompute. No best real or synthetic subset is chosen after seeing loss.

CPU test command: `python -B sources/test_census_contract.py`. On a server the actual pinned
trainer path comes from contract; local tests use the exact archived trainer plus the WSL
pilot/compat modules. Tests cover unchanged control replay, median versus RMS difference,
rejected GT/validation/partial calibration, unchanged quality gates, explicit unused
parameter norms, raw versus internal saturated input domains, actual native augmentation,
and independent70-state Adam/model checkpoint rebuild using a tiny CPU model.

The study quality gate is all-valid and common-event-active EPE improvement for both
generators, with no stratum degrading by more than5%. All methods remain visible on
failure; no long training or real scoring follows automatically. Single-seed short-study
results would be development evidence, not a final simulator ranking.
