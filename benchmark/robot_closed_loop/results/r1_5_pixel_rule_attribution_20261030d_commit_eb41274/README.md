# R1.5 pixel-level warp rule attribution

## Finding

The current path is `physics_plugin.py → physics_interpolator.py → physics_warp.py`. Runtime imports resolve to `PhysicsInterpolator.interpolate()` and `depth_aware_bidirectional_warp_projected_diagnostics()` / `_splat_projected_diagnostics()`; the legacy `interpolator.py` is not used here.

The strongest real-clip defect is inside `_splat_projected_diagnostics()` (`physics_warp.py`). Every positive bilinear weight enters the target z-buffer. The nearest-depth sample wins before weights are normalized, so a tiny foreground footprint can discard a much larger, slightly farther footprint and then be normalized to 100% of the output pixel. This turns a tiny subpixel projection difference into a full-pixel brightness jump.

One center-clip example at `(x=101, y=2)`, `t=0.5815 s`:

| Quantity | Value |
|---|---:|
| Direct-HDR luma at the exact saved state | 0.86914 |
| Reconstructed luma | 0.03912 |
| Absolute log-luma error | 3.101 |
| Selected endpoint-1 source | `(102, 3)`, segmentation ID 10 |
| Its projected depth and bilinear weight | 0.554865 m, `1.23e-5` |
| Rejected endpoint-1 source at the same target pixel | `(101, 2)`, same segmentation ID 10 |
| Rejected source depth and bilinear weight | 0.555581 m, 0.99298 |

The selected contribution is about 80,900 times lighter than the rejected contribution, yet the current z-buffer rule makes it determine the full pixel. The `+55 mm` clip has the same signature at `(148, 2)`: luma 0.03720 versus direct 0.87646, error 3.160 log; its nearer contribution weighs 0.000975 while the rejected contribution weighs 0.91316. These are actual cached clip pixels, not the synthetic helper probes.

## Single-rule cached interventions

The active NumPy path reproduced all 10 archived intermediate frames exactly (max absolute difference `0`) for each episode. We evaluated the saved 1 kHz Direct-HDR frames at 0.5815–0.5905 s; these are exact renders at their logged states, but do not define the hidden physical image between those times.

The diagnostic coverage candidate changes only how splat contributions are resolved: it sorts by projected depth, treats bilinear weights as pixel coverage, fills the remaining pixel area from farther layers, and normalizes only observed coverage when support is incomplete. It uses the existing `1e-5 m` depth tie band. It does not change thresholds, remove events, blur images, or use segmentation/GT to alter outputs.

| Episode | Baseline pixel-times with `|log error| > 0.2` | Resolved by coverage candidate | Fraction | Remaining candidate pixel-times > 0.2 | Baseline-bad log-error sum reduced |
|---|---:|---:|---:|---:|---:|
| Center 004 | 8,555 | 7,917 | 92.5% | 665 | 93.2% |
| +55 mm 005 | 7,851 | 6,626 | 84.4% | 1,377 | 85.0% |

The result is consistent across the nine interior times only (0.5815–0.5895 s): the candidate resolves 92.2% / 84.1% of baseline-bad pixel-times. Its maximum remaining error is 1.43 / 1.42 log, so the image path is substantially improved in this diagnostic but not fully explained or validated.

Strictly tiny winner coverage (`≤1e-4`, with same-endpoint farther coverage `>0.05`) occurs on 1,536 center and 561 `+55 mm` baseline-bad pixel-times; the candidate resolves 99.3% and 97.5% of those. At a 5% near-layer coverage bound, 8,040 / 5,648 baseline-bad pixel-times match the signature, and it resolves 98.1% / 94.9% of those. The latter statistic is a sensitivity description, not a tuned production threshold.

### The two review hypotheses

1. **Cross-endpoint z comparison:** the helper counterexample is real: `next_primary` can choose a farther endpoint-1 background over a nearer endpoint-0 foreground. A diagnostic “choose whichever endpoint z-buffer is nearer” intervention resolved only 10.2% / 12.5% of the baseline-bad pixels and reduced their summed error by 3.8% / increased it by 0.4%. It also worsened 1,092 / 1,271 already-bad pixel-times and created 19,939 / 19,433 total pixel-times above the error threshold. Endpoint images describe different points/times, so a simple binary nearest-endpoint rule is not a valid general correction. This behavior is not the dominant explanation for the real-clip error under that intervention.
2. **Tiny positive splat weights:** the per-endpoint coverage intervention fixes most of the real-clip error. The per-pixel source trace directly shows the winning tiny weight and the excluded large-weight source. This is the leading confirmed rule-level cause in these two clips.

At the exact `t1=0.5905 s` endpoint, the mathematical identity-projection intervention gives exact equality to the cached observed endpoint for both clips and removes all 348 / 160 endpoint error pixels. That is a narrow endpoint invariant; it does not explain the interior-time errors.

## Event and state evidence

With the same ideal DVS configuration and a separately maintained continuous per-pixel history for each input sequence, the 10 ms window produced:

| Episode | Current reconstruction | Coverage candidate | Exact saved Direct-HDR |
|---|---:|---:|---:|
| Center 004 | 6,184 events | 407 | 625 |
| +55 mm 005 | 4,313 events | 682 | 939 |

This confirms the image intervention changes the event stream in the same window. Event-count closeness is not a claim that the candidate is physically correct.

The current `hold` fallback has 0 unsupported pixels per saved frame in episode 004 and 1–2 in episode 005, compared with roughly 800–950 `>0.2` log-error pixels per frame. It cannot explain the dominant error share in this window; we did not expand into midpoint-hold experiments.

The earlier phrase “DVS threshold history couples pixels over time” was inaccurate. In the ideal response used here, the DVS state is per pixel. In a separate 10 ms replay of the previous robot/nonrobot region intervention, the segmentation-derived robot mask switched on 2 pixel-time boundaries in episode 004 and 18 in episode 005. The event-count nonadditivity in that short window was 4 and 50 events. This is compatible with a time-varying mask making an individual pixel alternate between image sources and thereby changing that pixel’s later threshold crossings; it is not evidence of cross-pixel DVS coupling. The earlier full-window nonadditivity values (13 and 277 events) have not been decomposed further.

## Controls

- **Static identity:** active and coverage paths reproduce the image exactly; ideal DVS produces zero events over 4 ms.
- **Cross-endpoint occlusion:** current `next_primary` returns the background value `0.01` for a near-depth-1 foreground versus far-depth-2 background; the nearest-z diagnostic returns foreground `1.0`. This establishes the helper behavior, not that binary endpoint selection is a suitable clip fix.
- **Same-endpoint partial occlusion:** at full foreground coverage the candidate returns foreground `1.0`; at 25% foreground footprint it returns `0.2575`, matching `0.25×1 + 0.75×0.01`. The legacy z-buffer returns `1.0` for both.
- **Subpixel continuity:** moving the near sample by `+1e-6 px` changes the legacy output from `0.01` to `1.0` (`0.99` jump); the coverage candidate changes to `0.01000099` (less than `1e-6` change). The negative and zero perturbations remain `0.01`.

## Reproduction and provenance

CPU-only command, from the repository root:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python benchmark/robot_closed_loop/diagnostics/diagnose_r1_5_pixel_rule_attribution.py \
  --cache-root benchmark/robot_closed_loop/results/r1_failure_chain_diagnosis_20261019a/native_hdr_cutoff059 \
  --output benchmark/robot_closed_loop/results/r1_5_pixel_rule_attribution_REPLAY_UNIQUE_ID
```

The research-side helper probe can be replayed with:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python \
  '/mnt/c/Users/zhaoyuhan/Documents/ChatGPT/New project/docs/r1_5_intermediate_review_2026-10-29/minimal_warp_probes.py'
```

- Run code commit: `eb41274c887c45874c212ee6317092add2600b46`.
- Diagnostic script SHA256: `a41661a7467d848095e644bde347bb0a2acaa392e063c0b6d01c5ce7048ea724`.
- Runtime module SHA256 (the same imported paths are recorded in `summary.json` → `runtime_modules`): `physics_plugin.py` `f01c6cb760153f09add80e0e18610a5710d07e0a7077e9552e0c7e7e009fb5cf`; `physics_interpolator.py` `3870f984b603c69c7dab2bc1a26a8d35218135af2ad139e50b115257777f09ef`; `physics_warp.py` `a7efbbf1e339084ae9c92e5252257a059aaeac4ff1f8269c1c245330bd51b3b7`; DVS `dvs_emulator.py` `914dc83525ff96dc2c1a1373819de74491cfbb03007b65e8c93609f6cd1cf638`.
- Inputs: episode 004 endpoint cache SHA256 `0525c18c8d89f5831ef0cb71ac9e3b6b130d60579daf374f06d5cfcee5b879d7`, frame evidence `b3afe5b5badee3d2606d343bb750aa970887928c41b8001a5fc4e9ff32b9c321`; episode 005 endpoint cache `91c1397642e317d6bb51150b4107c518b52e4fa9fcf5f1caf6a2a91315c7d434`, frame evidence `f161079a24c1bf9fee838a6fbb2a0fa5d785435b116f0ea5628e6efbba497236`.
- Runtime was WSL CPU, Python 3.14.6, NumPy 2.5.2, h5py 3.16.0. No remote connection, GPU, renderer, physics, or training was used.
- Research review files read: `REVIEW.md` SHA256 `0be4fe10879ef7ba7d7fb3ac9094677a15136db88d581802d384dfbfd302beb5`; `audit.json` `ffdd330f03a906aada64449309f614e21afea8fc53a7cb397679ec7d65beda18`; `minimal_warp_probes.py` `c6a943de86c02979ffb0b83e62e8fb9cbfa2c182ffbca4bd121615f42906553c`.

The research-side local audit differed by +3/-5 native events and did not retrieve the Direct-HDR HDF5 camera attributes. This report therefore claims exact replay only for the cached intermediate luminance arrays in the current WSL environment (max difference zero); it does not claim cross-environment elementwise native-event reproduction. Complete per-sample metrics, source contributors, per-pixel DVS references, controls, and hashes are in this folder's `summary.json`, CSV files, `minimal_warp_probes.stdout`, and `SHA256SUMS`.

## Decision

The cross-endpoint issue is confirmed as a helper-level behavior but its simple nearest-z fix is not acceptable. The tiny-weight depth winner is directly observed in both clips and the front-to-back coverage candidate passes the narrow static, full/partial occlusion, and subpixel controls while removing most large cached-image errors. This is sufficient to report the specific likely defect and a candidate algorithm, but not to switch the frozen core. Before switching, the candidate still needs review for NumPy/Torch parity, broader moving-surface/occlusion controls, radiometric boundaries, and the residual 665/1,377 anomalous pixel-times. The `next_primary` unknown/hold path remains a small residual possibility only; it is too sparse here to explain the main effect.
