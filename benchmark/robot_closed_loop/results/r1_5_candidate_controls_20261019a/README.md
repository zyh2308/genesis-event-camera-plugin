# R1.5 candidate analytic controls

This small control run exercises the diagnostic candidates only. It is not the episode 004/005 cached-trajectory comparison, a perception result, or a reason to switch the frozen simulator core.

## Results

- Static radiance remains unchanged: maximum absolute image error `0`; the ideal pixel model emits `0` events.
- A known log-radiance ramp from `0` to `0.5` over `10 ms` produces the expected intermediate values (`0.05` at the first sample and `0.50` at the last). The candidate event times are `4.000/8.000 ms`, matching direct smooth-frame input (`4.000/8.000 ms`); both produce 512 events.
- The synthetic exact-identity endpoint projection and exact endpoint return each have maximum image error `0`.
- Intermediate synthetic motion remains continuous: maximum projected displacement is `2.156 px`; the candidate intermediate projection is unchanged from the frozen projection (`0 px` difference). No intermediate coordinate rounding is used.

These controls show the candidate code can satisfy the static, endpoint, and known smooth-ramp cases. They do not demonstrate that log-linear interpolation matches real shadowing, occlusion, or the cached robot images. The cache comparison remains pending on g1.

## Provenance

- Code commit: `c427e574922aa5c9676151c3404437a74dce9f75`.
- Config SHA256: `13efc23498f82c9e0f27539c06abc4554b831a9173ca5e8d5fc2bd603a326226`.
- Candidate module SHA256: `98ba537df324133fb9b93faba8f5f83a42f74161aab88fb6dfa1e4e26996a5cc`.
- Cached-trajectory runner SHA256: `c12b65a24c7af00fc048658679a046a6ff6933a32c2daee146144965d9ccd8a5`.
- Analytic control output SHA256: `d4b7df4333a2e76d3076a82fcbd7fe3cfdf86c5db5b37eec67403afceef590e2`.
- Frozen core was not modified. The NumPy synthetic path was exercised; no Torch/CUDA numerical-path comparison was run.

Reproduce from the repository root:

```bash
python benchmark/robot_closed_loop/diagnostics/r1_5_consistency_candidates.py --analytic-control
```
