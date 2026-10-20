# Multi-reference geometry/cache controls (local CPU)

- Code commit: `4a4e51f717b9f177223ab99598e42571fc632f64`.
- All seven controls pass with Python warnings treated as errors.
- In addition to the geometry, visibility, ramp, and unknown-pixel controls, the suite verifies that precomputed moving-source rays and triangle topology produce bitwise-identical projection/raster outputs to uncached execution.
- This is synthetic NumPy CPU evidence only. It is not the g1 cached-episode result and does not establish end-to-end speed.
- Output: [`controls.json`](controls.json), SHA256 `7d3d7d4f15ce2fbfadc8d1c4f1e2762071f13d9eace8e89cfc79229ff297faf3`.
- Reproduce: `python -m benchmark.robot_closed_loop.diagnostics.run_r1_5_multiref_surface_controls` from the repository root.
