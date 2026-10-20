# R1.5 multi-reference surface controls (local CPU only)

- Candidate code commit: `1605c6e54aea72473e222d0f01d2bd6a26f00e82`.
- This small synthetic control suite validates SE(3) projection equivalence, complementary static surface use, both depth orders, equal-depth fail-closed behavior, log-radiance interpolation, and truly unknown pixels.
- It does not load Genesis, render the robot scene, or validate cached episodes. The NumPy fallback was used.
- Reproduce from the repository root with:

  ```bash
  python - <<'PY'
  import json
  from pathlib import Path
  from benchmark.robot_closed_loop.diagnostics.run_r1_5_multiref_surface_controls import run_controls
  print(json.dumps(run_controls(), indent=2))
  PY
  ```

- Structured output: [`controls.json`](controls.json), SHA256 `c967926022c884027108e70131a6c9f4e3d6c4836cc75f1ecf309d732925beb2`.
- No simulator core, task code, renderer, or shared server process was changed by this local control run.
