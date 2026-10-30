# R1.5 CUDA parity forensic run C (low-level projection contract)

## Result

The run completed on g1 GPU1 (RTX 3090; UUID `GPU-0262d545-e9bd-7fcb-66e5-2ef81465c460`) in 3.327 s of diagnostic compute. It establishes that the low-level NumPy and CUDA projection/compositing APIs agree after the candidate projector uses float64 geometry and casts only at the public float32 `uv/z` boundary.

- The old 106 midpoint XOR support decisions (79 in episode 004, 27 in episode 005) now have **zero** native projection+compositor mask disagreement. Both episodes have maximum common-supported log-luminance difference `4.77e-7`.
- Identical projection arrays through NumPy/CUDA compositors yielded zero mask, surface, and depth-layer label disagreement. Coverage sum differences were at most `1.20e-7`.
- The two projections differed only by roundoff (`2.84e-14 px` maximum; depth identical); the strict 0.995 support threshold was unchanged.
- CPU/CUDA DVS threshold probes were exact (12/12). The four reconstruction/compositor × DVS comparisons per episode had equal counts, pixel/polarity histograms, and event slots; matched time error was at most 4.3 ns. Near-threshold, tiny-support, and front/back occlusion controls passed.

## Important scope limit discovered after this run

This diagnostic called the low-level Torch projector with float64 cache inputs. The public `PhysicsInterpolator` call site still converted poses, intrinsics, object poses, and alphas to float32 before calling that kernel. Therefore run C verifies the corrected kernel/API, but by itself does **not** verify the public production path. That call-site conversion has now been corrected in a new candidate commit; run D exercises the full public path on all adaptive intermediate samples and is the integration acceptance. Run C is retained unchanged as a valid low-level result, not as proof of full integration.

## Provenance and reproduction

- Executed code commit: `4523b30c4f714c44358bdc213ba106d1e6fd1f28`.
- Run/archive commit: `596b384` (result and logs on `exp/r1_5-core-candidate-20261030`).
- Config: `configs/r1_5_cuda_support_forensics_20261031c.json`; hashes and full per-point traces are in this directory's `SHA256SUMS`.
- Input cache hashes: ep004 `0525c18c8d89f5831ef0cb71ac9e3b6b130d60579daf374f06d5cfcee5b879d7`; ep005 `91c1397642e317d6bb51150b4107c518b52e4fa9fcf5f1caf6a2a91315c7d434`.
- Remote runtime evidence: `../runtime_logs/r1_5_cuda_support_forensics_20261031c/`.

From the candidate repository root on g1, with the archived cache available:

```bash
python -u benchmark/robot_closed_loop/diagnostics/diagnose_r1_5_cuda_support_forensics.py \
  --cache-root benchmark/robot_closed_loop/results/r1_failure_chain_diagnosis_20261019a/native_hdr_cutoff059 \
  --prior-arrays benchmark/robot_closed_loop/results/r1_5_core_cuda_parity_20261030c/cuda_parity_arrays.npz \
  --config benchmark/robot_closed_loop/configs/r1_5_cuda_support_forensics_20261031c.json \
  --output benchmark/robot_closed_loop/results/r1_5_cuda_support_forensics_20261031c-rerun
```

This result does not validate renderer fidelity, real-time sample availability, or full-core freeze readiness.
