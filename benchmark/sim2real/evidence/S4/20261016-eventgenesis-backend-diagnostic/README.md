# EventGenesis backend diagnostic evidence

One bounded synthetic 10 ms interval at 346×260, on g1 physical GPU 1 (RTX 3090). The locked 300 Hz / 0.05 numerical sampler target requested 377 intermediate samples and capped at 300. This run timed the existing Torch CUDA projection/warp stack and DVS integration and compared selected alpha values to the NumPy reference.

Observed: projection + warp 2.4154 s; DVS integration 0.5668 s; combined 2.9822 s; peak allocated/reserved CUDA memory 6.60/7.47 GiB. At indices 0, 150 and 299, max absolute projected-coordinate error was 3.052e-5 px, depth error 2.384e-7 m, and warped-radiance difference 4.247e-4. Projection and warp validity masks had zero mismatches. Diagnostic exit code was 0 and stderr was empty.

The JSON is the machine-readable source of the measurements; stdout duplicates it. The deterministic synthetic fixture is constructed by the committed diagnostic script. This evidence is not a Genesis rollout, training bank, model training, or real-data evaluation.

Original g1 run directory:

`/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261016-stage4-flowdt1-seed0-v3/diagnostics/`
