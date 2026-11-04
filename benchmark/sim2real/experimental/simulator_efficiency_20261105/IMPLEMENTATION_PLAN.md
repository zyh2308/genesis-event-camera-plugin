# Simulator Efficiency Implementation Plan

**Goal:** Quantify safe reductions in the cost of the frozen complete Stage 9 sensor profile on the explicitly assigned g1 GPU 2.

**Architecture:** Keep the 17-file frozen core and all formal campaign inputs/queues immutable. Use a private addon and benchmark directory to compare the original implementation, bulk transfer, and a fused sensor-state candidate while preserving all 377 time samples, FP64 state, actual noise draws, refractory behavior, unknown masks, event multiplicity and endpoint geometry.

**Tech Stack:** Existing Torch 2.8 / CUDA environment; private C++/CUDA extension build using the existing isolated compiler toolchain; h5py/NumPy; same-GPU synchronized repeated benchmarks.

## Steps

1. Verify the released GPU 2 reservation, acquire its UUID lock and register a bounded private test lease. Keep g1 GPU 3/4/5 and g3 training unchanged.
2. Load fixed synthetic training endpoints before timing. Run frozen baseline with one warmup and repeated fresh-state trials; record setup separately, total steady-state latency, interpolation and DVS wall phases, actual sample count, memory, raw events and all tensor states/RNG state.
3. Implement bulk preprocessing/upload without changing sensor equations; compare raw events, masks, intermediate-frame hashes, state and RNG.
4. Implement a CUDA pixel-parallel, time-sequential full current-profile sensor path. Pre-draw random tensors through the original Torch generator in the same call order. Preserve FP64 and separate multiply/add ordering. Guard unsupported settings, zero-slope fallback timing and output capacity; never silently drop events.
5. Test known ramps, static scenes, multi-event/refractory behavior, noise stress, unsupported options, chunk boundaries and actual cached rich/pose inputs. Separate exact serialization, canonical event set, timestamp/state error and reference-repeat variability.
6. Benchmark paired candidates on all fixed cases, alternating execution order and reporting medians plus distributions. Keep data loading/setup separate; report full capture, not only the faster kernel. Retain failures and any fallback cases.
7. Run fixed-model localization/representation regression where existing acceptance data is available. No real-test tuning or new training. Candidates that fail gates remain diagnostic-only.
8. Archive addon code, raw logs, hashes and report in the shared repository. Leave formal experiments on the frozen implementation; release the temporary test lease after this bounded round.

No reduction in sample count, precision, resolution, noise or event count is part of this first round. Multi-rate geometric approximation remains a separate later experiment.
