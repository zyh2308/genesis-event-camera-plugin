# EventGenesis efficiency continuation: selected zbuf reuse candidate

This is an isolated engineering continuation of the archived index-reuse v1 candidate. The selected delta reuses the already-computed depth z-buffer for the first coverage-splat layer. Subsequent layers retain the original nearest-depth reduction. Candidate ordering, floating-point weighted reductions, sampling, complete noise/RNG, visibility, coverage and unknown-region rules are unchanged. No cross-frame cache is introduced.

## Selected and rejected sources

- `release/genesis_event_plugin/gpu_geometry.py` is the selected zbuf-reuse delta, byte-identical to `candidates/zbuf_reuse/genesis_event_plugin/gpu_geometry.py`.
- Selected geometry SHA256: `a5426aac911eec66c5552b99b4368cc7cc5fdde46557b64631122e1500c1f8e2`.
- Exact v1 reference geometry SHA256, read from its static build receipt: `e74793c4805eabfa2a19d9db02b5d829c73a12a61d3fed6b4cbeb8e5995a060b`.
- `candidates/zbuf_reuse/build_receipt.json` records the static scope checks: all other functions and the bounded-reference fallback are byte-identical to v1.
- `candidates/projection_storage/genesis_event_plugin/gpu_geometry.py` preserves the rejected first candidate, SHA256 `cc652fd35d200b16cf83c8c0e2d351df9d3841e0cda739a774b541984bef94d9`. It is not combined with the selected change.

The root `release/` is a delta, not a complete runnable package. Restore the complete archived v1 package using its versioned `sampler_cost_20261105/release/manifest.json`, then apply exactly this delta. The complete full-package manifest, artifact checksums, raw-artifact index and released GPU receipt are archived in evidence/. See [the first-delivery report](RESULTS_ZH.md).

## Measured pilot, as reported by the execution owner

The selected candidate was measured on the authorized g1 GPU2 in the registered deterministic configuration (`torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`). The warm cached-generation boundary includes initialization and generation to sorted CPU events; it excludes rendering, disk I/O and profiling.

| s2_00, 200ms, three trials per package | Archived v1 | Selected zbuf reuse |
|---|---:|---:|
| Median wall time, seconds | 34.41667269496247 | 33.875527661060914 |
| Maximum of three allocated peaks, GiB | 11.9453358655 | 11.6530499458 |
| Maximum of three reserved peaks, GiB | 13.994140625 | 13.505859375 |
| Events in each trial | 46,698 | 46,698 |

The median speed ratio is 1.015974512. This is a limited single-fragment deterministic pilot, not a five-scene performance aggregate or proof of universal equivalence. Profiling is excluded from these timings.

The projection-storage candidate failed selection. Its normal-mode s2_00 two-interval three-trial median was 1.849194 seconds versus v1 1.818727 seconds, with allocated peak 10.886 versus 10.883 GiB and reserved peak 12.314 versus 12.275 GiB. It was slower and did not reduce peak memory. These short normal-mode results use a different fragment length and numerical mode from the selected deterministic 200ms table; the initialized cached-generation boundary is the same. Do not divide values across the tables.

## Correctness status at this documentation update

- Selected zbuf candidate: all 35 registered component stress cases and five input classes with two physical intervals each passed the strict exact checks.
- The s3_00 20-interval/200ms continuous probe passed with 50,474 events; raw events, full state/RNG and intermediates were exact. Actual loaders passed on 6 flow, 20 pose and 100 reconstruction windows. No production freeze is implied.
- Default CUDA strict raw float64-event and formal-loader qualification remains failed as recorded in the handoff. No tolerance, timestamp quantization or consumer-output substitution changes that status.
- This change is not a frozen release or production upgrade. No existing bank, labels, checkpoint, frozen core, shared environment, formal training or real-test scoring is changed.
