# R1.5 block-localization recovery diagnosis (2026-10-19)

## Current conclusion

The exact-endpoint projection defect is reproduced in both NumPy and CPU Torch and has a narrowly correct diagnostic fix: at a mathematically exact source endpoint, return the source pixel grid and depth directly. This removes the endpoint image seam without rounding any intermediate motion. The matching core change is preserved as a review patch only; `genesis_event_plugin/` remains unchanged.

This defect does not explain the block-localization failures by itself. At the tested 0.5905 s endpoint, the block pixels already matched the direct render; the large residuals were on Panda/background pixels. With an exact-state direct-render event stream, the existing native-HDR Ours checkpoint still has large localization error, so event distribution/checkpoint compatibility remains an independent failure. Stable block localization has not been recovered.

## Evidence and measurements

### Endpoint projection

Episodes 004 (center) and 005 (+55 mm) were replayed from the same immutable trace. The trace has 59 actual rows over 50 ms, with 0.5–1.0 ms intervals (median 1.0 ms, effective mean 1160 Hz). No synthetic states or new physics rollout were used.

At 0.5905 s, the frozen NumPy source-1 projection differed from identity by at most `1.526e-5 px` / `6.104e-5 px`. The visible image had `348` / `160` pixels with absolute log-luminance error above `0.2`; the exact-identity candidate had zero error across all 76,800 pixels. In the full trace replay, every exact endpoint at 0.5605–0.6005 s also returned its source image exactly. At 0.5905 s, the target mask itself had zero pixels above the error threshold in both episodes; the residual was on robot and background pixels. This confirms a real renderer-warp endpoint inconsistency, but does not establish that it caused the target-position errors.

The Torch CPU path independently reproduced the mechanism at 0.5905 s: maximum projection error `7.629e-5 px` in both episodes; raw projected images had `641` / `990` visible pixels above log error `0.2`, while the identity candidate had zero. Exact endpoint handling changed no midpoint motion: at alpha 0.5 the raw/candidate image delta was zero and projected object motion remained about `0.477 px`.

The review patch [`r1_5_exact_endpoint_identity_core_proposal.patch`](../diagnostics/r1_5_exact_endpoint_identity_core_proposal.patch) passes `git apply --check`. It adds only exact `alpha == 0/1` identity branches to the NumPy and Torch projection functions. It has not been applied to the frozen core.

### Interval brightness and occlusion handling

The synthetic static/ramp controls remain the cleanest timing check: a static image produces no events; a 10 ms smooth log-luminance ramp from 0 to 0.5 places threshold events at 4 and 8 ms, rather than packing the full change into the first 1 ms subframe.

In the two real cached episodes, the conservative log-radiance candidate reduced mean target-region log error across trace rows from `0.00688` to `0.00098` (center) and `0.00734` to `0.00191` (+55 mm). Target pixels above `0.2` log error fell from `1041` to `100` and `616` to `76`, respectively. This candidate used exact same-state direct-render fills for uncertain pixels: 53 of 59 rows per episode requested at least one fill, covering about 9.3–9.4% of summed image pixels. A full frame render at each request was estimated to add `1.77 s` / `1.61 s` per 50 ms clip. These are diagnostic direct-render fills, not a proven low-cost reconstruction path.

### Event stream and learned localization

The exact-state direct HDR renders were regenerated for every actual trace row. At the six saved source timestamps per episode, HDR, finite-depth mask, segmentation, and pose matched the existing source cache exactly. Exact direct events from the nonuniform ~1.16 kHz rows had five-bin cosine `0.920` / `0.908` against the archived 2 kHz Direct-HDR events, but event counts were lower: `3539` vs `4505` (center) and `5739` vs `7575` (+55 mm) over the 50 ms interval. This is a sampling-density difference, not a same-rate convergence result.

The two old checkpoints were read at the same 21 query phases from 0.585–0.595 s. Each row uses only a causal five-bin event tensor; GT was joined after inference for scoring.

| Episode / input events | Ours state coverage | Ours XY P95 | V2E state coverage | V2E XY P95 |
|---|---:|---:|---:|---:|
| Center / archived Direct-HDR 2 kHz | 21/21 | 199.0 mm | 21/21 | 7.2 mm |
| +55 mm / archived Direct-HDR 2 kHz | 21/21 | 137.5 mm | 21/21 | 6.6 mm |
| Center / exact-state direct events (~1.16 kHz) | 21/21 | 127.8 mm | 21/21 | 10.6 mm |
| +55 mm / exact-state direct events (~1.16 kHz) | 21/21 | 167.1 mm | 21/21 | 12.6 mm |
| Center / endpoint-corrected candidate | 15/21 | 100.2 mm | 15/21 | 100.5 mm |
| +55 mm / endpoint-corrected candidate | 21/21 | 28.2 mm | 21/21 | 36.1 mm |
| Center / log-radiance with exact render fills | 21/21 | 68.5 mm | 21/21 | 369.1 mm |
| +55 mm / log-radiance with exact render fills | 21/21 | 17.6 mm | 21/21 | 5.6 mm |

The endpoint candidate loses six valid center states, and the log-radiance candidate's performance varies strongly by episode and model. Neither is stable evidence of recovered localization. The direct-render event stream remains easy for the V2E checkpoint and difficult for Ours, which points to a learned input-distribution mismatch in addition to the endpoint bug. The two episodes are development diagnostics, not an independent test or generator ranking.

## Cost, provenance, and artifacts

The full exact-state replay ran on g1 in the existing worktree `genesis-event-camera-plugin-r1-cadence-sweep-20261019d`, code commit `2690935cac25a943148201bb66c31af63d65fd23`. It used the CPU Genesis physics backend and CPU event/model computation; the headless renderer used idle GPU 0. It did not advance physics. Total Python wall time was `197.931 s`; the 118 exact-state render calls took `3.763 s` (`1.974 s` center, `1.789 s` +55 mm), and direct event emulation took `1.433 s`. The per-state render cost was about 30–33 ms in this run. Genesis reported a shared Quadrants cache-lock warning; no cache cleanup or other-user process changes were made.

- Scene config SHA256: `aa97f6f5a12af92561ef053407bb795bba2801582958b4ad1134bcd24497435a`.
- Control config SHA256: `3d289c2bef1c562b26f9b713c2a826b813a286a1f7e21f9ead0a01c63c5e0d80`.
- Candidate config SHA256: `13efc23498f82c9e0f27539c06abc4554b831a9173ca5e8d5fc2bd603a326226`.
- Reference manifest SHA256: `86cf6c0801365c0c1e5949e4e04d3178fa03f334997db6d09547f69a2b5af37c`.
- Source-cache manifest SHA256: `f143910056dbf407fb7a2808412f5dc9c9bb8554ea37a4a54f7d5449b4628291`.
- Ours/V2E checkpoint SHA256: `6ecea6a4520ae09b8cc99ebfd8da7a6a324363d2e57dbf73790e27ee2dbec8d2` / `a7140b63f3b5042b0dc0ed7cc3aa2974c4e9859db919f318826358883f7ce8a0`.
- Real-render result folder: [`r1_5_consistency_candidates_20261019b_realrender_g1`](r1_5_consistency_candidates_20261019b_realrender_g1/), `SHA256SUMS` file hash `49764b3f4e70b406c34d1c41d46ea344c13b9b9cb8d2ca85fd4404563c69ae15`.
- Earlier CPU-only cached replay: [`r1_5_consistency_candidates_20261019a_cached_cpu_g1`](r1_5_consistency_candidates_20261019a_cached_cpu_g1/), manifest hash `925be472660445564648fd418db06f8c3e50f35e7ef0a55d7b44fdeb7e401e83`.
- Torch endpoint replay: [`r1_5_endpoint_torch_candidate_20261019a_g1`](r1_5_endpoint_torch_candidate_20261019a_g1/), manifest hash `e6cf84082f1fd978527ab2d86d5d312097ac5473972f3e409532e2f2eea25123`.
- Runtime log SHA256: `6e61ffa166a25ca07a99118143a2c0c3a551bf2b17852642777dc045eaa8d786`; exit file content `0`; launcher [`r1_5_consistency_realrender_20261019b.launch.sh`](runtime_logs/r1_5_consistency_realrender_20261019b.launch.sh).

### Cadence dependence and endpoint-only event isolation

The 200/500/1000 Hz sweep used only the exact logged states selected from the same `.5505–.5905 s` trace window. At 1.15 kHz effective source cadence, the direct-render stream had 2,807 events in the center episode and 4,618 in +55 mm. The original warp/reconstruction path produced the following counts over the same 40 ms window:

| Requested cadence | Center | +55 mm |
|---:|---:|---:|
| 100 Hz existing endpoint path | 23,690 | 24,776 |
| 200 Hz exact rows | 53,177 | 45,090 |
| 500 Hz exact rows | 133,592 | 111,557 |
| 1000 Hz exact rows | 259,960 | 218,197 |
| Exact-state direct-render input, actual rows | 2,807 | 4,618 |

This is a large generation-path discrepancy. More frequent exact source frames do not make the warped events converge to the direct-render stream: 5-bin cosine against the archived 2 kHz reference was `.182/.260` at 200 Hz, `.138/.226` at 500 Hz, and `.127/.220` at 1000 Hz (center/+55). The direct-render stream scored `.920/.908`. At the `.5905 s` query, the old Ours checkpoint happened to be accurate on the plugin streams for these two episodes, while V2E degraded sharply at 500/1000 Hz. This is a single-cutoff development diagnostic, not evidence of stable perception.

To isolate the endpoint contribution, commit `2d782fbd75c7dc3d64b39e6af32364caa5e2f795` adds a diagnostic-only wrapper that replaces only each interval's final `alpha=1` reconstructed luminance frame with the exact endpoint luminance. It does not change any intermediate warp, round motion, touch core code, or advance physics. At 200/500/1000 Hz, endpoint replacement reduced event counts by `17.5–30.1%`, but still left `7–76×` as many events as direct-render input, depending on episode and cadence:

| Cadence | Center: original → exact terminal | +55 mm: original → exact terminal |
|---:|---:|---:|
| 200 Hz | 53,177 → 41,034 | 45,090 → 32,391 |
| 500 Hz | 133,592 → 102,250 | 111,557 → 77,978 |
| 1000 Hz | 259,960 → 214,501 | 218,197 → 154,442 |

The endpoint-only candidate did not consistently improve the Ours checkpoint at `.5905 s`: center XY errors were `6.0/6.4/4.8 mm` on the original streams and `4.9/4.9/5.4 mm` with the exact terminal candidate; +55 mm errors were `.6/3.6/1.9 mm` and `2.7/6.7/5.4 mm`. V2E errors also changed sharply with rate and source. Therefore the endpoint defect is real and contributes to event inflation, but does not account for the main mismatch or explain all localization failures.

The separate brightness control remains a second confirmed limitation: `next_primary` applies the interval's endpoint luminance at the first generated child frame, moving 4/8 ms ramp events to 0.4/0.8 ms. A log-radiance candidate with actual-render fills reduced target-region image residuals and improved some replay queries, but needed at least one direct-render fill on 53/59 state rows per episode. It is not yet an affordable or validated event-generation path. Current evidence therefore localizes the unresolved real-trajectory discrepancy to the intermediate reconstruction/photometric path as a whole; it does not yet identify one unique failing operation.

The endpoint cadence isolation run completed on g1 with status `PASS`, code commit `2d782fbd75c7dc3d64b39e6af32364caa5e2f795`, config SHA256 `8fdd5e668ae6c2316ac9ad4449511c1f9fef14d7c1ce28b2a98874931f951f3b`, and total Python wall time `289.005 s`. It rendered 47 exact rows per episode (about 1.51/1.55 s per episode) and ran event generation/model inference on CPU; GPU 0 was used for headless rendering. No other GPU or process was touched.

- Endpoint-isolation results: [`r1_5_endpoint_terminal_cadence_isolation_20261019e`](r1_5_endpoint_terminal_cadence_isolation_20261019e/); output `SHA256SUMS` file hash `cca6858a08ff3ed1ee1c0d48c8e50a81c6e323c7e98e6394969256024b0cfdaf`.
- Endpoint-isolation runtime log SHA256 `6b64a389e6736f00ac4d4d14eccac6965bf42762535ef206a947283ba020cc34`; exit file SHA256 `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa` contains `0`; launcher [`r1_5_endpoint_terminal_cadence_isolation_20261019e.launch.sh`](runtime_logs/r1_5_endpoint_terminal_cadence_isolation_20261019e.launch.sh) SHA256 `4e09c8dd10879c99f3cc8063561f86271f67fbc455fc7739e9ed29fc6b44f5a1`.
- The completed 200/500/1000 Hz sweep is [`r1_ours_localization_cadence_sweep_20261019d`](r1_ours_localization_cadence_sweep_20261019d/), manifest SHA256 `010ef8dda4351b4b6f10ead6f4e8d3dd3b0edd837ef9881a24c73e63c144959b`.
- Exact-state real-render results, CPU cached replay, Torch endpoint verification and the minimal core proposal remain linked above. The proposal patch SHA256 is `69db33d165e2b3965e597836049bdf959d5837bac4e906a937e9f12f23079536` and passes `git apply --check`; no frozen-core file was changed.

## Next bounded step

Do not switch the frozen core or start another training round yet. The cadence and endpoint-isolation runs show the reconstruction path itself changes event volume far more than endpoint projection alone can explain. The endpoint NumPy/Torch patch is a generally correct, narrow fix and is ready for research review, but it should not be presented as a full localization repair. The known early-luminance behavior also needs its own isolated diagnostic candidate on these episodes.

The next bounded technical step is a diagnostic-only temporal-contrast candidate: keep the same geometry and visible-surface rules, distribute endpoint log-radiance change over the interval rather than applying it at the first child, and request actual renderer frames wherever correspondence/visibility is uncertain. On the same two cached episodes, compare image residual, per-child and total events, source-region event counts, 5-bin tensors, and all 21 query phases. First require static no-event, exact endpoint identity, smooth-ramp event timing, and preserved intermediate motion. If this candidate passes those controls and materially reduces the event-count/cosine gap without an impractical renderer cost, prepare a review package containing the minimal NumPy/Torch core diff and legacy-artifact invalidation scope. Only after that review should a single matched small retrain be planned, evaluated on at least one new development episode not used to tune the candidate.

Stable perception remains unproven: the exact-render stream still gives the existing Ours checkpoint large position errors over the 21 query phases, and endpoint-only event correction is not a reliable localization improvement. No R2 rollout or formal Ours/V2E ranking was started.
