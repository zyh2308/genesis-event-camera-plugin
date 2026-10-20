# R1.5 multi-reference cached replay (g1)

This is the first complete cached-clip result for the selective multi-reference candidate. It reuses five existing 100 Hz Direct-HDR source views per 40 ms interval, reprojects moving surfaces, reuses observed fixed-camera background pixels, and keeps a full-frame Direct-HDR fallback for any unsupported or ambiguous pixel. No physics rows were synthesized and no new rollout was run.

## Provenance

- Executed code commit: `378442a45afb85e494acbc53f95859028ddd9163`.
- Candidate configuration used: [`candidate_config_used.json`](candidate_config_used.json), SHA256 `db23df18bc103cffc0f8bddec883c0a40eba6cc56506c9d980c04b09d45e1b55`.
- The raw run manifest is [`SHA256SUMS`](SHA256SUMS), SHA256 `4249a35be49eb2e54ea8172b779317edeaae7f8f825c64978dbc105f325dcb64`.
- The archive-level manifest, including this README and the used configuration snapshot, is [`ARCHIVE_SHA256SUMS`](ARCHIVE_SHA256SUMS); its hash is recorded in `robot_status.md`.
- The input cache is the existing, read-only `r1_failure_chain_diagnosis_20261019a/native_hdr_cutoff059`; per-file trace and source-cache hashes are in [`summary.json`](summary.json).
- Ours/V2E checkpoints are old development weights. Their hashes are recorded in `summary.json`; inference is diagnostic only.
- Frozen simulator core is unchanged. Results are not ranking, a new physical rollout, or an independent test.

## Result

| Episode | Mean requested pixels | Full-frame fallbacks | Candidate total | Exact Direct-HDR + DVS |
|---|---:|---:|---:|---:|
| 004 center | 0.2473% | 42/42 | 7.102 s | 1.220 s |
| 005 +55 mm | 0.3105% | 42/42 | 6.107 s | 1.713 s |

The candidate reduces unsupported/ambiguous area from the earlier 14.5–14.9% to about 0.25–0.31%, while the tested per-pixel depth ordering is preserved. Any remaining requested pixel still triggers a whole-frame fallback because the current renderer has no partial-frame API. As a result, all intermediate frames were direct rendered; raw candidate and exact Direct-HDR event arrays are equal due to that fallback. This is evidence of much better surface coverage, not yet a useful or correct event reconstruction path. At the measured costs the candidate remains 3.6–5.8 times slower than direct rendering.

The diagnostic montage marks render-request pixels in yellow on the left; the right image is the final full direct-rendered frame used after fallback. Pre-fallback supported-pixel errors and support fractions are in each `frame_metrics.jsonl`; raw event streams, query tensors, and per-episode summaries are included. The supervised g1 command and PID/log/exit evidence are in [`runtime_logs`](../runtime_logs/r1_5_multiref_surface_cached_replay_20261021d/README.md).

## Reproduction

On g1, in the isolated worktree at the executed code commit:

```bash
cd /home/zhaoyuhan/Eventbased_WAM/repositories/genesis-event-camera-plugin-r1_5_multiref_20261021c
bash benchmark/robot_closed_loop/diagnostics/launch_r1_5_multiref_surface_cached_replay_20261021d.sh
```

The launcher refuses existing result/log paths, checks both episodes and checkpoints, starts a user tmux session, pins the Python environment, hides CUDA, and records the PID and exit code. The command runs episode 004 and 005 from the same previously archived trace/cache.
