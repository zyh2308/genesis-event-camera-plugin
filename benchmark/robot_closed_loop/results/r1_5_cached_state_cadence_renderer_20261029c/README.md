# R1.5 cadence / MSAA replay and old-checkpoint localization

## Question

On the same exact robot/target trajectory, do render cadence and the GL
multisample switch change the Direct-HDR event stream enough to affect the old
Ours checkpoint's square localization?

## Result

The source replay reproduces the archived physical trace and full raw event
file exactly. The renderer-only replay then uses the same 81 true physical
states from `0.5505` through `0.5905 s`; it advances no physics. All three
generated streams are evaluated with the same old Ours checkpoint at the same
seven causal cutoffs (`0.5855` through `0.5905 s`). The checkpoint is diagnostic
only; no model was trained or selected here.

| Cached render condition | Events | Archive events matched within 0.5 ms | XY error mean / p95 | Velocity error mean / p95 |
|---|---:|---:|---:|---:|
| MSAA on, sparse 47-frame cadence | 4,618 | 3,463 / 6,182 (56.0%) | 10.64 / 17.19 mm | 37.37 / 60.37 mm/s |
| MSAA on, dense 2 kHz cadence | 5,899 | 4,627 / 6,182 (74.8%) | 12.08 / 21.29 mm | 22.15 / 45.70 mm/s |
| MSAA off, dense 2 kHz cadence | 7,123 | 3,338 / 6,182 (54.0%) | 15.71 / 21.13 mm | 22.52 / 35.51 mm/s |
| Original archived online stream | 6,182 | reference | 58.69 / 108.33 mm | 16.55 / 33.48 mm/s |

State availability was 7/7 for every source. These seven adjacent queries are
one diagnostic clip, not seven independent episodes.

This narrows the failure: changing the cached renderer's event input moves the
same checkpoint from a 58.7 mm average position error to 10.6–15.7 mm. Within
the cached path, denser sampling improves event agreement with the archive and
reduces velocity error, but slightly worsens the mean position error. Turning
off GL multisampling changes the event count and worsens localization. Neither
factor alone explains or removes the archive/re-render gap. The cached 47-frame
stream agrees with the previously saved E stream to within floating-point
roundoff (maximum event-value difference `1.11e-16`), so this is reproducible
within the cached path.

The next check is to capture the HDR frames already produced by the original
online event source during another exact-seed replay, without extra render or
physics calls, and compare them pixel-by-pixel to these cached-state frames.
That will separate image/render differences from DVS initialization/history
and cadence effects.

## Cost and renderer audit

The three cached conditions took `168.31 s` total wall time, including one
Genesis scene build per condition. Per condition, scene build was `47.90–52.74
s`, rendering `0.97–1.55 s`, and DVS processing `0.67–1.66 s`. The GL audit
recorded 4-sample Genesis HDR framebuffers. MSAA-off means
`GL_MULTISAMPLE=false` while the existing 4-sample FBO remains allocated; it is
not a separately allocated single-sample framebuffer. The on/off segmentation
was identical. Their log-luma difference had mean `0.00658`, p95 `0.00335`,
and `0.77%` of pixels above `0.2` absolute log difference.

These are diagnostic costs with repeated scene initialization, not a steady
state production-rate estimate. The frozen simulator core was not modified.

## Reproduction and provenance

- Branch: `exp/robot-closed-loop-v1`.
- Cached renderer commit: `855b4b0bc7de96170b4bfee8d7002b370782f346`.
- Original R0.5 source runner commit: `2100645b0473c294760789bc4475b644f261d9cf`.
- Cadence/MSAA config SHA256: `767f7396486ee98589b751cde53cb58c5d14a7170be022aedb0ad0785958ca94`.
- Renderer diagnostic SHA256: `a4226abf6ef5032890fd5565c22e46f12a887749013faabfff9e5998ad687efa`.
- Scene config SHA256: `aa97f6f5a12af92561ef053407bb795bba2801582958b4ad1134bcd24497435a`.
- Old checkpoint SHA256: `d391fec293e2c6c50cdb267e15f742ed09e77e7b28e56a79fdf700f633e804e9`.
- Original full state trace SHA256: `310837ea07ab2f819f590ccce91b3952dd9e85ea5005537bc1f4d1a339bff28c`.
- Original raw event file SHA256: `b698817a8347ca48ad75015f0cb81c3bfc711958a60d3387d468c70456a7b2b6`.
- Dense 81-state cache SHA256: `b87efd4d7b5bfda94b0987f0d682671a4230246c5f9886eb086600542bc075bb`.
- E sparse event array source file SHA256: `c46f7bc28504ed9f43e76c334533534291d0de64b0c0f9bd429fa1a49eab58d8`.
- Exact-query localization analysis code was committed in `bac8463fea68dd0ae78b193190f2407e9f3ed85f`; subsequent launcher commit is `3653e3a3dfa63ab956f5259cb99059182b7353af`.

Run the same local diagnostic for each cached stream:

```bash
BASE=benchmark/robot_closed_loop/results/r1_5_cached_state_cadence_renderer_20261029c
EVIDENCE=benchmark/robot_closed_loop/results/r1_5_reference_reproduction_local_20261029a/remote_metadata/compact_replay_evidence_v3.json
SCENE=benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json
CHECKPOINT=benchmark/robot_closed_loop/results/r1_shared_ideal_small9_20261018b/paired_r1/ours_fit/spatial_event_state.pt
for CONDITION in msaa_on_sparse47 msaa_on_dense2khz msaa_off_dense2khz; do
  python benchmark/robot_closed_loop/diagnostics/analyze_r1_5_seed_replay_localization.py \
    --compact-evidence "$EVIDENCE" \
    --fresh-shadow-events "$BASE/remote_data/artifacts/events_${CONDITION}.npz" \
    --fresh-source-label "$CONDITION" --scene-config "$SCENE" \
    --checkpoint "$CHECKPOINT" --output "$BASE/remote_data/localization_$CONDITION"
done
```

Remote renderer output was produced on g1 GPU0, RTX 3090 UUID
`GPU-fd9f1bd4-83c4-cd74-0c88-241db6ec7e23`, with no physics advancement. Raw
HDR frames, generated event arrays, renderer audit summary, and the three
per-query checkpoint outputs are under `remote_data/`; hashes are listed in
`SHA256SUMS`.

## Limits

This is one +55 mm trajectory, one old checkpoint, and seven nearby cutoffs.
It does not prove which renderer pass or DVS history field causes the mismatch,
does not establish a general model capability, and is not a retraining,
closed-loop, or formal-ranking result.
