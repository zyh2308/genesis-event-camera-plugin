# Actual receipt analysis, read-only

`summarize_actual_results.py` uses Python standard library only. It reads externally SHA-bound actual smoke/study completion JSON and writes a fresh output directory; it never imports a training script, uses SSH/GPU, or modifies frozen sources/results. Five CPU fixtures exercise full48-row reporting,36+12 independent gate recomputation, missing-seed/NaN/source/smoke rejection, one-condition regression and frozen-depth mismatch rejection.

Minimal proof checked: complete2-update paired smoke; six full288 tape admissions; RAW/mask cache and source/science/resource hashes; first native scalar oracles and independent model+auxiliary+Adam readback; exclusive GPU3 start/end proof; all18x64 original C1 numerical/full-pose baseline checks; study six288=1728 updates from source-init; unchanged teacher/size and real/GT-input isolation; all6 runs×4 condition×2 arms with native seven metrics and full window inventories; independent scalar recalculation of both registered gate groups. These are receipt consistency checks. External server checkpoint/input/log bytes are not silently claimed independently rehashed on a Windows download.

Outputs include per-run/condition/arm values and mask auxiliary diagnostics, native per-rollout CSV, three-seed means/ranges, each method's delta from its own RAW C1, and each condition's EG/V2E metric comparisons. No seed or illumination selection, p-values, or global winner. Clean384 spans96 trajectories; matched-normal/half/eighth64 span fixed4. Formal30+8 are still not retested/passed.

Current locked training producer does not save the candidate's per-window attention maps or wrong-target-cell categories. The summary explicitly marks these unavailable; improved mask IoU or center mean is not proof that the28 wrong peaks disappeared. A separate zero-update candidate export is needed for that claim.

Example (smoke alone is allowed and produces no invented study metrics):

`python -B summarize_actual_results.py --smoke actual_smoke_completion.json --smoke-sha256 ACTUAL_SHA --study actual_study_completion.json --study-sha256 ACTUAL_SHA --out NEW_DIR`

## Read-only resource/performance observations

The raw training tensors occupy roughly1.50GB GPU plus0.30GB mask128 labels; networks/optimizer are small. Input preparation converts one416 episode at a time to128, so the main risk is not32GB GPU capacity. A normal416 batch32 temporary is about66MB. Each method writes approximately2GB RAW+mask cache. Repeated HDF521-frame target/count checks, file SHA reads and per-window mask resizing dominate preparation, followed by six288 replay admissions and18x64 original full-pose parity checks. No epoch line is printed until these complete, so several quiet minutes are plausible.

Host RAM holds several copies while concatenating normal/auxiliary/validation tensors and endpoint regions; a few GB above cache size is plausible, not measured. In study preparation, the local `cached` variable can retain a prior loaded CPU cache until replacement/function exit, adding roughly one cache's size. This is an efficiency limitation, not a demonstrated OOM. Do not modify or restart the running locked trial for this observation. These are shape-derived estimates; no current server RSS/VRAM or timing was measured by this child.

## Separate zero-update candidate peak exporter

`export_candidate_peaks.py` runs only after the complete actual study receipt passes the same consistency admission. It uses the same strict GPU3 guard and frozen E context, SHA-bound prepared RAW128 caches, original C1 control models, and the completed candidate's bare CenterScale checkpoints. It forwards both models unchanged, compares every control UV against the original reference within2e-6, and records all1,152 paired cases (2 methods×3seeds×3lights×64windows). Original FP32 probability maps and same-order index are retained for control/candidate. No auxiliary head, filtered events, decoder adjustment, teacher input, optimizer or real data is involved.

Visible physical endpoint masks, endpoint union and their fractional16-cell occupancy are only used after predictions. Per-row native peak coordinate/cell, point membership, cell overlap, category and original-UV sanity are reported. Empty/unavailable regions are not proof of background events; endpoint union remains approximate. Four additional CPU fixtures check native probability expectation, unaltered maps, first-index ties, cell-vs-point-vs-union distinctions and invalid/NaN rejection. The exporter never changes the training gate or selects a winner.

`python -B export_candidate_peaks.py --study N/aux_mask_gpu3_v2_study/completion.json --study-sha256 ACTUAL_STUDY_SHA --contract N/MASK_AUX_CONTRACT.json --contract-sha256 d2860889e582223bf248d7b8e8e6fcc8f80aa71041a371243fdae06010e312ff --resource-contract N/MASK_AUX_RUNTIME_RESOURCE_V2.json --resource-contract-sha256 c1dc975cd625a5b865e184368aacd576d09b07ca82937079f7468c1af37cfacf --v2-sources ACTUAL_LOCKED_V2_SOURCES --analysis-source-manifest-sha256 ACTUAL_ANALYSIS_MANIFEST_SHA --out NEW_PEAK_DIR`

Run only through the existing authorized GPU3 queue after study completes. This child has not submitted or executed it. Both actual native-model performance and any wrong-peak reduction remain unknown until server execution. The source/input/model files actually read by this exporter are rehashed at completion; unused training ancestors are retained via the study receipt and not falsely claimed re-read.
