# Selected-three HQF development evaluation

This isolated evaluator does **not** launch training, connect to a server, acquire a GPU,
authorize itself, or score anything during import/prepare/verify.

The selected sequences are `boxes`, `reflective_materials`, `still_life`. They were selected
after earlier real results. All outputs are labelled **posthoc selected HQF3 development**;
they are neither HQF14 results nor an independent blind test.

## Required training receipt

`F=/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261108-recon-selected-full-v1`

The evaluator requires the new training package's complete `frozen_checkpoint_matrix.json`
and `completion.json` plus every method × seed completion. It independently checks all
180,000 update-log rows, all 600 synthetic validations, the synthetic-only minimum and tie
rule, selected CP byte and tensor hashes, exact recipe/trainer, original per-seed initial
states, shared whole-rollout 24/8 split, final 12,000-update CP bytes, final synthetic
stability and restored-boundary receipts. Selected CPs may be earlier than update12,000;
every training run must finish the same full budget before any real prediction begins.

The exact original HQF14 data manifest is hash-bound. The selected manifest copies exactly
three complete original rows, preserving every path, frame, event, hash and timestamp. It
verifies every file of those three rows without touching unselected real sequences. The
original manifest and historical results are never changed.

## CPU tests

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
RECON_SELECTED_TRAINING_SOURCES="$F/sources" \
python -B sources/test_selected_hqf.py
```

Seven tests pass in local WSL Python3.10/CPU PyTorch2.9.1; they use temporary synthetic
fixtures only. One full-file-lineage test needs CPU PyTorch. No LPIPS weights or real
images are opened by these tests. Actual pinned metric imports and LPIPS assets have an
additional fail-closed CPU preflight immediately before inference on the evaluation host.

## Prepare and independently verify

Use a fresh `E`, with source files installed at `$E/sources` and sources unchanged after
the protocol is prepared. NFS asset paths are inherited from `F/sources/registered_plan.json`;
they must actually resolve and pass hashes on the host. No automatic path relocation occurs.

```sh
CUDA_VISIBLE_DEVICES='' python -B "$E/sources/evaluate_selected_hqf.py" prepare \
  --training-root "$F" --out "$E/protocol"
CUDA_VISIBLE_DEVICES='' python -B "$E/sources/evaluate_selected_hqf.py" verify \
  --protocol "$E/protocol/selected3_protocol.json"
```

`prepare` emits a disabled `authorization_template_DO_NOT_RUN.json`. An external research
parent must bind the resulting protocol SHA and the existing user instruction in a separate
authorization receipt. Its schema is `recon_selected_hqf3_development_authorization_v1`;
`authorized=true`, `task=reconstruction`, `result_scope=posthoc_selected_hqf3_development`,
`no_feedback_to_training_or_selection=true` and `all15_frozen_before_any_real_prediction=true`
are mandatory, along with nonempty instruction/review notes and the exact protocol digest.
The evaluator never enables this template itself.

## Run only after verification, authorization and an external safe GPU lease

```sh
CUDA_VISIBLE_DEVICES="$LEASED_GPU_UUID" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python -B "$E/sources/evaluate_selected_hqf.py" run \
  --protocol "$E/protocol/selected3_protocol.json" \
  --authorization "$E/authorization.json" --out "$E/evaluation" \
  --device cuda:0 --enable-real-scoring
```

All45 event-only inferences run first and are frozen before any metric scoring. The
inference adapter, recurrent-reset policy, event intervals, five-bin representation,
crop and photometry are unchanged. Every inferable frame except the initial anchor is
scored. LPIPS-Alex, SSIM, MSE and PSNR use the unchanged pinned metric functions. There is
no brightness fit, output-selected frame/ROI, real-checkpoint selection or method aliasing.

## Canonical results

`evaluation/selected3_summary.json`, `method_table.csv`, `per_seed.csv`,
`per_sequence.csv` and `RESULTS.md` contain all15 runs. Frame means are computed within
each sequence, three sequence means receive equal weight within seed, then the three seed
means are summarized by mean and **sample SD, ddof=1**. No seed or failed arm is dropped.
An incomplete/nonfinite matrix is rejected. Positive-infinite PSNR from an exact match is
represented explicitly, with undefined SD, rather than converting it to a finite score.

The original pinned `score_pair` handles the dynamic sequence inventory correctly in its
numerical loops but hardcodes historical HQF14 wording and one metadata count. It is imported
without changing a line; its untouched output is retained as
`legacy_score_raw_unchanged.json`. Do not cite that legacy metadata as a dataset claim.
The canonical selected3 summary records actual sequence/frame counts and this limitation.

Every inference, protocol, source, authorization and metric asset is hash-bound. Existing
output directories are refused; any exception preserves logs/state as failed. This tool
does not implement automatic restart or own external leases. A future parent watcher may
wait for the training receipts, run the same CPU admission, acquire one genuinely free
resource and invoke the frozen command under the user's already granted authorization.
