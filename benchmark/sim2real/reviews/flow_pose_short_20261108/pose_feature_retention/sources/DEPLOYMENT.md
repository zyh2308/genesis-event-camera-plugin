# Local implementation ready; no server task started here

Use the existing owned GPU4 queue only, UUID `GPU-0602549d-5c0b-133e-63ab-a01d1f6a49d3`. No new GPU lease/reservation, SSH connection or training process is created by this package. All output directories must be absent. Queue owner supplies bounded timeouts and stops on any failed phase. Protected reconstruction/flow tasks are untouched.

Run CPU tests from the bundle directory: `python -B -m unittest -v test_feature_retention`. All fixtures are included in the same bundle; no local-machine or sibling-folder test path is required.

Worker: `run_pose_feature_retention.py`. Common flags are `--campaign`, `--prior-source-dir`, `--prior-source-manifest-sha256`, `--radiance-train`, `--radiance-validation`, `--prior-study`, `--diagnostic`, `--out`. Prior native manifest should bind actual server SHAe237e55a02a359d224d9be3ba33c404100fd7a98509f2994e715c1d7c16e4502. Old native file remains SHA85951e6d43e59140b81d7303370a6b252fec902dc065ff17bb30faf8e4fc6186.

1. `--phase calibrate --out NEW/phase0`. It uses only TRAIN data and pinned C1 checkpoints. The common validation/diagnostic arguments are accepted for interface consistency but not opened by calibration. Output `phase0/calibration.json` must pass; independently obtain its SHA.
2. `--phase smoke --calibration NEW/phase0/calibration.json --calibration-sha256 ACTUAL_SHA --out NEW/smoke`. Verify exit0 and `smoke/feature_retention_execution_receipt.json`; independently obtain that receipt SHA.
3. `--phase study --calibration … --calibration-sha256 … --smoke-receipt NEW/smoke/feature_retention_execution_receipt.json --smoke-receipt-sha256 ACTUAL_SHA --out NEW/study`.

Recommended existing-queue limits: phase0600s, smoke600s, study7200s, with owner timeout kill-after60s applying only to its child. These are bounds, not predicted runtimes. No automatic real scoring is included.

Sources derive from the prior audited policy and evaluator wrapper using included `build_policy.py`/`build_worker.py`. Old reference files are unmodified bytes. The only candidate training change is the feature retention loss; control-reuse checks and original evaluation remain intact. Current status is local CPU verified, server deployment pending; no new scientific result has been written.
