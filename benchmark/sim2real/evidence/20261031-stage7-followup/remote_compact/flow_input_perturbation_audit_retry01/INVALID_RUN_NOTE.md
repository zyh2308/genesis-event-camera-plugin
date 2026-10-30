# Retry 01 interpretation status

The remote process exited with code 0 and produced a JSON file, but this result is **invalid and must not be used for scientific interpretation**. A stale Python loop variable caused the sc2-x sample to be evaluated with the sc2-y flow target. This made the reported zero-flow baseline disagree with the independently audited baseline, while the script did not yet contain a cross-check that would reject the mismatch.

The original log and JSON are preserved unchanged as evidence of the failed diagnostic implementation. The corrected retry adds per-sample label binding plus mandatory prediction-hash, normal-EPE, and zero-flow-baseline checks against the independent checkpoint audit before accepting any result. No training was performed in retry 01; only the 100-update checkpoints already registered in the protocol were loaded. The GPU reservation was released normally.
