# One synthetic-only protected-feature candidate

This is a bounded hypothesis test, not a claim that the half/dark tradeoff is solved. The two previous extremes are fixed: C1 adapts all features and improves dark but harms half; hard feature freezing restores much of half but removes most dark gains. The candidate permits feature learning while softly retaining source features on protected normal/half inputs.

## Fixed scientific recipe

Canonical candidate **feature_retention** differs from C1 by one added term. Architecture, initialization, rotation, input representation, core/noise/occlusion, optimizer, labels, augmentations, training order and budget are unchanged. Auxiliary half:eighth sampling remains3:1. C1 base objective is 0.5 normal supervision +0.5 physical auxiliary supervision +2 normal-view output-teacher retention. There is no half output KD and no dark feature KD. Both feature extractor and heads train normally; the independent localizer size head remains frozen exactly as in C1.

For feature tensors F and same-input frozen-source tensors T, define per-sample relative error r=mean((F−T)^2)/max(mean(T^2),1e−6), averaging all feature channels/spatial elements. T and its denominator are detached. Added term is λ×0.5×[mean r(normal)+mean(mask_half×r(auxiliary))]. Masked auxiliary loss is divided by original batch size; it is not renormalized to the number of half examples. The teacher receives each protected view itself, never a different brightness view. No depth target or pseudo-label is supplied by this feature term.

All12 new fits (2generators×3seeds×2branches) retain six epochs, batch32,288updates, AdamW lr1.5e−4/wd.01 and gradient clip5. Reuse original C0/C1 24 branch models at their fixed288updates, with source/model/Adam/history/log/label/augmentation checks. New candidate consumes exactly C1's label and augmentation trajectory. No checkpoint reselection.

## Phase0: zero-update TRAIN-only calibration

The source initial state gives exactly zero feature loss and is **not** used for a fake gradient ratio. Instead, each of twelve fixed completed C1 branch checkpoints is probed on its original first TRAIN batch, reproducing its registered indices,3:1 mask and geometric affine draws. No optimizer is constructed. Compute g_j=feature-parameter gradient norm of full C1 objective and f_j=feature-parameter gradient norm of the proposed feature term at C1. Common λ=0.1/median_j(f_j/g_j). Thus the median weighted feature/primary gradient norm is10% on this calibration set. This coefficient rule and target are our conservative adaptation, not a published optimum. No coefficient sweep or outcome-based adjustment.

Training registration contains only twelve old C1 training receipts and six common source initializer hashes. Calibration does not load validation frames, validation values or any real data. It verifies nonzero finite gradients at drifted C1, zero initial feature discrepancy, untouched model files/states and all twelve method/seed/branch identities. All six model/data helper source hashes are recorded and verified before subsequent phases. Nonfinite/zero gradient or an unusable derived scale stops the phase rather than silently clipping the result.

## Phase1 dependency gates

`calibrate`, `smoke`, and `study` have separate new output directories. Smoke requires passed phase0 receipt with explicit SHA and current source-manifest binding. It performs twelve true one-update fits (both generators,seed0,three arms,two branches) plus independent model/Adam reconstruction. Study requires the SHA-bound passed smoke receipt as well as the same calibration; it adds only12×288updates. Source and λ metadata are saved in checkpoints and execution receipts. No automatic continuation to real evaluation.

Full synthetic evaluation/gates are the byte-locked original native main: normal four-window-per-trajectory subset and full96-rollout condition, plus all windows of four disjoint half/eighth physical-relit validation trajectories. Original30 illumination checks and original8 confirmation checks are unchanged. All30 booleans are additionally recalculated against **original immutable C0 metrics**, preventing an easier baseline. Individual seed failures cannot be hidden by a mean. Recomputed control summaries must match their original values within rtol1e−6/atol1e−8. Real scoring remains disabled even if the candidate passes.

The native dictionary slot `paired_consistency` is only an interface alias and is explicitly mapped to canonical **feature_retention** in protocol/receipt. It does not mean that half output consistency was used. No metric/gate function is replaced.

## Primary-method context and limits

[Less-forgetting Learning in DNNs](https://arxiv.org/abs/1607.00122) regularizes hidden representations between source and adapted networks. [DELTA, ICLR2019](https://openreview.net/pdf?id=rkgbwsAcYm) and its [author code](https://github.com/lixingjian/DELTA) retain source feature maps using attention weighting. We use a simpler same-view final-feature penalty, not the full DELTA algorithm or its attention mechanism. [Kumar etal., ICLR2022](https://openreview.net/pdf?id=UYneFzXSJWh) studies fine-tuning-induced feature distortion under distribution shifts; its results motivate care with features but do not prove success for our regression problem.

The frozen source can also contain errors. A soft feature anchor may still impede necessary adaptation. Existing diagnostic swaps were limited seed0 subsets; they neither certify the candidate nor replace full30+8 evaluation. No claim of E-POSE improvement is made before independent permitted real confirmation.
