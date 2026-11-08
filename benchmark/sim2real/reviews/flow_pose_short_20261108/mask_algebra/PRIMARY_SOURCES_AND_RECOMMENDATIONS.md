# Read-only objective / representation review, 2026-11-08

No simulator, noise, dataset, trainer, checkpoint or metric source was changed. No SSH,
training or real scoring was performed for this review. Local CPU tensors only were used
for the separately retained native-mask algebra audit.

## Five primary sources

1. **Jonschkowski et al., What Matters in Unsupervised Optical Flow, ECCV2020.**
   <https://arxiv.org/pdf/2006.04902> — Sec3 photometric consistency / occlusion and
   Table3 compare Charbonnier, SSIM and Census; Table4 shows mask-gradient stopping matters.
   Census is strongest for KITTI in their comparisons, while SSIM can be stronger in some
   Sintel settings. This is evidence for testing, not a guarantee on synthetic event input.
2. **Authors' UFlow implementation.**
   <https://github.com/google-research/google-research/blob/master/uflow/uflow_utils.py>
   — `compute_loss` combines `stop_gradient(not_occluded_masks * valid_warp_masks)` with
   a masked normalized photometric loss; `census_transform`, `soft_hamming`, `census_loss`
   implement soft Census and exclude patch borders; `weighted_ssim` / the SSIM branch offer
   another illumination-robust appearance comparison. Do not copy their coefficients while
   retaining a different reduction convention without an explicit scaling contract.
3. **Authors' UnFlow implementation, AAAI2018.**
   <https://github.com/simonmeister/UnFlow/blob/master/src/e2eflow/core/losses.py>
   — `compute_losses`, `create_outgoing_mask`, `ternary_loss` implement predicted outgoing
   masks, optional forward/backward consistency masks and soft Census. For a backward
   estimate they evaluate the same network on reversed inputs. A backward field is not
   obtained by simply negating the forward field. For event inputs, reversing interval order
   alone is insufficient: timestamp-bin and polarity conventions also require validation.
4. **Jung et al., Less-forgetting Learning in Deep Neural Networks.**
   <https://arxiv.org/pdf/1607.00122> — SecII Eq3 penalizes the Euclidean distance between
   frozen source and adapting target penultimate features on the same input. Its motivating
   setting explicitly includes different sensors and lighting. The paper freezes classifier
   boundaries; that part does not directly justify freezing our center/depth regression heads.
5. **Li, Grandvalet & Davoine, Explicit Inductive Bias for Transfer Learning with
   Convolutional Networks, ICML2018.**
   <https://proceedings.mlr.press/v80/li18a.html> — L2-SP regularizes toward the original
   trained parameter vector rather than toward zero. It supports a lightweight alternative
   for retaining an existing representation. Ordinary AdamW weight decay is not L2-SP.

## Current STE loss: established facts and limits

`run_steflow_stage2_pilot.py::objective_dt1` (line188 in the WSL repository) calls the
unchanged four-scale photometric loss and adds `10 * smooth_loss`. The archived downstream
`multiscaleloss.py` has SHA
`ff401317e49ce37ae0f8e9bfd6b82f133549b4a0271be35095f74bdc50a439c0`.
It uses per-pixel Charbonnier **SUM**, averaged over scales, plus multiscale second-order
smoothness **MEAN**. The `event_images` argument is not used in the photometric function.

`warp` returns `sampled_next * binary_inbounds_mask`. The photometric function subsequently
subtracts `previous`, so invalid locations remain as black-image residual constants.
However, hard thresholding makes the mask derivative zero. With the original SUM reduction,
removing those residual constants does **not** alter the flow gradient in our eight actual
native-function cases, or the corresponding small convolution's parameter gradient.
`mask.requires_grad=True` alone does not mean it has a nonzero derivative.

The retained `result.json` and `stdout.log` contain all eight cases and exact zero gradient
differences, including the alternative weighting after Charbonnier. Do not train mask-only
as an improvement or infer a cause of zero-flow collapse from the lower scalar loss.
Likewise, physical GT-invalid pixels include occlusion/geometry; they are not the same set
as predicted out-of-frame samples. A previously observed physical-invalid gradient fraction
cannot be attributed to this out-of-frame constant term.

The nonsquare CPU negative case also exposes upstream `cv2.resize(...,(height,width))`.
OpenCV expects `(width,height)`. Formal training256x256 and square352x352 evaluation paths
are unaffected. This is retained as a restricted upstream issue, not a training root cause.

## Minimal next objective candidate

First use the mask-only correction as a **negative gradient control**, not a trained arm.
Then compare original Charbonnier with one soft-Census candidate on synthetic windows:
unchanged model / four scales / geometry / pixel-SUM / smoothness10, no additional SSIM,
backward model or GT mask. Census compares local intensity structure and can tolerate some
photometric changes that violate raw brightness constancy. It cannot supply missing motion
information, recover unobservable flat regions or solve genuine occlusions by itself.

Compute valid warp support from predicted flow only; discard Census patch borders and
patches intersecting invalid support. Detach masks. Calibrate a **single shared** Census
coefficient from frozen training-window gradient norms across both generators, before any
optimizer step; preserve smoothness10 and record units. Never fit it to validation/real
scores or use a separate favorable coefficient per generator. This is a declared new
downstream-loss ablation, not an exact reproduction of published STE-FlowNet training.

SSIM is a credible alternative, but stacking both in the first test introduces another
mixing coefficient. Forward/backward occlusion handling requires a semantically correct
reversed-event model pass and more compute; it should follow a demonstrated occlusion
failure rather than be bundled into this minimum candidate.

## Pose: preserving features versus only outputs

Current inspected source:
`outputs/pose_priority_20261108/pose_priority_policy.py::Fitter.fit`, lines104–155,
and the original `pose_two_branch_coverage.py::fit/fit_center`.
They copy a frozen teacher, but existing retention compares predicted centers, sizes,
depth-related outputs and/or center logits. Trainable parameters share AdamW LR1.5e-4.
Freezing `model.size` in the center branch does not freeze its shared feature extractor.

Two small, separate controls are justified:

- **Differential LR:** retain head LR1.5e-4 and use backbone LR1.5e-5. The 0.1 ratio is
  a conservative experiment choice, not an exact hyperparameter prescribed by the papers.
  This reduces representation drift but does not guarantee no forgetting.
- **Soft feature retention:** preserve the existing supervised losses and add mean
  squared penultimate-feature error between student and its own frozen same-method teacher
  on clean replay inputs. Teacher is eval/no-grad and immutable; same spatial augmentation
  must feed each pair. No new inference head or external model is required. Compare against
  the same-budget control; choose the coefficient using training/synthetic evidence only.

Start with clean replay feature retention. Forcing every dark-input feature to match a
bright teacher's features can preserve brightness-specific nuisance factors and obstruct
necessary adaptation. Same-input teacher features on extremely dark views may themselves
be poor. Features can also be stable while the task remains wrong; keep per-seed/per-light
depth, center, rotation and ADD-S stability gates unchanged.

L2-SP toward the initial backbone is a later cheaper alternative, not a reason to combine
three regularizers immediately. Hard-frozen backbone controls are informative but remove
all representational adaptation; soft retention / smaller LR tests a different tradeoff.
Neither the cited classification/domain-shift results nor current synthetic performance
proves real E-POSE improvement. Previously seen E-POSE remains development evidence.
