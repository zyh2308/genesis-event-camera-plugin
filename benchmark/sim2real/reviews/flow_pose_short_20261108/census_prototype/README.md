# Isolated soft-Census candidate — CPU prototype only

No training, deployment, SSH, real scoring, learned checkpoint loading or actual
TRAIN-weight calibration has been performed. Original/native files remain unchanged.

This is a declared alternative **downstream objective**, not the original published
STE-FlowNet loss and not a simulator/noise/label change. It is motivated by the authors'
implementations of [UnFlow](https://github.com/simonmeister/UnFlow/blob/master/src/e2eflow/core/losses.py)
(`ternary_loss`) and [UFlow](https://github.com/google-research/google-research/blob/master/uflow/uflow_utils.py)
(`census_transform`, `soft_hamming`, `census_loss`), and UFlow's
[ECCV2020 comparison](https://arxiv.org/pdf/2006.04902).

For grayscale endpoint images in [0,1], first multiply by255. For each 7x7 patch:

```
d_k(I,x) = I(x+offset_k) - I(x)
T_k(I,x) = d_k / sqrt(0.81 + d_k^2)
D(x) = sum_k (T_k(I_prev,x)-T_k(warp(I_next,pred_flow),x))^2
                 / (0.1+(T_k(I_prev,x)-T_k(warp(I_next,pred_flow),x))^2)
L_scale = sum_x stopgrad(M_patch(x)) * (abs(D(x))+0.01)^0.4
L_census = (L_scale0 + L_scale1 + L_scale2 + L_scale3) / 4
```

These are the authors' soft transform / Hamming / robust constants, channel **SUM**,
followed by pixel **SUM**, maintaining the original formal loss's four equally weighted
scales. UFlow normally uses a masked mean; this prototype explicitly retains SUM so
that changing reduction is not an undeclared second intervention. OpenCV INTER_LINEAR
resizing of endpoint labels and native per-scale pixel displacement units are preserved.
The prototype requires the same square scales as the formal training path.

The warp uses predicted flow and existing align_corners=True semantics. Its support is
the native >=0.9999 sampling-of-ones test, detached. Patches touching out-of-frame support
or an image border are excluded. Requiring the entire patch to be supported is a declared
conservative extension to center-only masking. No GT mask argument exists. This mask
does not identify physical occlusions inside the image.

The current loss's hard-mask/SUM correction alone was shown to have identical gradients:
see ../flow_mask_algebra_20261108. It is a negative control and must not be mistaken for
the improvement tested here. Census actually changes the appearance-matching gradient.

`calibrate_shared_weight` accepts already measured, fixed **TRAIN** window gradient
records for both EventGenesis and V2E. It refuses missing pairs, changed label/augmentation
hashes, per-method changing model snapshots, GT, real/validation data and optimizer steps.
It defines one common scalar:

```
lambda = sqrt(sum_{generator,train_window} ||grad_outputs L_native_photo||^2
              / sum_{generator,train_window} ||grad_outputs L_census||^2)
new_objective = lambda * L_census + 10 * original_smoothness
```

The norm space must be all four original flow-output tensors for the same frozen model,
same fixed windows and same shared augmentations. The coefficient must be frozen once,
shared by both generators and retained in the experimental receipt **before any optimizer
step**. No separate favorable coefficient, validation sweep or real feedback is allowed.
This only matches pooled initial gradient magnitude; it does not guarantee matching
gradient directions, per-window weights, convergence or downstream quality. This package
does not choose the TRAIN list, load a model or calculate a real coefficient itself.

CPU controls:

```sh
CUDA_VISIBLE_DEVICES='' python -B test_soft_census.py
CUDA_VISIBLE_DEVICES='' python -B run_cpu_oracles.py --out cpu_oracles.json
```

The analytic controls test1px texture translation, the same geometry after positive
affine intensity change and gamma change, correct neighboring displacement-gradient
directions, zero loss/gradient for fully out-of-frame support, the inability to infer
motion from constant patches, four-scale and pixel-SUM batch duplication, input-domain
guards and strict TRAIN-only shared calibration. Identity algebra uses FP64; translation
and lighting controls use FP32. The original provisional FP32 ideal-identity assertion
observed a1.219e-8 soft-Hamming residual from grid rounding; it is retained in a separate
observation rather than claimed to be bitwise zero.

These are meaningful loss-function oracles, not evidence of generator superiority.
The next step, if the independent current diagnostics justify it, is a bounded synthetic
forward/gradient probe and actual paired TRAIN-only calibration. Any optimizer experiment
needs a separately frozen protocol and must retain the existing synthetic/real isolation.
