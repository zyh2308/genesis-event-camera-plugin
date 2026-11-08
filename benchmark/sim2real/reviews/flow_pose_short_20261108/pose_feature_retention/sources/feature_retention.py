"""Same-view feature preservation; no depth labels supplied by a teacher."""
from __future__ import annotations
import math
import statistics
import torch

TARGET_GRADIENT_RATIO = .1
NORMALIZATION_EPS = 1e-6


def per_sample_relative_feature_loss(student, teacher):
    if student.shape != teacher.shape or student.ndim < 2:
        raise ValueError('Matched feature maps required')
    reference = teacher.detach()
    dims = tuple(range(1,student.ndim))
    # Teacher RMS squared. No gradient enters either teacher or denominator.
    denominator = reference.square().mean(dims).clamp_min(NORMALIZATION_EPS)
    return (student-reference).square().mean(dims)/denominator


def protected_feature_loss(model, teacher, normal, auxiliary, half_mask):
    if half_mask.shape != (len(auxiliary),) or len(normal) != len(auxiliary):
        raise ValueError('One protected-view mask per paired sample required')
    with torch.no_grad():
        normal_reference = teacher.features(normal)
        half_reference = teacher.features(auxiliary)
    normal_loss = per_sample_relative_feature_loss(model.features(normal),normal_reference)
    auxiliary_loss = per_sample_relative_feature_loss(model.features(auxiliary),half_reference)
    # No reweighting to the number of half samples; eighth samples contribute 0.
    return .5*(normal_loss.mean()+(auxiliary_loss*half_mask.to(auxiliary_loss)).mean())


def feature_gradient_l2(loss, model):
    parameters = tuple(p for p in model.features.parameters() if p.requires_grad)
    if not parameters: raise ValueError('Feature calibration requires trainable features')
    gradients = torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=False)
    return float(sum(g.detach().double().square().sum() for g in gradients).sqrt())


def common_coefficient(records, expected_count=12):
    if len(records) != expected_count: raise ValueError('Calibrate every fixed generator/seed/branch')
    ratios=[]
    for row in records:
        g,f = row['C1_objective_feature_gradient_l2'],row['feature_retention_gradient_l2']
        if not (math.isfinite(g) and math.isfinite(f) and g>1e-12 and f>1e-12):
            raise ValueError('Nonzero finite gradients required away from the source initial state')
        ratios.append(f/g)
    coefficient=TARGET_GRADIENT_RATIO/statistics.median(ratios)
    if not math.isfinite(coefficient) or not 1e-6 <= coefficient <= 1e6:
        raise ValueError('Unusable calibration scale; stop instead of silently clipping')
    return coefficient,[coefficient*r for r in ratios]
