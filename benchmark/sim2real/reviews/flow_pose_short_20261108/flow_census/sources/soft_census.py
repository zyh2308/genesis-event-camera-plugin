"""Isolated soft-Census prototype: no model/optimizer/training/real-data access."""
import math

import torch
import torch.nn.functional as F

PAPER_CODE = 'https://github.com/simonmeister/UnFlow/blob/master/src/e2eflow/core/losses.py'
UFLOW_CODE = 'https://github.com/google-research/google-research/blob/master/uflow/uflow_utils.py'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def _image(image):
    require(image.ndim == 4 and image.shape[1] == 1 and image.dtype in (torch.float32, torch.float64),
            'Expected floating [B,1,H,W] grayscale image')
    require(bool(torch.isfinite(image).all()) and bool((image >= 0).all()) and bool((image <= 1).all()),
            'Census expects finite grayscale in [0,1], scaled internally to [0,255]')


def warp_and_predicted_support(image, flow):
    _image(image)
    b, _, h, w = image.shape
    require(flow.shape == (b, 2, h, w) and flow.device == image.device and flow.dtype == image.dtype,
            'Flow must match native image grid and dtype; units are px per interval at this scale')
    require(bool(torch.isfinite(flow).all()), 'Nonfinite predicted flow')
    yy, xx = torch.meshgrid(torch.arange(h, device=flow.device, dtype=flow.dtype),
                           torch.arange(w, device=flow.device, dtype=flow.dtype), indexing='ij')
    grid = torch.stack((xx, yy))[None] + flow
    grid = torch.stack((2 * grid[:, 0] / max(w-1, 1)-1, 2 * grid[:, 1] / max(h-1, 1)-1), dim=-1)
    warped = F.grid_sample(image, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    support = F.grid_sample(torch.ones_like(image), grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    # Same >=.9999 support test as native; make its zero derivative explicit.
    support = (support >= .9999).to(image.dtype).detach()
    return warped, support


def census_transform(image, patch_size=7):
    _image(image)
    require(isinstance(patch_size, int) and patch_size >= 3 and patch_size % 2 == 1,
            'Census patch size must be odd and >=3')
    b, _, h, w = image.shape
    intensity = image * 255.
    neighbors = F.unfold(intensity, kernel_size=patch_size, padding=patch_size//2)
    neighbors = neighbors.reshape(b, patch_size*patch_size, h, w)
    diff = neighbors - intensity
    return diff / torch.sqrt(.81 + diff.square())


def patch_support(predicted_support, patch_size):
    # Exclude patch borders and any patch intersecting predicted out-of-frame support.
    # This is more conservative than center-only masking, and uses no GT/GT validity.
    kernel = predicted_support.new_ones(1, 1, patch_size, patch_size)
    count = F.conv2d(predicted_support, kernel, padding=patch_size//2)
    return (count >= patch_size*patch_size-.0001).to(predicted_support.dtype).detach()


def census_loss_sum(previous, following, predicted_flow, patch_size=7):
    """Author soft transform/Hamming/robust formula, pixel SUM, predicted masks only.

    Native gray resizing and per-scale flow units are supplied unchanged by the caller.
    This function has no GT-mask argument and deliberately performs no valid-count mean.
    """
    _image(previous); _image(following)
    require(previous.shape == following.shape and previous.device == following.device and previous.dtype == following.dtype,
            'Endpoint image grids differ')
    warped, inbounds = warp_and_predicted_support(following, predicted_flow)
    # Out-of-frame padding can be outside [0,1] only due to numerical rounding; no clamp
    # or brightness/histogram fit is added. Valid input's bilinear interpolation is bounded.
    a, b = census_transform(previous, patch_size), census_transform(warped, patch_size)
    distance_sq = (a-b).square()
    # Authors use a SUM over patch channels, not a mean. Its units need calibration.
    hamming = (distance_sq / (.1 + distance_sq)).sum(dim=1, keepdim=True)
    robust = (hamming.abs() + .01).pow(.4)
    mask = patch_support(inbounds, patch_size)
    return (robust * mask).sum(), dict(predicted_inbounds=inbounds, loss_support=mask,
                                     soft_hamming=hamming, pointwise_robust=robust)


def native_gray_resize(image, hw, like):
    """Match native OpenCV INTER_LINEAR label resizing; image labels need no gradient."""
    import cv2
    import numpy as np
    _image(image)
    require(image.requires_grad is False, 'Endpoint-label tensor must not require gradients')
    h, w = hw
    # Native formal runs use square scales, so its historical (height,width) call is
    # identical to OpenCV's correct (width,height) tuple. Do not extend it silently.
    require(h == w, 'This candidate preserves the original formal square-scale contract')
    rows = [cv2.resize(v[0], (w, h), interpolation=cv2.INTER_LINEAR)
            for v in image.detach().cpu().numpy()]
    return torch.from_numpy(np.stack(rows)[:, None]).to(device=like.device, dtype=like.dtype)


def multiscale_census_sum(outputs, previous, following, patch_size=7):
    require(isinstance(outputs, (tuple, list)) and len(outputs) == 4, 'Require original four predicted scales')
    losses, details = [], []
    for flow in outputs:
        h, w = flow.shape[-2:]
        p = native_gray_resize(previous, (h, w), flow)
        n = native_gray_resize(following, (h, w), flow)
        loss, detail = census_loss_sum(p, n, flow, patch_size)
        losses.append(loss); details.append(detail)
    # Same four unit weights and division by number of scales as original objective_dt1.
    return sum(losses)/4., details


def calibrate_shared_weight(records, expected_window_ids):
    """TRAIN-only pooled RMS-gradient match; never separate generator coefficients.

    Native/census gradient norms must be measured for the same fixed model, augment,
    windows and four native flow outputs. This utility does not load or select anything.
    It returns metadata only, and never modifies an objective or optimizer.
    """
    require(isinstance(expected_window_ids, (tuple, list)) and len(expected_window_ids) > 0 and
            len(set(expected_window_ids)) == len(expected_window_ids), 'Fixed unique TRAIN-window list required')
    methods = ('eventgenesis', 'v2e')
    expected = {(m, wid) for m in methods for wid in expected_window_ids}
    require(len(records) == len(expected) and {(r['method'], r['window_id']) for r in records} == expected,
            'Every frozen TRAIN window must appear once for both generators')
    native_sq = census_sq = 0.
    views, snapshots = {}, {}
    for r in records:
        require(r['split'] == 'train' and r['optimizer_updates'] == 0 and r['GT_used'] is False and
                r['real_data_used'] is False, 'Calibration cannot use validation/real/GT or optimizer updates')
        require(r['gradient_space'] == 'four_native_flow_outputs' and len(r['model_state_sha256']) == 64,
                'Calibration model/gradient convention missing')
        if r['method'] in snapshots:
            require(snapshots[r['method']] == r['model_state_sha256'], 'Model snapshot changed during calibration')
        snapshots[r['method']] = r['model_state_sha256']
        if r['window_id'] in views:
            require(views[r['window_id']] == r['physical_label_and_augmentation_sha256'], 'Unpaired calibration views')
        views[r['window_id']] = r['physical_label_and_augmentation_sha256']
        a, b = r['native_photo_gradient_l2'], r['census_photo_gradient_l2']
        require(all(isinstance(v, (float, int)) and math.isfinite(v) and v >= 0 for v in (a,b)), 'Bad gradient norm')
        native_sq += a*a; census_sq += b*b
    require(native_sq > 0 and census_sq > 0, 'Cannot calibrate zero total gradients')
    value = math.sqrt(native_sq / census_sq)
    return dict(schema='flow_census_train_gradient_calibration_v1', shared_census_weight=value,
                formula='sqrt(sum_both_methods_windows native_photo_grad_l2^2 / sum_both_methods_windows census_photo_grad_l2^2)',
                methods=list(methods), expected_window_ids=list(expected_window_ids), records=records,
                original_smoothness_coefficient=10., reduction='pixel SUM then equal4 scale mean',
                GT_used=False, validation_used=False, real_data_used=False, optimizer_updates=0,
                limitation='Matches pooled gradient magnitude only, not gradient directions or optimal task performance.')
