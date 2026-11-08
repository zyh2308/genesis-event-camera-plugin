"""Census v2: unchanged values; internal bilinear-roundoff domain guard only."""
import math

import torch
import torch.nn.functional as F

PAPER_CODE = 'https://github.com/simonmeister/UnFlow/blob/master/src/e2eflow/core/losses.py'
UFLOW_CODE = 'https://github.com/google-research/google-research/blob/master/uflow/uflow_utils.py'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def _image(image, interpolated=False):
    require(image.ndim == 4 and image.shape[1] == 1 and image.dtype in (torch.float32, torch.float64),
            'Expected floating [B,1,H,W] grayscale image')
    # Bilinear interpolation uses four floating products/sums. A bound of eight
    # dtype eps admits their roundoff without clipping or altering any value.
    tolerance = 8 * torch.finfo(image.dtype).eps if interpolated else 0.
    require(bool(torch.isfinite(image).all()) and bool((image >= -tolerance).all()) and bool((image <= 1+tolerance).all()),
            'Census expects finite grayscale in [0,1], scaled internally to [0,255]')


def warp_and_predicted_support(image, flow, input_interpolated=False):
    _image(image, interpolated=input_interpolated)
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


def census_transform(image, patch_size=7, interpolated=False):
    _image(image, interpolated=interpolated)
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


def census_loss_sum(previous, following, predicted_flow, patch_size=7, inputs_interpolated=False):
    """Author soft transform/Hamming/robust formula, pixel SUM, predicted masks only.

    Native gray resizing and per-scale flow units are supplied unchanged by the caller.
    This function has no GT-mask argument and deliberately performs no valid-count mean.
    """
    _image(previous, interpolated=inputs_interpolated); _image(following, interpolated=inputs_interpolated)
    require(previous.shape == following.shape and previous.device == following.device and previous.dtype == following.dtype,
            'Endpoint image grids differ')
    warped, inbounds = warp_and_predicted_support(following, predicted_flow, input_interpolated=inputs_interpolated)
    # Out-of-frame padding can be outside [0,1] only due to numerical rounding; no clamp
    # or brightness/histogram fit is added. Valid input's bilinear interpolation is bounded.
    a, b = census_transform(previous, patch_size, interpolated=inputs_interpolated), census_transform(warped, patch_size, interpolated=True)
    distance_sq = (a-b).square()
    # Authors use a SUM over patch channels, not a mean. Its units need calibration.
    hamming = (distance_sq / (.1 + distance_sq)).sum(dim=1, keepdim=True)
    robust = (hamming.abs() + .01).pow(.4)
    mask = patch_support(inbounds, patch_size)
    return (robust * mask).sum(), dict(predicted_inbounds=inbounds, loss_support=mask,
                                     soft_hamming=hamming, pointwise_robust=robust)


def native_gray_resize(image, hw, like, input_interpolated=False):
    """Match native OpenCV INTER_LINEAR label resizing; image labels need no gradient."""
    import cv2
    import numpy as np
    _image(image, interpolated=input_interpolated)
    require(image.requires_grad is False, 'Endpoint-label tensor must not require gradients')
    h, w = hw
    # Native formal runs use square scales, so its historical (height,width) call is
    # identical to OpenCV's correct (width,height) tuple. Do not extend it silently.
    require(h == w, 'This candidate preserves the original formal square-scale contract')
    rows = [cv2.resize(v[0], (w, h), interpolation=cv2.INTER_LINEAR)
            for v in image.detach().cpu().numpy()]
    return torch.from_numpy(np.stack(rows)[:, None]).to(device=like.device, dtype=like.dtype)


def multiscale_census_sum(outputs, previous, following, patch_size=7, inputs_interpolated=False):
    require(isinstance(outputs, (tuple, list)) and len(outputs) == 4, 'Require original four predicted scales')
    losses, details = [], []
    for flow in outputs:
        h, w = flow.shape[-2:]
        p = native_gray_resize(previous, (h, w), flow, input_interpolated=inputs_interpolated)
        n = native_gray_resize(following, (h, w), flow, input_interpolated=inputs_interpolated)
        loss, detail = census_loss_sum(p, n, flow, patch_size, inputs_interpolated=True)
        losses.append(loss); details.append(detail)
    # Same four unit weights and division by number of scales as original objective_dt1.
    return sum(losses)/4., details


# Shared lambda uses all82 requires-grad parameters;70 are connected in dt1.
# Its only definition is census_contract.shared_lambda, with unused entries zero.
