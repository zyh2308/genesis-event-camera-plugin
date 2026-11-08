"""Auxiliary synthetic reconstruction losses; no training/configuration side effects.

Edge formula follows the MPRNet CVPR 2021 author's implementation:
https://github.com/swz30/MPRNet/blob/main/Deblurring/losses.py

Only channel/device/dtype handling is generalized. Both losses return a scalar
mean over the input NCHW tensor. The training caller must accumulate this mean
over time using the SAME convention as its existing perceptual loss; this module
never clamps intensities, normalizes images, or alters TC loss. Public modules
return weighted losses and expose .weight, matching the native trainer API.
"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _validate_pair(prediction: Tensor, target: Tensor) -> None:
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("prediction and target must have the same NCHW shape")
    if any(size == 0 for size in prediction.shape):
        raise ValueError("prediction and target must be nonempty")
    if prediction.device != target.device or prediction.dtype != target.dtype:
        raise ValueError("prediction and target must share device and dtype")
    if not prediction.is_floating_point():
        raise TypeError("prediction and target must be floating point")


def photometric_l1(prediction: Tensor, target: Tensor) -> Tensor:
    """Mean absolute intensity error, with no clipping or intensity fitting."""
    _validate_pair(prediction, target)
    return (prediction - target).abs().mean()


class PhotometricL1(nn.Module):
    """Native-trainer compatible weighted mean absolute intensity error."""

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()
        self.weight = float(weight)

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        return self.weight * photometric_l1(prediction, target)


class LaplacianEdgeLoss(nn.Module):
    """Charbonnier distance between MPRNet's Laplacian residuals.

    With G the author's separable 5x5 Gaussian, D retaining even indices, and
    U inserting zeros at odd indices, R(x) = x - G(4 U(D(G(x)))). The loss is
    mean(sqrt((R(prediction)-R(target))**2 + eps**2)). Every G convolution
    uses replicate padding. At equal inputs its value is eps, and its gradient
    is zero. No eps subtraction or border crop is applied.

    The official replicate-padding/downsampling construction has a finite-image
    boundary response, including for a constant intensity offset; removing that
    response would change the author's formula.
    """

    def __init__(self, weight: float = 0.05, eps: float = 1e-3) -> None:
        super().__init__()
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.weight = float(weight)
        self.eps = float(eps)
        # Match the original torch.Tensor constants (float32), then support
        # module.to(...) and input-based conversion without any global CUDA call.
        k = torch.tensor([0.05, 0.25, 0.4, 0.25, 0.05], dtype=torch.float32)
        self.register_buffer("kernel", torch.outer(k, k)[None, None])

    def _gaussian(self, image: Tensor) -> Tensor:
        channels = image.shape[1]
        kernel = self.kernel.to(device=image.device, dtype=image.dtype)
        kernel = kernel.expand(channels, 1, 5, 5).contiguous()
        return F.conv2d(F.pad(image, (2, 2, 2, 2), mode="replicate"),
                        kernel, groups=channels)

    def laplacian_residual(self, image: Tensor) -> Tensor:
        filtered = self._gaussian(image)
        down = filtered[:, :, ::2, ::2]
        up = torch.zeros_like(filtered)
        up[:, :, ::2, ::2] = down * 4
        return image - self._gaussian(up)

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        _validate_pair(prediction, target)
        delta = (self.laplacian_residual(prediction)
                 - self.laplacian_residual(target))
        return self.weight * torch.sqrt(delta.square() + self.eps * self.eps).mean()
