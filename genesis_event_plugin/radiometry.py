"""Linear radiance/radiometry utilities for the physics-first event path.

The legacy plugin turns display RGB into an 8-bit grayscale image before the
DVS model.  This module keeps a separate, explicitly linear path.  Values are
never tone-mapped or clipped here; the sensor gain/white level is configured
downstream in :class:`DvsNoiseConfig`.
"""

from typing import Sequence

import numpy as np


DEFAULT_LUMINANCE_WEIGHTS = np.asarray(
    (0.2126, 0.7152, 0.0722), dtype=np.float32
)


def _as_float_array(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim not in (2, 3):
        raise ValueError(f"image must be HxW or HxWx3, got {arr.shape}")
    if arr.ndim == 3 and arr.shape[-1] != 3:
        raise ValueError(f"RGB image must have three channels, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("image contains NaN or infinite values")
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.float32) / 255.0
    return arr.astype(np.float32, copy=False)


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """Convert normalized sRGB code values to linear light.

    Integer inputs are interpreted as 8-bit display values.  Floating inputs
    must be in the normalized ``[0, 1]`` range.
    """
    arr = _as_float_array(rgb)
    if arr.ndim != 3:
        raise ValueError("srgb_to_linear expects an RGB image")
    if arr.min() < -1e-6 or arr.max() > 1.0 + 1e-6:
        raise ValueError("floating sRGB values must be in [0, 1]")
    arr = np.clip(arr, 0.0, 1.0)
    return np.where(
        arr <= 0.04045,
        arr / 12.92,
        ((arr + 0.055) / 1.055) ** 2.4,
    ).astype(np.float32)


def linear_luminance(
    linear_rgb: np.ndarray,
    weights: Sequence[float] = DEFAULT_LUMINANCE_WEIGHTS,
) -> np.ndarray:
    """Collapse linear RGB/radiance to a non-negative sensor luminance."""
    arr = np.asarray(linear_rgb, dtype=np.float32)
    if arr.ndim == 2:
        out = arr
    elif arr.ndim == 3 and arr.shape[-1] == 3:
        w = np.asarray(weights, dtype=np.float32)
        if w.shape != (3,):
            raise ValueError("luminance weights must have shape (3,)")
        out = np.tensordot(arr, w, axes=([-1], [0]))
    else:
        raise ValueError(f"expected HxW or HxWx3, got {arr.shape}")
    if not np.isfinite(out).all() or out.min() < 0:
        raise ValueError("linear luminance must be finite and non-negative")
    return out.astype(np.float32, copy=False)


def prepare_linear_radiance(
    image: np.ndarray,
    input_space: str = "srgb",
    exposure_scale: float = 1.0,
    black_level: float = 0.0,
    weights: Sequence[float] = DEFAULT_LUMINANCE_WEIGHTS,
) -> np.ndarray:
    """Prepare a linear luminance frame for the event sensor.

    ``input_space='srgb'`` is an explicitly marked fallback for Genesis
    backends that expose only display RGB.  ``input_space='linear'`` accepts
    floating HDR/radiance values, including values above one.
    """
    if input_space not in {"srgb", "linear"}:
        raise ValueError("input_space must be 'srgb' or 'linear'")
    if exposure_scale <= 0:
        raise ValueError("exposure_scale must be positive")

    if input_space == "srgb":
        linear = srgb_to_linear(image)
    else:
        raw = np.asarray(image)
        if raw.ndim == 3 and raw.shape[-1] == 3:
            linear = raw.astype(np.float32, copy=False)
        elif raw.ndim == 2:
            linear = raw.astype(np.float32, copy=False)
        else:
            raise ValueError(f"expected HxW or HxWx3, got {raw.shape}")
        if not np.isfinite(linear).all() or linear.min() < 0:
            raise ValueError("linear radiance must be finite and non-negative")

    if black_level != 0:
        linear = np.maximum(linear - float(black_level), 0.0)
    linear = linear * float(exposure_scale)
    return linear_luminance(linear, weights=weights)


def log_radiance(luminance: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Return numerically stable log radiance for the contrast bound."""
    if eps <= 0:
        raise ValueError("eps must be positive")
    lum = np.asarray(luminance, dtype=np.float32)
    if not np.isfinite(lum).all() or lum.min() < 0:
        raise ValueError("luminance must be finite and non-negative")
    return np.log(np.maximum(lum, float(eps))).astype(np.float32)
