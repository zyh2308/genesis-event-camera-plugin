"""Linear radiance/radiometry utilities for the physics-first event path.

The legacy plugin turns display RGB into an 8-bit grayscale image before the
DVS model.  This module keeps a separate, explicitly linear path.  Values are
never tone-mapped or clipped here; the sensor gain/white level is configured
downstream in :class:`DvsNoiseConfig`.
"""

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np


DEFAULT_LUMINANCE_WEIGHTS = np.asarray(
    (0.2126, 0.7152, 0.0722), dtype=np.float32
)

RESPONSE_MODES = {"video_linlog", "physical_log", "calibrated_transfer"}
CALIBRATION_STATUSES = {"uncalibrated", "effective", "calibrated"}


@dataclass(frozen=True)
class RadiometricResponse:
    """Explicit transfer definition between linear input and log response.

    ``video_linlog`` is the historical V2E-compatible transfer and retains its
    8-bit DN transition. ``physical_log`` is deliberately different: it
    applies a configurable affine sensor response and a numerical floor, never
    importing the DN=20 video convention. ``calibrated_transfer`` loads a
    monotone lookup table from JSON and is intended for measured camera data.

    The fields are calibration metadata, not claims that the values describe a
    particular EVK4 unit. A calibrated status must be supplied explicitly.
    """

    mode: str = "physical_log"
    gain: float = 1.0
    offset: float = 0.0
    floor: float = 1e-6
    calibration_status: str = "uncalibrated"
    calibration_id: str = ""
    transfer_path: Optional[str] = None

    def validate(self) -> None:
        if self.mode not in RESPONSE_MODES:
            raise ValueError(f"unknown radiometric response mode: {self.mode}")
        if self.gain <= 0 or self.floor <= 0:
            raise ValueError("radiometric gain and floor must be positive")
        if self.calibration_status not in CALIBRATION_STATUSES:
            raise ValueError(
                "calibration_status must be uncalibrated, effective, or calibrated"
            )
        if self.mode == "calibrated_transfer" and not self.transfer_path:
            raise ValueError("calibrated_transfer requires transfer_path")
        if self.mode == "calibrated_transfer" and self.calibration_status == "uncalibrated":
            raise ValueError("a calibrated transfer table cannot be marked uncalibrated")

    def metadata(self) -> dict:
        return {"radiometric_response": asdict(self)}


def load_calibrated_transfer(path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """Load a measured piecewise-linear input→log-response table.

    JSON schema::

        {"input": [..], "log_response": [..]}

    The table is intentionally explicit and finite. Values outside its range
    must be rejected by the DVS emulator rather than silently clipped.
    """
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    x = np.asarray(payload.get("input"), dtype=np.float64)
    y = np.asarray(payload.get("log_response"), dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.size < 2 or x.shape != y.shape:
        raise ValueError("calibrated transfer needs equal 1-D input/log_response arrays")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(np.diff(x) <= 0):
        raise ValueError("calibrated transfer input must be finite and strictly increasing")
    return x, y


def physical_log_response(
    luminance: np.ndarray,
    gain: float = 1.0,
    offset: float = 0.0,
    floor: float = 1e-6,
) -> np.ndarray:
    """Apply the explicit physical-log response without a video DN knee."""
    lum = np.asarray(luminance, dtype=np.float32)
    if not np.isfinite(lum).all() or (lum < 0).any():
        raise ValueError("luminance must be finite and non-negative")
    if gain <= 0 or floor <= 0:
        raise ValueError("gain and floor must be positive")
    return np.log(np.maximum(float(gain) * lum + float(offset), float(floor))).astype(np.float32)


def radiometric_response_metadata(response: RadiometricResponse) -> dict:
    response.validate()
    return response.metadata()


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
