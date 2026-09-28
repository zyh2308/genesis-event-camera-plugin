"""Physical camera/optic profiles used to configure and audit Genesis cameras.

Manufacturer values are useful nominal constraints, but they are not a
replacement for unit-level intrinsic, distortion, focus, or bias calibration.
"""

from dataclasses import asdict, dataclass, replace
import math
from typing import Dict, Optional, Tuple, Union

import numpy as np


@dataclass(frozen=True)
class EventCameraProfile:
    """Manufacturer-level sensor and lens description.

    ``native_resolution_wh`` follows the camera convention ``(width, height)``.
    The profile's field-of-view values describe the complete supplied optic and
    therefore take precedence over a thin-lens calculation from focal length.
    """

    profile_name: str
    camera_model: str
    sensor_model: str
    native_resolution_wh: Tuple[int, int]
    pixel_pitch_um: float
    sensor_diagonal_mm: float
    color_filter: str
    lens_model: str
    lens_manufacturer: str
    focal_length_mm: float
    f_number: float
    mount: str
    horizontal_fov_deg: float
    vertical_fov_deg: float
    diagonal_fov_deg: float
    minimum_focus_distance_m: float
    nominal_contrast_threshold_log: float
    contrast_threshold_nonuniformity_fraction_max: float
    typical_latency_us_at_1klux: float
    background_rate_hz_at_1klux_max: float
    background_rate_hz_at_5lux_max: float
    dynamic_range_db_guaranteed: float
    calibration_status: str
    source_urls: Tuple[str, ...]
    default_bias_offsets: Tuple[Tuple[str, int], ...]
    distortion_model: str = "uncalibrated"

    @property
    def sensor_size_mm(self) -> Tuple[float, float]:
        width, height = self.native_resolution_wh
        pitch_mm = self.pixel_pitch_um * 1e-3
        return width * pitch_mm, height * pitch_mm

    def nominal_intrinsics(
        self, resolution_wh: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """Return the nominal pinhole K used by Genesis.

        Genesis accepts a *vertical* FOV and uses square-pixel focal lengths.
        Using the manufacturer's vertical FOV therefore reproduces its vertical
        framing exactly.  This K remains a nominal initialization: real K and
        distortion must be measured on the physical camera.
        """
        width, height = resolution_wh or self.native_resolution_wh
        if width <= 0 or height <= 0:
            raise ValueError("resolution must contain positive width and height")
        native_aspect = self.native_resolution_wh[0] / self.native_resolution_wh[1]
        if not math.isclose(width / height, native_aspect, rel_tol=0, abs_tol=1e-6):
            raise ValueError(
                "EVK4 nominal intrinsics require an uncropped 16:9 image; "
                "crop/resize intrinsics must be supplied by the data pipeline"
            )
        focal_px = 0.5 * height / math.tan(
            math.radians(0.5 * self.vertical_fov_deg)
        )
        return np.asarray(
            [
                [focal_px, 0.0, 0.5 * width],
                [0.0, focal_px, 0.5 * height],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def genesis_camera_kwargs(
        self, resolution_wh: Optional[Tuple[int, int]] = None
    ) -> Dict[str, object]:
        """Return safe optics kwargs for ``scene.add_camera``.

        F/number is intentionally not mapped to Genesis ``aperture``: the two
        quantities do not share a calibrated unit/model.  Pinhole is the
        reproducible baseline until radiometry and defocus are validated.
        """
        resolution = resolution_wh or self.native_resolution_wh
        self.nominal_intrinsics(resolution)
        return {
            "res": resolution,
            "fov": self.vertical_fov_deg,
            "model": "pinhole",
        }

    def field_size_at_distance(self, distance_m: float) -> Tuple[float, float]:
        """Nominal visible width/height on a plane normal to the optical axis."""
        if distance_m <= 0:
            raise ValueError("distance_m must be positive")
        width = 2.0 * distance_m * math.tan(
            math.radians(0.5 * self.horizontal_fov_deg)
        )
        height = 2.0 * distance_m * math.tan(
            math.radians(0.5 * self.vertical_fov_deg)
        )
        return width, height

    def minimum_distance_for_extent(
        self, width_m: float, height_m: float, margin_fraction: float = 0.15
    ) -> float:
        """Distance needed to contain a planar extent with a framing margin."""
        if width_m <= 0 or height_m <= 0:
            raise ValueError("width_m and height_m must be positive")
        if margin_fraction < 0:
            raise ValueError("margin_fraction must be non-negative")
        scale = 1.0 + margin_fraction
        distance_w = scale * width_m / (
            2.0 * math.tan(math.radians(0.5 * self.horizontal_fov_deg))
        )
        distance_h = scale * height_m / (
            2.0 * math.tan(math.radians(0.5 * self.vertical_fov_deg))
        )
        return max(distance_w, distance_h, self.minimum_focus_distance_m)

    def validate_genesis_camera(
        self,
        camera,
        strict: bool = False,
        fov_tolerance_deg: float = 0.25,
    ) -> Tuple[str, ...]:
        """Validate a Genesis camera against this nominal optical profile.

        Non-strict mode permits an uncropped 16:9 resize for efficient RL and
        returns warnings for metadata/logging. Strict mode additionally
        requires native 1280x720 output and raises on every mismatch.
        """
        messages = []
        violations = []
        resolution = tuple(int(v) for v in camera.res)
        native = self.native_resolution_wh
        if resolution != native:
            message = (
                "render resolution %s is not EVK4 native %s; this is a resized "
                "RL representation, not native-sensor validation" % (resolution, native)
            )
            messages.append(message)
            if strict:
                violations.append(message)

        if not math.isclose(
            resolution[0] / resolution[1],
            native[0] / native[1],
            rel_tol=0,
            abs_tol=1e-6,
        ):
            message = (
                "render aspect ratio does not match the EVK4 16:9 active array; "
                "an explicit crop model and adjusted intrinsics are required"
            )
            messages.append(message)
            violations.append(message)

        camera_fov = getattr(camera, "fov", None)
        if camera_fov is None:
            message = "Genesis camera does not expose vertical fov for EVK4 validation"
            messages.append(message)
            violations.append(message)
        elif abs(float(camera_fov) - self.vertical_fov_deg) > fov_tolerance_deg:
            message = (
                "Genesis vertical fov %.3f deg differs from EVK4 optic nominal "
                "%.3f deg" % (float(camera_fov), self.vertical_fov_deg)
            )
            messages.append(message)
            violations.append(message)

        model = getattr(camera, "model", getattr(camera, "_model", None))
        if model is not None and model != "pinhole":
            message = (
                "EVK4 baseline expects a pinhole render; thin-lens aperture and "
                "defocus are not yet radiometrically calibrated"
            )
            messages.append(message)
            violations.append(message)

        # Non-strict relaxes only native spatial resolution. An EVK4 profile
        # with another aspect ratio, FOV or camera model would describe a
        # different optical system and must not silently proceed.
        if violations:
            raise ValueError("; ".join(violations))
        return tuple(messages)

    def to_metadata(self) -> Dict[str, object]:
        metadata = asdict(self)
        metadata["default_bias_offsets"] = dict(self.default_bias_offsets)
        sensor_width, sensor_height = self.sensor_size_mm
        metadata["sensor_width_mm"] = sensor_width
        metadata["sensor_height_mm"] = sensor_height
        metadata["intrinsics_status"] = "manufacturer_nominal_not_unit_calibrated"
        metadata["distortion_coefficients"] = None
        return metadata


EVK4_HD_IMX636 = EventCameraProfile(
    profile_name="prophesee_evk4_hd_sfa0820_5m_nominal",
    camera_model="Prophesee EVK4-HD",
    sensor_model="Sony IMX636ES",
    native_resolution_wh=(1280, 720),
    pixel_pitch_um=4.86,
    sensor_diagonal_mm=7.14,
    color_filter="monochrome",
    lens_model="SFA 0820-5M",
    lens_manufacturer="Soyo Security Co.",
    focal_length_mm=8.0,
    f_number=2.0,
    mount="C-mount (EVK4 body supports C/CS; supplied C-CS ring)",
    horizontal_fov_deg=41.4,
    vertical_fov_deg=23.6,
    diagonal_fov_deg=47.0,
    minimum_focus_distance_m=0.1,
    nominal_contrast_threshold_log=0.25,
    contrast_threshold_nonuniformity_fraction_max=0.06,
    typical_latency_us_at_1klux=220.0,
    background_rate_hz_at_1klux_max=0.1,
    background_rate_hz_at_5lux_max=10.0,
    dynamic_range_db_guaranteed=86.0,
    calibration_status="manufacturer_nominal_not_unit_calibrated",
    source_urls=(
        "https://www.prophesee.ai/wp-content/uploads/2026/03/EVK4-HD-Prophesee-Evaluation-Kit-Brief-2026.pdf",
        "https://www.prophesee.ai/wp-content/uploads/2024/10/EVK4_HD_Prophesee_Evaluation_Kit_Camera_Manual_1.2_OK.pdf",
        "https://www.sony-semicon.com/files/62/flyer_industry/IMX636-AAMR-Flyer.pdf",
        "https://docs.prophesee.ai/stable/hw/manuals/biases.html",
    ),
    default_bias_offsets=(
        ("bias_diff", 0),
        ("bias_diff_on", 0),
        ("bias_diff_off", 0),
        ("bias_fo", 0),
        ("bias_hpf", 0),
        ("bias_refr", 0),
    ),
)


CAMERA_PROFILES = {
    "prophesee_evk4_hd": EVK4_HD_IMX636,
    "evk4_hd": EVK4_HD_IMX636,
    "evk4": EVK4_HD_IMX636,
}


def get_camera_profile(
    profile: Union[str, EventCameraProfile]
) -> EventCameraProfile:
    """Resolve a profile name and return an isolated immutable value."""
    if isinstance(profile, EventCameraProfile):
        return replace(profile)
    key = str(profile).lower()
    if key not in CAMERA_PROFILES:
        raise ValueError(
            "Unknown camera profile %r. Available: %s"
            % (profile, sorted(CAMERA_PROFILES))
        )
    return replace(CAMERA_PROFILES[key])
