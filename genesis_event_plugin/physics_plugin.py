"""Genesis-facing wrapper for the physics-first event pipeline.

This is intentionally a new class rather than a silent change to the legacy
``GenesisEventPlugin``.  The caller supplies two explicit providers:

* ``radiance_provider`` returns linear HDR/radiance (H×W or H×W×3);
* ``motion_state_provider`` returns ``camera_to_world`` and absolute
  ``object_to_world`` transforms keyed by the segmentation IDs.

Keeping these providers explicit prevents a display-RGB fallback from being
mistaken for a calibrated Genesis radiance buffer.
"""

from dataclasses import asdict, replace
import logging
from typing import Callable, Dict, Mapping, Optional

import numpy as np

from .adaptive_sampler import AdaptiveSamplingConfig
from .camera_model import EventCameraProfile, get_camera_profile
from .camera_margin import CameraMargin, crop_margin
from .dvs_emulator import DvsEmulator
from .noise_model import DvsNoiseConfig, get_preset
from .physics_interpolator import PhysicsInterpolator
from .radiometry import prepare_linear_radiance
from .recorder import GenesisEventRecorder

logger = logging.getLogger(__name__)


def _array(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _pose(value, name):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite 4x4 transform")
    return value


class GenesisPhysicsEventPlugin:
    """Radiance + physical-warp event plugin (v2 experimental mainline)."""

    def __init__(
        self,
        output_dir: Optional[str] = None,
        preset: str = "real_v2",
        noise_config: DvsNoiseConfig = None,
        camera_profile: EventCameraProfile = None,
        sampler_config: AdaptiveSamplingConfig = None,
        input_space: str = "linear",
        radiance_scale: float = 1.0,
        radiance_white_level: float = 1.0,
        radiance_log_threshold: float = 20.0,
        disocclusion_policy: str = "hold",
        warp_composite: str = "next_primary",
        warp_backend: str = "auto",
        camera_margin: CameraMargin = None,
        device: str = None,
        radiance_provider: Optional[Callable] = None,
        motion_state_provider: Optional[Callable] = None,
        frame_provider: Optional[Callable] = None,
        output_rgb_frames: bool = False,
    ):
        if input_space not in {"linear", "srgb"}:
            raise ValueError("input_space must be 'linear' or 'srgb'")
        self.output_dir = output_dir
        self.input_space = input_space
        self.radiance_provider = radiance_provider
        self.motion_state_provider = motion_state_provider
        self.frame_provider = frame_provider
        self.camera_profile = (
            get_camera_profile(camera_profile) if isinstance(camera_profile, str)
            else camera_profile
        )
        self.noise_config = replace(noise_config or get_preset(preset))
        self.noise_config.log_input = False
        self.noise_config.radiance_input = True
        self.noise_config.radiance_scale = float(radiance_scale)
        self.noise_config.radiance_white_level = float(radiance_white_level)
        self.noise_config.radiance_log_threshold = float(radiance_log_threshold)
        self.noise_config.validate()
        self.device = device
        self.sampler_config = sampler_config
        self.disocclusion_policy = disocclusion_policy
        self.warp_composite = warp_composite
        self.warp_backend = warp_backend
        if camera_margin is not None and not isinstance(camera_margin, CameraMargin):
            raise TypeError("camera_margin must be a CameraMargin instance")
        self.camera_margin = camera_margin
        self.output_rgb_frames = bool(output_rgb_frames)

        self.scene = None
        self.cam = None
        self.interpolator = None
        self.dvs_emulator = None
        self.recorder = None
        self._prev = None
        self._initialized = False
        self.total_steps = 0
        self.total_events = 0

    def _sensor_crop(self, value):
        if value is None or self.camera_margin is None:
            return value
        return crop_margin(value, self.camera_margin)

    def attach(self, scene, cam):
        if self.motion_state_provider is None:
            raise ValueError(
                "v2 requires motion_state_provider(scene, cam); provide complete "
                "camera_to_world and object_to_world poses explicitly"
            )
        self.scene, self.cam = scene, cam
        self.interpolator = PhysicsInterpolator(
            np.asarray(cam.intrinsics, dtype=np.float64),
            sampler_config=self.sampler_config,
            disocclusion_policy=self.disocclusion_policy,
            composite=self.warp_composite,
            device=self.device,
            warp_backend=self.warp_backend,
        )
        render_resolution_wh = (int(cam.res[0]), int(cam.res[1]))
        if self.camera_margin is not None:
            nominal_wh = (
                render_resolution_wh[0] - self.camera_margin.horizontal,
                render_resolution_wh[1] - self.camera_margin.vertical,
            )
            # Validate against the actual expanded render target.  A useful
            # over-render margin may be larger than the nominal sensor width;
            # the only invalid case is when it consumes the expanded target.
            self.camera_margin.validate_for(render_resolution_wh)
            if nominal_wh[0] <= 0 or nominal_wh[1] <= 0:
                raise ValueError("camera margin leaves no nominal sensor window")
        else:
            nominal_wh = render_resolution_wh
        resolution = (int(nominal_wh[1]), int(nominal_wh[0]))
        self.dvs_emulator = DvsEmulator(
            res=resolution, config=self.noise_config, device=self.device
        )
        if self.camera_profile is not None:
            for message in self.camera_profile.validate_genesis_camera(cam, strict=False):
                logger.warning("Camera profile check: %s", message)
        if self.output_dir is not None:
            self.recorder = GenesisEventRecorder(
                output_dir=self.output_dir,
                output_rgb_frames=self.output_rgb_frames,
                resolution=resolution,
                sensor_config=asdict(self.noise_config),
                camera_config={
                    "render_resolution_wh": list(render_resolution_wh),
                    "sensor_resolution_wh": list(nominal_wh),
                    "render_intrinsics": np.asarray(cam.intrinsics).tolist(),
                    "camera_margin": (
                        vars(self.camera_margin) if self.camera_margin is not None else None
                    ),
                    "input_space": self.input_space,
                    "radiance_scale": self.noise_config.radiance_scale,
                    "radiance_white_level": self.noise_config.radiance_white_level,
                },
            )
        return self

    def start_episode(self):
        if self.dvs_emulator is None:
            raise RuntimeError("call attach() before start_episode()")
        self.dvs_emulator.reset()
        self._prev = None
        self._initialized = False
        self.total_steps = 0
        self.total_events = 0
        if self.recorder is not None:
            self.recorder.start_episode()

    def _time(self) -> float:
        if hasattr(self.scene, "cur_t"):
            value = self.scene.cur_t
            if hasattr(value, "item"):
                value = value.item()
            return float(value)
        raise RuntimeError("Genesis scene must expose cur_t for physical timestamps")

    def _render(self):
        if self.frame_provider is not None:
            payload = self.frame_provider(self.scene, self.cam)
            if not isinstance(payload, Mapping):
                raise ValueError("frame_provider must return a mapping")
            if not {"depth", "seg"}.issubset(payload):
                raise ValueError("frame_provider must provide depth and seg")
            rgb = payload.get("rgb")
            depth = payload["depth"]
            seg = payload["seg"]
            radiance = payload.get("radiance")
            return _array(rgb), _array(depth), _array(seg), _array(radiance)
        rgb, depth, seg, _ = self.cam.render(
            rgb=True, depth=True, segmentation=True
        )
        rgb, depth, seg = _array(rgb), _array(depth), _array(seg)
        return rgb, depth, seg, None

    def _radiance(self, rgb, provided=None):
        if provided is not None:
            return prepare_linear_radiance(provided, input_space="linear")
        if self.radiance_provider is not None:
            value = self.radiance_provider(self.scene, self.cam)
            return prepare_linear_radiance(value, input_space="linear")
        if self.input_space == "linear":
            raise RuntimeError(
                "linear HDR mode requires radiance_provider; Genesis public "
                "Camera.render() currently exposes display RGB, not radiance"
            )
        return prepare_linear_radiance(rgb, input_space="srgb")

    def _motion_state(self):
        state = self.motion_state_provider(self.scene, self.cam)
        if not isinstance(state, Mapping):
            raise ValueError("motion_state_provider must return a mapping")
        camera = _pose(state["camera_to_world"], "camera_to_world")
        objects = {
            int(key): _pose(value, f"object_to_world[{key}]")
            for key, value in state.get("object_to_world", {}).items()
        }
        return {"camera_to_world": camera, "object_to_world": objects}

    @staticmethod
    def _object_deltas(previous: Mapping[int, np.ndarray], current: Mapping[int, np.ndarray]):
        return {
            key: current[key] @ np.linalg.inv(previous[key])
            for key in previous.keys() & current.keys()
        }

    def capture(self, action=None):
        if self.cam is None or self.dvs_emulator is None:
            raise RuntimeError("call attach() before capture()")
        rgb, depth, seg, provided_radiance = self._render()
        radiance = self._radiance(rgb, provided=provided_radiance)
        motion = self._motion_state()
        t_current = self._time()
        self.total_steps += 1

        if self._prev is None:
            self.dvs_emulator.initialize(self._sensor_crop(radiance), t_current)
            self._prev = {
                "radiance": radiance,
                "depth": depth,
                "seg": seg,
                "t": t_current,
                "motion": motion,
                "rgb": rgb,
            }
            self._initialized = True
            if self.recorder is not None and self.output_rgb_frames:
                self.recorder.record_rgb_frame(self._sensor_crop(rgb), t_current)
            return np.empty((0, 4), dtype=np.float64)

        object_delta = self._object_deltas(
            self._prev["motion"]["object_to_world"], motion["object_to_world"]
        )
        frames, timestamps = self.interpolator.interpolate(
            self._prev["radiance"], self._prev["depth"], self._prev["seg"],
            radiance, depth, seg,
            self._prev["motion"]["camera_to_world"], motion["camera_to_world"],
            object_delta, self._prev["t"], t_current, input_space="linear",
        )
        sensor_frames = [self._sensor_crop(frame) for frame in frames]
        chunks = [
            self.dvs_emulator.generate_events(frame, ts)
            for frame, ts in zip(sensor_frames, timestamps)
        ]
        events = (
            np.concatenate([chunk for chunk in chunks if len(chunk)], axis=0)
            if any(len(chunk) for chunk in chunks)
            else np.empty((0, 4), dtype=np.float64)
        )
        if len(events):
            events = events[np.argsort(events[:, 0])]
            self.total_events += len(events)
            if self.recorder is not None:
                self.recorder.record_aer_events(events)
        if self.recorder is not None:
            if self.output_rgb_frames:
                self.recorder.record_rgb_frame(self._sensor_crop(rgb), t_current)
            if action is not None:
                self.recorder.record_action(action)
        self._prev = {
            "radiance": radiance, "depth": depth, "seg": seg,
            "t": t_current, "motion": motion, "rgb": rgb,
        }
        return events

    def end_episode(self):
        if self.recorder is not None:
            self.recorder.end_episode()

    def close(self):
        if self.recorder is not None:
            self.recorder.close()
