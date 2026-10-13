"""Frozen, auditable configuration for all EventGenesis experiments.

The configuration is intentionally independent from the simulator core.  It
is converted into the existing ``GenesisPhysicsEventPlugin`` constructor
arguments at the runner boundary, so a hidden experiment parameter cannot be
introduced by a second generation implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import yaml

from genesis_event_plugin.adaptive_sampler import AdaptiveSamplingConfig
from genesis_event_plugin.camera_margin import CameraMargin
from genesis_event_plugin.noise_model import DvsNoiseConfig, get_preset


FROZEN_CORE_COMMIT = "e79e656ac00f7847b98db9e12bf43b093f3afff1"
CONFIG_SCHEMA_VERSION = "eventgenesis-config-v1"


def _tuple2(value: Any, name: str) -> Tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must contain exactly two integers")
    result = tuple(int(v) for v in value)
    if min(result) <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _reject_unknown(mapping: Mapping[str, Any], allowed: set[str], section: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"unknown {section} configuration key(s): {', '.join(unknown)}")


@dataclass(frozen=True)
class SensorConfig:
    resolution_xy: Tuple[int, int] = (1280, 720)
    preset: str = "real_v2"
    pos_thres: float = 0.205
    neg_thres: float = 0.195
    sigma_thres: float = 0.04
    cutoff_hz: float = 50.0
    leak_rate_hz: float = 0.0
    leak_jitter_fraction: float = 0.0
    shot_noise_rate_hz: float = 0.0
    shot_noise_inten_factor: float = 0.25
    depth_noise_scale: float = 0.0
    photoreceptor_noise: bool = False
    noise_rate_cov_decades: float = 0.0
    refractory_period_s: float = 0.0
    hot_pixel_rate_hz: float = 0.0
    pixel_noise_map_path: Optional[str] = None
    correlated_noise_alpha: float = 0.701838977169265
    correlated_noise_scale: float = 0.016
    seed: int = 42
    readout_model: str = "ideal_readout"

    def to_noise_config(self) -> DvsNoiseConfig:
        # Start from the named preset for calibrated ancillary fields (for
        # example the project noise map), then overwrite every exposed value.
        cfg = get_preset(self.preset)
        values = {
            "pos_thres": self.pos_thres,
            "neg_thres": self.neg_thres,
            "sigma_thres": self.sigma_thres,
            "cutoff_hz": self.cutoff_hz,
            "leak_rate_hz": self.leak_rate_hz,
            "leak_jitter_fraction": self.leak_jitter_fraction,
            "shot_noise_rate_hz": self.shot_noise_rate_hz,
            "shot_noise_inten_factor": self.shot_noise_inten_factor,
            "depth_noise_scale": self.depth_noise_scale,
            "photoreceptor_noise": self.photoreceptor_noise,
            "noise_rate_cov_decades": self.noise_rate_cov_decades,
            "refractory_period_s": self.refractory_period_s,
            "hot_pixel_rate_hz": self.hot_pixel_rate_hz,
            "correlated_noise_alpha": self.correlated_noise_alpha,
            "correlated_noise_scale": self.correlated_noise_scale,
            "seed": self.seed,
            "readout_model": self.readout_model,
        }
        if self.pixel_noise_map_path is not None:
            values["pixel_noise_map_path"] = self.pixel_noise_map_path
        for key, value in values.items():
            setattr(cfg, key, value)
        cfg.validate()
        return cfg


@dataclass(frozen=True)
class RadiometryConfig:
    input_space: str = "linear"
    response_mode: str = "physical_log"
    radiance_scale: float = 1.0
    white_level: float = 1.0
    log_threshold: float = 20.0
    gain: float = 1.0
    offset: float = 0.0
    floor: float = 1e-6
    calibration_status: str = "uncalibrated"
    calibration_id: str = ""
    calibrated_transfer_path: Optional[str] = None


@dataclass(frozen=True)
class ReconstructionConfig:
    source_dt_s: float = 0.01
    contrast_threshold: float = 0.2
    safety_factor: float = 0.5
    min_samples: int = 2
    max_samples: int = 300
    flow_percentile: float = 100.0
    max_boundary_motion_px: float = 1.0
    max_relative_depth_change: float = 0.1
    max_camera_motion_px: float = 1.0
    max_articulation_motion_px: float = 1.0
    radiance_residual_threshold: float = 0.1
    sensor_eps_target: float = 0.05
    max_disocclusion_fraction: float = 0.0
    visibility_guard: bool = True
    strict_sampler: bool = True
    disocclusion_policy: str = "hold"
    warp_composite: str = "next_primary"
    warp_backend: str = "numpy"
    strict_fidelity: bool = False
    camera_margin_xy: Tuple[int, int] = (0, 0)

    def to_sampler_config(self, cutoff_hz: float) -> AdaptiveSamplingConfig:
        tau = 1.0 / (2.0 * 3.141592653589793 * cutoff_hz) if cutoff_hz > 0 else None
        return AdaptiveSamplingConfig(
            contrast_threshold=self.contrast_threshold,
            safety_factor=self.safety_factor,
            min_samples=self.min_samples,
            max_samples=self.max_samples,
            flow_percentile=self.flow_percentile,
            max_boundary_motion_px=self.max_boundary_motion_px,
            max_relative_depth_change=self.max_relative_depth_change,
            max_camera_motion_px=self.max_camera_motion_px,
            max_articulation_motion_px=self.max_articulation_motion_px,
            radiance_residual_threshold=self.radiance_residual_threshold,
            sensor_eps_target=self.sensor_eps_target,
            sensor_time_constant_s=tau,
            max_disocclusion_fraction=self.max_disocclusion_fraction,
            use_visibility_guard=self.visibility_guard,
            strict=self.strict_sampler,
        )


@dataclass(frozen=True)
class SceneConfig:
    resolution_xy: Tuple[int, int] = (1280, 720)
    camera_profile: str = "evk4_hd"
    camera_id: str = "camera0"
    object_ids: Tuple[int, ...] = field(default_factory=tuple)
    robot_link_ids: Tuple[int, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class OutputConfig:
    timestamp_unit: str = "s"
    event_timestamp_unit: str = "us"
    event_layout: str = "t,x,y,p_01"
    output_rgb_frames: bool = False


@dataclass(frozen=True)
class EventGenesisConfig:
    schema_version: str = CONFIG_SCHEMA_VERSION
    core_commit: str = FROZEN_CORE_COMMIT
    genesis_version: str = "1.2.2"
    seed: int = 42
    sensor: SensorConfig = field(default_factory=SensorConfig)
    radiometry: RadiometryConfig = field(default_factory=RadiometryConfig)
    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def validate(self) -> None:
        if self.schema_version != CONFIG_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        if self.core_commit != FROZEN_CORE_COMMIT:
            raise ValueError(
                "formal experiments must pin the frozen core commit "
                f"{FROZEN_CORE_COMMIT}; got {self.core_commit}"
            )
        if self.radiometry.input_space not in {"linear", "srgb"}:
            raise ValueError("radiometry.input_space must be linear or srgb")
        if self.radiometry.response_mode not in {"physical_log", "video_linlog", "calibrated_transfer"}:
            raise ValueError("unsupported radiometry.response_mode")
        if self.radiometry.response_mode == "calibrated_transfer" and not self.radiometry.calibrated_transfer_path:
            raise ValueError("calibrated_transfer requires a transfer path")
        if self.reconstruction.source_dt_s <= 0:
            raise ValueError("reconstruction.source_dt_s must be positive")
        if self.output.timestamp_unit != "s" or self.output.event_timestamp_unit != "us":
            raise ValueError("the v1 sidecar contract uses seconds and AER microseconds")
        _tuple2(self.sensor.resolution_xy, "sensor.resolution_xy")
        _tuple2(self.scene.resolution_xy, "scene.resolution_xy")
        self.sensor.to_noise_config()
        self.reconstruction.to_sampler_config(self.sensor.cutoff_hz).validate()

    def camera_margin(self) -> Optional[CameraMargin]:
        h, v = self.reconstruction.camera_margin_xy
        return None if h == 0 and v == 0 else CameraMargin(
            left=int(h) // 2,
            right=int(h) - int(h) // 2,
            top=int(v) // 2,
            bottom=int(v) - int(v) // 2,
        )

    def plugin_kwargs(self, *, output_dir: Optional[str], radiance_provider=None,
                      motion_state_provider=None, frame_provider=None) -> Dict[str, Any]:
        self.validate()
        return {
            "output_dir": output_dir,
            "preset": self.sensor.preset,
            "noise_config": self.sensor.to_noise_config(),
            "camera_profile": self.scene.camera_profile,
            "sampler_config": self.reconstruction.to_sampler_config(self.sensor.cutoff_hz),
            "input_space": self.radiometry.input_space,
            "radiance_scale": self.radiometry.radiance_scale,
            "radiance_white_level": self.radiometry.white_level,
            "radiance_log_threshold": self.radiometry.log_threshold,
            "response_mode": self.radiometry.response_mode,
            "radiometric_gain": self.radiometry.gain,
            "radiometric_offset": self.radiometry.offset,
            "radiometric_floor": self.radiometry.floor,
            "radiometric_calibration_status": self.radiometry.calibration_status,
            "radiometric_calibration_id": self.radiometry.calibration_id,
            "calibrated_transfer_path": self.radiometry.calibrated_transfer_path,
            "sensor_eps_target": self.reconstruction.sensor_eps_target,
            "readout_model": self.sensor.readout_model,
            "disocclusion_policy": self.reconstruction.disocclusion_policy,
            "warp_composite": self.reconstruction.warp_composite,
            "warp_backend": self.reconstruction.warp_backend,
            "camera_margin": self.camera_margin(),
            "strict_fidelity": self.reconstruction.strict_fidelity,
            "radiance_provider": radiance_provider,
            "motion_state_provider": motion_state_provider,
            "frame_provider": frame_provider,
            "output_rgb_frames": self.output.output_rgb_frames,
        }


def _build(data: Mapping[str, Any]) -> EventGenesisConfig:
    _reject_unknown(data, {"schema_version", "core_commit", "genesis_version", "seed", "sensor", "radiometry", "reconstruction", "scene", "output"}, "root")
    sensor_data = dict(data.get("sensor", {}))
    _reject_unknown(sensor_data, set(SensorConfig.__dataclass_fields__), "sensor")
    if "resolution_xy" in sensor_data:
        sensor_data["resolution_xy"] = _tuple2(sensor_data["resolution_xy"], "sensor.resolution_xy")
    sensor = SensorConfig(**sensor_data)
    radio_data = dict(data.get("radiometry", {}))
    _reject_unknown(radio_data, set(RadiometryConfig.__dataclass_fields__), "radiometry")
    radiometry = RadiometryConfig(**radio_data)
    recon_data = dict(data.get("reconstruction", {}))
    _reject_unknown(recon_data, set(ReconstructionConfig.__dataclass_fields__), "reconstruction")
    if "camera_margin_xy" in recon_data:
        margin = recon_data["camera_margin_xy"]
        if not isinstance(margin, (list, tuple)) or len(margin) != 2:
            raise ValueError("reconstruction.camera_margin_xy must contain two integers")
        margin = tuple(int(v) for v in margin)
        if min(margin) < 0:
            raise ValueError("reconstruction.camera_margin_xy must be non-negative")
        recon_data["camera_margin_xy"] = margin
    reconstruction = ReconstructionConfig(**recon_data)
    scene_data = dict(data.get("scene", {}))
    _reject_unknown(scene_data, set(SceneConfig.__dataclass_fields__), "scene")
    for key in ("resolution_xy",):
        if key in scene_data:
            scene_data[key] = _tuple2(scene_data[key], f"scene.{key}")
    for key in ("object_ids", "robot_link_ids"):
        if key in scene_data:
            scene_data[key] = tuple(int(v) for v in scene_data[key])
    scene = SceneConfig(**scene_data)
    output_data = dict(data.get("output", {}))
    _reject_unknown(output_data, set(OutputConfig.__dataclass_fields__), "output")
    output = OutputConfig(**output_data)
    config = EventGenesisConfig(
        schema_version=data.get("schema_version", CONFIG_SCHEMA_VERSION),
        core_commit=data.get("core_commit", FROZEN_CORE_COMMIT),
        genesis_version=str(data.get("genesis_version", "1.2.2")),
        seed=int(data.get("seed", 42)),
        sensor=sensor,
        radiometry=radiometry,
        reconstruction=reconstruction,
        scene=scene,
        output=output,
    )
    config.validate()
    return config


def load_config(path: str | Path) -> EventGenesisConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, Mapping):
        raise ValueError("configuration root must be a mapping")
    return _build(data)


def config_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_to_dict(config: EventGenesisConfig) -> Dict[str, Any]:
    """Return JSON-safe config values for provenance manifests."""
    def convert(value):
        if hasattr(value, "__dataclass_fields__"):
            return {k: convert(getattr(value, k)) for k in value.__dataclass_fields__}
        if isinstance(value, tuple):
            return [convert(v) for v in value]
        return value
    return convert(config)
