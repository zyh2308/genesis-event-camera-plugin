"""Causal direct-HDR event input and a minimal event kinematics adapter.

The adapter consumes only timestamped pixel events and fixed camera calibration.
It does not receive target pose, segmentation, depth, or future simulator state.
Its centroid tracker is a plumbing smoke estimator, not a trained estimator.
"""

from __future__ import annotations

from collections import deque
from bisect import bisect_right
from dataclasses import dataclass
import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from benchmark.robot_closed_loop.predictor import MotionState


@dataclass
class EventTrackerSample:
    state: MotionState
    event_count: int
    time_bin_count: int
    generated_at_s: float


@dataclass(frozen=True)
class EventObservation:
    """A timestamped estimate plus explicit freshness and tracking health."""

    state_status: str
    tracking_status: str
    state: Optional[MotionState]
    age_s: Optional[float]
    sample_generated_at_s: Optional[float]
    event_count_window: int
    time_bin_count_window: int


class EventKinematicsTracker:
    """Short-window event centroid regression with fixed workspace calibration."""

    def __init__(
        self,
        camera_to_world_cv: np.ndarray,
        resolution_wh: Sequence[int],
        vertical_fov_deg: float,
        workspace_x_m: Sequence[float],
        workspace_y_m: Sequence[float],
        target_center_z_m: float,
        *,
        window_s: float = 0.030,
        bin_s: float = 0.005,
        max_state_age_s: float = 0.030,
    ):
        self.camera_to_world = np.asarray(camera_to_world_cv, dtype=np.float64).reshape(4, 4)
        self.world_to_camera = np.linalg.inv(self.camera_to_world)
        self.width, self.height = (int(v) for v in resolution_wh)
        fy = self.height / (2.0 * math.tan(math.radians(float(vertical_fov_deg)) / 2.0))
        self.fx = fy
        self.fy = fy
        # Genesis Camera.intrinsics uses the continuous image center (W/2,H/2),
        # matching its rasterizer projection. Pixel indices are integer
        # centers only after adding the half-pixel offset during rasterization.
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0
        self.target_center_z_m = float(target_center_z_m)
        self.window_s = float(window_s)
        self.bin_s = float(bin_s)
        self.max_state_age_s = float(max_state_age_s)
        corners = []
        for x in workspace_x_m:
            for y in workspace_y_m:
                for z in (0.0, self.target_center_z_m * 2.0):
                    corners.append(self._project_world(np.asarray([x, y, z], dtype=np.float64)))
        uv = np.asarray(corners, dtype=np.float64)
        self.roi = (
            max(0, int(np.floor(np.min(uv[:, 0]) - 8))),
            min(self.width, int(np.ceil(np.max(uv[:, 0]) + 9))),
            max(0, int(np.floor(np.min(uv[:, 1]) - 8))),
            min(self.height, int(np.ceil(np.max(uv[:, 1]) + 9))),
        )
        self._events: deque[Tuple[float, float, float, int]] = deque()
        self.samples: List[EventTrackerSample] = []
        self._sample_generated_times: List[float] = []
        self._confidence_times: List[float] = []
        self._confidence_values: List[Tuple[int, int]] = []
        self.last_confidence = {"event_count": 0, "time_bin_count": 0}

    def _record_confidence(self, now_s: float, event_count: int, time_bin_count: int) -> None:
        self.last_confidence = {
            "event_count": int(event_count), "time_bin_count": int(time_bin_count),
        }
        now = float(now_s)
        if not self._confidence_times or now > self._confidence_times[-1] + 1e-12:
            self._confidence_times.append(now)
            self._confidence_values.append((int(event_count), int(time_bin_count)))
        else:
            self._confidence_values[-1] = (int(event_count), int(time_bin_count))

    def _project_world(self, point_world: np.ndarray) -> np.ndarray:
        point_camera = self.world_to_camera @ np.append(point_world, 1.0)
        z = float(point_camera[2])
        if z <= 1e-6:
            return np.asarray([self.width / 2.0, self.height / 2.0])
        return np.asarray([
            self.fx * point_camera[0] / z + self.cx,
            self.fy * point_camera[1] / z + self.cy,
        ])

    def _event_to_world(self, x_px: float, y_px: float) -> Optional[np.ndarray]:
        ray_camera = np.asarray([
            (float(x_px) - self.cx) / self.fx,
            (float(y_px) - self.cy) / self.fy,
            1.0,
        ])
        direction_world = self.camera_to_world[:3, :3] @ ray_camera
        origin_world = self.camera_to_world[:3, 3]
        if abs(direction_world[2]) < 1e-9:
            return None
        scale = (self.target_center_z_m - origin_world[2]) / direction_world[2]
        if scale <= 0:
            return None
        return origin_world + scale * direction_world

    def push(self, events: np.ndarray, now_s: float) -> Optional[EventTrackerSample]:
        arr = np.asarray(events, dtype=np.float64).reshape(-1, 4)
        x0, x1, y0, y1 = self.roi
        if len(arr):
            # DvsEmulator currently emits sorted timestamps. Sorting here also
            # keeps deque expiry correct for other causal event sources.
            arr = arr[np.argsort(arr[:, 0], kind="stable")]
            valid = (
                (arr[:, 1] >= x0) & (arr[:, 1] < x1)
                & (arr[:, 2] >= y0) & (arr[:, 2] < y1)
            )
            for event in arr[valid]:
                self._events.append((float(event[0]), float(event[1]), float(event[2]), int(event[3])))
        cutoff = float(now_s) - self.window_s
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()
        if len(self._events) < 12:
            self._record_confidence(now_s, len(self._events), 0)
            return None

        bins: Dict[int, List[Tuple[float, np.ndarray]]] = {}
        for t, x, y, _polarity in self._events:
            point = self._event_to_world(x, y)
            if point is None:
                continue
            key = int(math.floor(t / self.bin_s))
            bins.setdefault(key, []).append((t, point[:2]))
        valid_bins = sorted(key for key, points in bins.items() if len(points) >= 3)
        self._record_confidence(now_s, len(self._events), len(valid_bins))
        if len(valid_bins) < 3:
            return None

        # Use each bin's actual event-time median. The position is also a
        # median over that bin, so assigning the bin's right edge made the
        # state timestamp up to one full bin newer than its measurements.
        times = np.asarray([
            np.median([sample[0] for sample in bins[key]]) for key in valid_bins
        ], dtype=np.float64)
        centers = np.asarray([
            np.median(np.stack([sample[1] for sample in bins[key]]), axis=0)
            for key in valid_bins
        ], dtype=np.float64)

        # Least-squares slope requires time and position to use the same
        # centering. Subtracting the last timestamp while centering positions
        # on their mean inflated the denominator and returned 0.12727 m/s for
        # a six-bin 0.4 m/s linear trajectory.
        centered_t = times - times.mean()
        denom = float(np.dot(centered_t, centered_t))
        velocity_xy = np.zeros(2, dtype=np.float64) if denom < 1e-12 else centered_t @ centers / denom
        latest_t = min(float(now_s), float(times[-1]))
        state = MotionState(
            latest_t,
            np.asarray([centers[-1, 0], centers[-1, 1], self.target_center_z_m], dtype=np.float64),
            np.asarray([velocity_xy[0], velocity_xy[1], 0.0], dtype=np.float64),
        )
        sample = EventTrackerSample(
            state, len(self._events), len(valid_bins), float(now_s),
        )
        if not self.samples or float(now_s) > self.samples[-1].generated_at_s + 1e-9:
            self.samples.append(sample)
            self._sample_generated_times.append(float(now_s))
        return sample

    def observe(self, cutoff_s: float, now_s: Optional[float] = None) -> EventObservation:
        # Causality is determined by when this regression was computed, not
        # only by the median timestamp attached to its position.
        sample_index = bisect_right(
            self._sample_generated_times, float(cutoff_s) + 1e-12,
        ) - 1
        selected = None if sample_index < 0 else self.samples[sample_index]

        now = float(cutoff_s if now_s is None else now_s)
        confidence_index = bisect_right(
            self._confidence_times, float(cutoff_s) + 1e-12,
        ) - 1
        if confidence_index < 0:
            event_count, time_bin_count = 0, 0
        else:
            event_count, time_bin_count = self._confidence_values[confidence_index]
        event_count = int(event_count)
        time_bin_count = int(time_bin_count)
        if event_count == 0:
            tracking_status = "no_event"
        elif event_count < 12 or time_bin_count < 3:
            tracking_status = "tracking_lost"
        else:
            tracking_status = "tracking"

        age = None if selected is None else max(0.0, now - selected.state.time_s)
        if selected is not None and age > self.max_state_age_s:
            return EventObservation(
                "stale", tracking_status, None, age, selected.generated_at_s,
                event_count, time_bin_count,
            )
        if tracking_status == "no_event":
            return EventObservation(
                "no_event", tracking_status, None, age,
                None if selected is None else selected.generated_at_s,
                event_count, time_bin_count,
            )
        if tracking_status == "tracking_lost":
            return EventObservation(
                "tracking_lost", tracking_status, None, age,
                None if selected is None else selected.generated_at_s,
                event_count, time_bin_count,
            )
        if selected is None:
            return EventObservation(
                "unavailable", tracking_status, None, None, None,
                event_count, time_bin_count,
            )
        state = selected.state
        return EventObservation(
            "fresh", tracking_status,
            MotionState(state.time_s, state.position_m.copy(), state.velocity_m_s.copy()),
            age, selected.generated_at_s, event_count, time_bin_count,
        )

    def latest(self) -> Optional[MotionState]:
        if not self.samples:
            return None
        sample = self.samples[-1].state
        return MotionState(sample.time_s, sample.position_m.copy(), sample.velocity_m_s.copy())


class DirectHdrEventSource:
    """Render linear HDR at physics cadence and emit paired 1/2 kHz event streams."""

    def __init__(self, config: Dict, physics_dt_s: float, controller_hz: int = 2000):
        self.config = config
        self.physics_dt_s = float(physics_dt_s)
        self.controller_hz = int(controller_hz)
        if abs(self.physics_dt_s - 0.0005) > 1e-9:
            raise ValueError("paired 1/2 kHz reference requires a 0.5 ms physics base step")
        if self.controller_hz not in {1000, 2000}:
            raise ValueError("event controller sampling rate must be 1000 or 2000 Hz")
        from genesis_event_plugin.dvs_emulator import DvsEmulator
        from genesis_event_plugin.genesis_adapter import camera_to_world_cv
        from genesis_event_plugin.noise_model import DvsNoiseConfig

        h, w = (int(v) for v in config["camera"]["resolution_wh"][::-1])
        response = DvsNoiseConfig(
            pos_thres=0.2, neg_thres=0.2, sigma_thres=0.0,
            cutoff_hz=0.0, leak_rate_hz=0.0, leak_jitter_fraction=0.0,
            shot_noise_rate_hz=0.0, refractory_period_s=0.0,
            log_input=True, readout_model="ideal_readout",
        )
        self.emulators = {
            hz: DvsEmulator(res=(h, w), config=response, device="cpu")
            for hz in (1000, 2000)
        }
        self.events: Dict[int, List[np.ndarray]] = {1000: [], 2000: []}
        self.trackers: Dict[int, EventKinematicsTracker] = {}
        self.initialized = False
        self.render_ms: List[float] = []
        self.dvs_ms: Dict[int, List[float]] = {1000: [], 2000: []}
        self.render_cold_ms = None
        self.render_calls = 0
        self.frames_processed = 0
        self.last_hdr = None
        self.snapshots: Dict[str, Tuple[float, np.ndarray]] = {}
        self.camera_to_world_cv = None
        self.camera_intrinsics = None

    def initialize(self, camera: object, t_s: float = 0.0) -> None:
        from genesis_event_plugin.genesis_adapter import camera_to_world_cv

        started = time.perf_counter()
        hdr, _, _, _ = camera.render(rgb=True, depth=False, segmentation=False, radiance=True)
        elapsed = (time.perf_counter() - started) * 1000.0
        self.render_ms.append(elapsed)
        self.render_cold_ms = elapsed
        self.render_calls += 1
        self.last_hdr = np.asarray(hdr, dtype=np.float32)
        log_gray = _hdr_to_log_gray(hdr)
        for emulator in self.emulators.values():
            emulator.initialize(log_gray, t=float(t_s))
        camera_pose = camera_to_world_cv(camera)
        self.camera_to_world_cv = np.asarray(camera_pose, dtype=np.float64).copy()
        cfg = self.config
        width, height = (int(v) for v in cfg["camera"]["resolution_wh"])
        focal = height / (2.0 * math.tan(math.radians(float(cfg["camera"]["vertical_fov_deg"])) / 2.0))
        self.camera_intrinsics = {
            "fx": focal, "fy": focal, "cx": width / 2.0, "cy": height / 2.0,
        }
        x_range = cfg["target"]["launch_x_range_m"]
        x_min = float(cfg["controller"]["intercept_x_m"]) - 0.10
        x_max = float(x_range[1]) + 0.04
        y_limit = max(abs(float(v)) for v in cfg["target"]["launch_y_range_m"]) + 0.06
        common = (
            camera_pose, cfg["camera"]["resolution_wh"], cfg["camera"]["vertical_fov_deg"],
            (x_min, x_max), (-y_limit, y_limit), float(cfg["target"]["size_m"][2]) / 2,
        )
        maximum_age = float(cfg["controller"].get("maximum_event_state_age_s", 0.030))
        self.trackers = {
            hz: EventKinematicsTracker(*common, max_state_age_s=maximum_age)
            for hz in (1000, 2000)
        }
        self.initialized = True

    def advance(self, camera: object, t_s: float) -> None:
        if not self.initialized:
            self.initialize(camera, t_s)
            return
        started = time.perf_counter()
        hdr, _, _, _ = camera.render(rgb=True, depth=False, segmentation=False, radiance=True)
        elapsed = (time.perf_counter() - started) * 1000.0
        self.render_ms.append(elapsed)
        self.render_calls += 1
        self.last_hdr = np.asarray(hdr, dtype=np.float32)
        log_gray = _hdr_to_log_gray(hdr)
        self.frames_processed += 1
        start_dvs = time.perf_counter()
        events_2k = self.emulators[2000].generate_events(log_gray, t=float(t_s))
        self.dvs_ms[2000].append((time.perf_counter() - start_dvs) * 1000.0)
        if len(events_2k):
            self.events[2000].append(events_2k)
        self.trackers[2000].push(events_2k, float(t_s))

        if self.frames_processed % 2 == 0:
            start_dvs = time.perf_counter()
            events_1k = self.emulators[1000].generate_events(log_gray, t=float(t_s))
            self.dvs_ms[1000].append((time.perf_counter() - start_dvs) * 1000.0)
            if len(events_1k):
                self.events[1000].append(events_1k)
            self.trackers[1000].push(events_1k, float(t_s))

    def observe(self, cutoff_s: float, now_s: Optional[float] = None) -> EventObservation:
        return self.trackers[self.controller_hz].observe(cutoff_s, now_s)

    def latest(self, rate_hz: int) -> Optional[MotionState]:
        return self.trackers[int(rate_hz)].latest()

    def capture_snapshot(self, label: str, time_s: float) -> None:
        if self.last_hdr is not None and label not in self.snapshots:
            self.snapshots[label] = (float(time_s), self.last_hdr.copy())

    def save(self, out_dir) -> Dict[str, object]:
        from pathlib import Path

        out = Path(out_dir)
        arrays = {}
        for hz, batches in self.events.items():
            events = np.concatenate(batches, axis=0) if batches else np.empty((0, 4), dtype=np.float64)
            arrays[f"events_{hz}hz"] = events
        np.savez_compressed(out / "direct_hdr_events_1khz_2khz.npz", **arrays)
        snapshot_items = list(self.snapshots.items())
        if snapshot_items:
            np.savez_compressed(
                out / "direct_hdr_key_snapshots.npz",
                labels=np.asarray([label for label, _ in snapshot_items]),
                times_s=np.asarray([item[0] for _, item in snapshot_items], dtype=np.float64),
                hdr=np.stack([item[1] for _, item in snapshot_items]),
            )
            try:
                from PIL import Image
                for label, (_time_s, image) in snapshot_items:
                    rgb = np.clip(image / (1.0 + image), 0.0, 1.0)
                    rgb = np.power(rgb, 1.0 / 2.2)
                    Image.fromarray(np.asarray(rgb * 255.0, dtype=np.uint8)).save(out / f"direct_hdr_{label}.png")
            except ImportError:
                pass
        return {
            "controller_sampling_hz": self.controller_hz,
            "sampling_hz": [1000, 2000],
            "physics_dt_s": self.physics_dt_s,
            "render_calls": self.render_calls,
            "event_frame_count": self.frames_processed,
            "event_count_by_rate": {str(hz): int(sum(len(a) for a in batches)) for hz, batches in self.events.items()},
            "cold_first_render_ms": self.render_cold_ms,
            "steady_render_mean_ms": float(np.mean(self.render_ms[1:])) if len(self.render_ms) > 1 else None,
            "steady_render_p50_ms": float(np.median(self.render_ms[1:])) if len(self.render_ms) > 1 else None,
            "steady_render_p95_ms": float(np.percentile(self.render_ms[1:], 95)) if len(self.render_ms) > 1 else None,
            "dvs_mean_ms_per_sample": {str(hz): float(np.mean(v)) if v else None for hz, v in self.dvs_ms.items()},
            "dvs_p95_ms_per_sample": {str(hz): float(np.percentile(v, 95)) if v else None for hz, v in self.dvs_ms.items()},
            "event_source": "log-linear-HDR direct render → shared genesis_event_plugin.DvsEmulator",
            "raw_event_schema": "[t_seconds_from_launch, x_pixel, y_pixel, polarity(+1 ON, -1 OFF)]",
            "pixel_response": {
                "on_threshold": 0.2, "off_threshold": 0.2, "threshold_std": 0.0,
                "noise": False, "bandwidth": "infinite", "readout_model": "ideal_readout",
                "shared_pixel_response_model_with_plugin": True,
            },
            "estimator_boundary": "events plus fixed pinhole calibration only; no GT pose, target mask, depth, or future state",
            "estimator_status": "causal centroid/regression plumbing smoke; not a trained or validated perception model",
            "state_freshness_policy": {
                "maximum_age_s": self.trackers[self.controller_hz].max_state_age_s,
                "no_event": "return no usable state and hold current lateral command",
                "tracking_lost": "return no usable state and hold current lateral command",
                "stale": "return no usable state and hold current lateral command",
                "closure": "do not initiate without a fresh state; if already started, keep fingers closed",
                "lift_gate": "requires a fresh state at the last high-level control update",
            },
            "camera_to_world_cv": None if self.camera_to_world_cv is None else self.camera_to_world_cv.tolist(),
            "camera_intrinsics_fx_fy_cx_cy": None if self.camera_intrinsics is None else [
                self.camera_intrinsics["fx"], self.camera_intrinsics["fy"],
                self.camera_intrinsics["cx"], self.camera_intrinsics["cy"],
            ],
        }


def _hdr_to_log_gray(hdr: np.ndarray) -> np.ndarray:
    rgb = np.asarray(hdr, dtype=np.float32)
    gray = np.clip(rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722, 0.0, None)
    return np.log(gray.astype(np.float64) + 1e-6)


def _voxelize(events: np.ndarray, width: int, height: int, bins: int = 64, bin_s: float = 0.010) -> Dict[int, np.ndarray]:
    voxels: Dict[int, np.ndarray] = {}
    for t, x, y, polarity in np.asarray(events, dtype=np.float64).reshape(-1, 4):
        key = int(math.floor(float(t) / bin_s))
        voxel = voxels.setdefault(key, np.zeros((2, bins, bins), dtype=np.float32))
        xx = min(bins - 1, max(0, int(float(x) * bins / width)))
        yy = min(bins - 1, max(0, int(float(y) * bins / height)))
        voxel[0 if polarity > 0 else 1, yy, xx] += 1.0
    return voxels


def voxel_convergence_summary(events_1k: np.ndarray, events_2k: np.ndarray, resolution_wh: Sequence[int]) -> Dict[str, object]:
    width, height = (int(v) for v in resolution_wh)
    vox_1k = _voxelize(events_1k, width, height)
    vox_2k = _voxelize(events_2k, width, height)
    common = sorted(set(vox_1k).intersection(vox_2k))
    cosines = []
    relative_counts = []
    for key in common:
        a, b = vox_1k[key].ravel(), vox_2k[key].ravel()
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom > 0:
            cosines.append(float(np.dot(a, b) / denom))
        total_a, total_b = float(a.sum()), float(b.sum())
        if max(total_a, total_b) > 0:
            relative_counts.append(abs(total_a - total_b) / max(total_a, total_b))
    return {
        "voxel_shape": [2, 64, 64], "temporal_bin_ms": 10.0,
        "common_nonempty_bins": len(common),
        "mean_same_bin_voxel_cosine": None if not cosines else float(np.mean(cosines)),
        "p10_same_bin_voxel_cosine": None if not cosines else float(np.percentile(cosines, 10)),
        "mean_same_bin_event_count_relative_difference": None if not relative_counts else float(np.mean(relative_counts)),
        "status": "paired 1/2 kHz voxels made from the same 0.5 ms physical/HDR trajectory",
    }
