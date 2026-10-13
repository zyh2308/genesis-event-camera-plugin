"""One capture runner shared by Quality, Sim-to-Real, and Robot experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from genesis_event_plugin.physics_plugin import GenesisPhysicsEventPlugin

from .adapter import ExperimentAdapter
from .config import EventGenesisConfig
from .export import DatasetExporter


def _scene_time(scene) -> float:
    value = getattr(scene, "cur_t", None)
    if value is None:
        raise RuntimeError("Genesis scene must expose cur_t for synchronized timestamps")
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


class ExperimentRunner:
    """Bind one adapter to the frozen v3 plugin and a reproducible dataset.

    The runner does not step Genesis.  The owning task/environment controls
    physics and calls :meth:`capture` once per endpoint, so Quality and Robot
    code can use their native episode loops while sharing exactly this event
    generation path.
    """

    def __init__(self, config: EventGenesisConfig, adapter: ExperimentAdapter,
                 *, dataset_root: Optional[str | Path] = None,
                 config_path: Optional[str | Path] = None,
                 run_id: str = "run-0000"):
        config.validate()
        if not isinstance(adapter, ExperimentAdapter):
            raise TypeError("adapter must be an ExperimentAdapter")
        self.config = config
        self.adapter = adapter
        self.exporter = (
            DatasetExporter(dataset_root, config, config_path=config_path,
                            experiment_kind=adapter.kind, run_id=run_id)
            if dataset_root is not None else None
        )
        self.scene = None
        self.camera = None
        self.plugin = None
        self.episode_index = -1
        self.frame_index = 0
        self._previous_timestamp_s = None
        self._closed = False

    def attach(self, scene, camera):
        if self._closed:
            raise RuntimeError("runner is closed")
        self.scene, self.camera = scene, camera
        expected_render = tuple(int(v) for v in self.config.scene.resolution_xy)
        actual_render = tuple(int(v) for v in camera.res)
        if actual_render != expected_render:
            raise ValueError(
                "camera resolution does not match frozen scene.resolution_xy: "
                f"expected {expected_render}, got {actual_render}"
            )
        margin = self.config.camera_margin()
        if margin is None:
            expected_sensor = actual_render
        else:
            expected_sensor = (
                actual_render[0] - margin.horizontal,
                actual_render[1] - margin.vertical,
            )
        if expected_sensor != tuple(self.config.sensor.resolution_xy):
            raise ValueError(
                "sensor.resolution_xy is inconsistent with camera resolution and "
                f"margin: expected {expected_sensor}, got {self.config.sensor.resolution_xy}"
            )
        output_dir = self.exporter.plugin_output_dir if self.exporter else None
        kwargs = self.config.plugin_kwargs(
            output_dir=output_dir,
            radiance_provider=None,
            motion_state_provider=self.adapter.motion_state_provider,
            frame_provider=self.adapter.frame_provider,
        )
        self.plugin = GenesisPhysicsEventPlugin(**kwargs)
        self.plugin.attach(scene, camera)
        if self.exporter:
            self.exporter.write_metadata(
                scene={"camera_id": self.config.scene.camera_id},
                extra={"adapter_kind": self.adapter.kind},
            )
        return self

    def start_episode(self) -> None:
        if self.plugin is None:
            raise RuntimeError("call attach() before start_episode()")
        self.plugin.start_episode()
        self.episode_index += 1
        self.frame_index = 0
        self._previous_timestamp_s = None

    def capture(self, action=None) -> np.ndarray:
        if self.plugin is None or self.scene is None or self.camera is None:
            raise RuntimeError("call attach() before capture()")
        if self.episode_index < 0:
            raise RuntimeError("call start_episode() before capture()")
        timestamp_s = _scene_time(self.scene)
        if self._previous_timestamp_s is not None:
            observed_dt = timestamp_s - self._previous_timestamp_s
            tolerance = max(1e-9, 1e-6 * self.config.reconstruction.source_dt_s)
            if abs(observed_dt - self.config.reconstruction.source_dt_s) > tolerance:
                raise ValueError(
                    "scene endpoint interval differs from frozen source_dt_s: "
                    f"expected {self.config.reconstruction.source_dt_s}, got {observed_dt}"
                )
        start = int(self.plugin.total_events)
        effective_action = self.adapter.action_snapshot(
            self.scene, self.camera, timestamp_s, action=action
        )
        events = self.plugin.capture(action=effective_action)
        end = int(self.plugin.total_events)
        events = np.asarray(events, dtype=np.float64).reshape((-1, 4))
        event_start = end - len(events) if len(events) else end
        # ``total_events`` is the authoritative recorder-independent count;
        # start is retained for diagnostics and catches unexpected plugins.
        if start > end:
            raise RuntimeError("plugin event count moved backwards")
        event_start = end - len(events)
        if len(events):
            event_ts_start = float(events[0, 0])
            event_ts_end = float(events[-1, 0])
        else:
            event_ts_start = event_ts_end = None
        if self.exporter:
            self.exporter.sidecar.append(
                episode_index=self.episode_index,
                frame_index=self.frame_index,
                timestamp_s=timestamp_s,
                event_index_start=event_start,
                event_index_end=end,
                events_timestamp_start_s=event_ts_start,
                events_timestamp_end_s=event_ts_end,
                state=self.adapter.state_snapshot(self.scene, self.camera, timestamp_s),
                labels=self.adapter.label_snapshot(self.scene, self.camera, timestamp_s),
                action=effective_action,
            )
        self.frame_index += 1
        self._previous_timestamp_s = timestamp_s
        return events

    def end_episode(self) -> None:
        if self.plugin is not None and self.episode_index >= 0:
            self.plugin.end_episode()

    def close(self) -> None:
        if self._closed:
            return
        if self.plugin is not None:
            self.plugin.close()
        if self.exporter is not None:
            self.exporter.finalize()
        self._closed = True
