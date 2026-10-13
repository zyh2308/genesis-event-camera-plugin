"""Experiment-specific adapters around one EventGenesis capture interface.

Adapters do not render images or generate events.  They only expose the state
and labels required by a particular paper line.  This keeps Quality,
Sim-to-Real, and Robot runs on the same plugin and timestamp path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional


Provider = Callable[..., Any]


def _invoke(provider: Optional[Provider], *args, default=None):
    if provider is None:
        return default
    return provider(*args)


@dataclass
class ExperimentAdapter:
    """Small callback adapter shared by every experiment family.

    ``frame_provider`` must return the v3 mapping containing ``depth`` and
    ``seg`` and, for the frozen linear-radiance protocol, ``radiance``.
    ``motion_state_provider`` must return ``camera_to_world`` and
    ``object_to_world``.  The remaining providers are metadata-only and never
    enter the event-generation algorithm.
    """

    kind: str
    frame_provider: Provider
    motion_state_provider: Provider
    state_provider: Optional[Provider] = None
    label_provider: Optional[Provider] = None
    action_provider: Optional[Provider] = None

    def __post_init__(self) -> None:
        if self.kind not in {"quality", "sim2real", "robot"}:
            raise ValueError("kind must be quality, sim2real, or robot")
        if not callable(self.frame_provider) or not callable(self.motion_state_provider):
            raise TypeError("frame_provider and motion_state_provider are required callbacks")

    def state_snapshot(self, scene, camera, timestamp_s: float) -> Dict[str, Any]:
        value = _invoke(self.state_provider, scene, camera, float(timestamp_s), default={})
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise TypeError("state_provider must return a mapping")
        return dict(value)

    def label_snapshot(self, scene, camera, timestamp_s: float) -> Dict[str, Any]:
        value = _invoke(self.label_provider, scene, camera, float(timestamp_s), default={})
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise TypeError("label_provider must return a mapping")
        return dict(value)

    def action_snapshot(self, scene, camera, timestamp_s: float, action=None):
        if action is not None:
            return action
        return _invoke(self.action_provider, scene, camera, float(timestamp_s), default=None)


class QualityAdapter(ExperimentAdapter):
    def __init__(self, frame_provider: Provider, motion_state_provider: Provider,
                 state_provider: Optional[Provider] = None):
        super().__init__("quality", frame_provider, motion_state_provider, state_provider=state_provider)


class Sim2RealAdapter(ExperimentAdapter):
    def __init__(self, frame_provider: Provider, motion_state_provider: Provider,
                 state_provider: Optional[Provider] = None,
                 label_provider: Optional[Provider] = None):
        super().__init__("sim2real", frame_provider, motion_state_provider,
                         state_provider=state_provider, label_provider=label_provider)


class RobotAdapter(ExperimentAdapter):
    def __init__(self, frame_provider: Provider, motion_state_provider: Provider,
                 state_provider: Provider, action_provider: Optional[Provider] = None,
                 label_provider: Optional[Provider] = None):
        super().__init__("robot", frame_provider, motion_state_provider,
                         state_provider=state_provider, label_provider=label_provider,
                         action_provider=action_provider)
