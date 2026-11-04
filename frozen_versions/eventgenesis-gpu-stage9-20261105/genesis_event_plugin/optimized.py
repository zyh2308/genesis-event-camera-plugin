"""Explicit optimized entry; unsupported configurations keep the reference path."""
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import torch
from .physics_plugin import GenesisPhysicsEventPlugin
from .noise_model import DvsNoiseConfig
from .gpu_interpolator import ResidentPhysicsInterpolator,DeferredDiagnostics
from .gpu_sensor import generate_device_sequence

class OptimizedGenesisPhysicsEventPlugin(GenesisPhysicsEventPlugin):
    optimization_active=False

    def attach(self,*args,**kwargs):
        result=super().attach(*args,**kwargs)
        profile=json.loads((Path(__file__).resolve().parent.parent/'stage9_sensor_profile.json').read_text())
        expected=asdict(DvsNoiseConfig(**profile['noise_config_kwargs']))
        reasons=[]
        try:same_profile=asdict(self.noise_config)==expected
        except (TypeError,ValueError):same_profile=False
        if not same_profile:reasons.append('sensor profile outside verified Stage9 configuration')
        if self.warp_backend!='torch' or not str(self.device).startswith('cuda'):reasons.append('requires CUDA Torch geometry')
        if self.warp_composite!='coverage_time_log_linear' or self.disocclusion_policy!='hold':reasons.append('unsupported geometry policy')
        if self.camera_margin is not None or self.intermediate_state_provider is not None:reasons.append('margin or state provider uses reference path')
        if tuple(self.cam.res) not in {(320,240),(346,260)}:reasons.append('resolution outside verified Stage9 shapes')
        self.optimization_active=not reasons
        self.optimization_status={'active':self.optimization_active,'reference_fallback_reasons':reasons}
        if self.optimization_active:self.interpolator=ResidentPhysicsInterpolator.from_reference(self.interpolator)
        return result

    def _generate_device_sequence(self,frames,times):
        value=generate_device_sequence(self.dvs_emulator,torch.stack(frames),times)
        self.fallbacks=getattr(self.dvs_emulator,'_candidate_fallbacks',0)
        return value

    def materialize_diagnostics(self):
        if isinstance(self.interpolator.last_visibility,DeferredDiagnostics):self.interpolator.last_visibility.materialize()
        list(self.interpolator.last_intermediate_z)

    def capture(self, action=None):
        if not self.optimization_active:
            return super().capture(action)
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
        frames, timestamps = self.interpolator.interpolate_device(
            self._prev["radiance"], self._prev["depth"], self._prev["seg"],
            radiance, depth, seg,
            self._prev["motion"]["camera_to_world"], motion["camera_to_world"],
            object_delta, self._prev["t"], t_current, input_space="linear",
            object_world0=self._prev["motion"]["object_to_world"],
            object_world1=motion["object_to_world"],
            intermediate_state_provider=self.intermediate_state_provider,
        )
        self.interpolator.last_uncertainty[
            "radiometric_calibration_status"
        ] = self.noise_config.radiometric_calibration_status
        sensor_frames = [self._sensor_crop(frame) for frame in frames]
        chunks = self._generate_device_sequence(sensor_frames,timestamps)
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
