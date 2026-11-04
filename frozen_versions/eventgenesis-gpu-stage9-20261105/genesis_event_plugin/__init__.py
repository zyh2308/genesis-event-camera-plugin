"""
Genesis Event Camera Plugin

从 Genesis 物理仿真生成高质量 DVS 事件数据的 Python 插件。

核心组件:
  GenesisEventPlugin   — 顶层 API
  GenesisInterpolator  — Phase 1-6 插帧
  DvsEmulator          — DVS 像素模型
  GenesisEventRecorder — HDF5 数据记录
  DvsNoiseConfig       — 噪声配置

Presets:
  'clean' | 'noisy' | 'low_light' | 'high_speed' | 'overexposure'

快速开始:

    from genesis_event_plugin import GenesisEventPlugin

    plugin = GenesisEventPlugin(output_dir='output/', preset='real_v2')
    plugin.attach(scene, cam, [cube])
    plugin.start_episode()
    for _ in range(n_steps):
        scene.step()
        events = plugin.capture(action)
    plugin.end_episode()
    plugin.close()
"""

from .plugin import GenesisEventPlugin
from .noise_model import DvsNoiseConfig, get_preset, custom_config, PRESETS
from .dvs_emulator import DvsEmulator, DvsBatchEmulator
from .interpolator import GenesisInterpolator
from .adaptive_sampler import AdaptiveSamplingConfig, AdaptiveSamplingResult, PhysicsAdaptiveSampler
from .physics_interpolator import PhysicsInterpolator, RenderRequest, RenderRequiredError
from .physics_plugin import GenesisPhysicsEventPlugin
from .radiometry import (
    prepare_linear_radiance,
    log_radiance,
    srgb_to_linear,
    RadiometricResponse,
    physical_log_response,
    load_calibrated_transfer,
)
from .readout import ReadoutModel, IdealReadout, build_readout_model
from .calibration import CalibrationBundle
from .physics_warp import (
    projected_target_visibility,
    depth_aware_bidirectional_warp_projected_diagnostics,
    endpoint_radiance_residual,
)
from .torch_physics import (
    prepare_linear_radiance_torch,
    compute_se3_flow_torch,
    depth_aware_bidirectional_warp_torch,
    depth_aware_bidirectional_warp_projected_batch_torch,
)
from .genesis_adapter import (
    camera_to_world_cv,
    link_to_world_cv,
    make_motion_state_provider,
    make_frame_provider,
)
from .recorder import GenesisEventRecorder
from .camera_model import (
    CAMERA_PROFILES,
    EVK4_HD_IMX636,
    EventCameraProfile,
    get_camera_profile,
)
from .camera_margin import CameraMargin, estimate_required_margin, pad_intrinsics, crop_margin

__all__ = [
    'GenesisEventPlugin',
    'DvsNoiseConfig',
    'get_preset',
    'custom_config',
    'PRESETS',
    'DvsEmulator',
    'DvsBatchEmulator',
    'GenesisInterpolator',
    'AdaptiveSamplingConfig',
    'AdaptiveSamplingResult',
    'PhysicsAdaptiveSampler',
    'PhysicsInterpolator',
    'RenderRequest',
    'RenderRequiredError',
    'GenesisPhysicsEventPlugin',
    'prepare_linear_radiance',
    'log_radiance',
    'srgb_to_linear',
    'RadiometricResponse',
    'physical_log_response',
    'load_calibrated_transfer',
    'ReadoutModel',
    'IdealReadout',
    'build_readout_model',
    'CalibrationBundle',
    'projected_target_visibility',
    'depth_aware_bidirectional_warp_projected_diagnostics',
    'endpoint_radiance_residual',
    'prepare_linear_radiance_torch',
    'compute_se3_flow_torch',
    'depth_aware_bidirectional_warp_torch',
    'depth_aware_bidirectional_warp_projected_batch_torch',
    'camera_to_world_cv',
    'link_to_world_cv',
    'make_motion_state_provider',
    'make_frame_provider',
    'GenesisEventRecorder',
    'EventCameraProfile',
    'EVK4_HD_IMX636',
    'CAMERA_PROFILES',
    'get_camera_profile',
    'CameraMargin',
    'estimate_required_margin',
    'pad_intrinsics',
    'crop_margin',
]

__version__ = '0.2.0a1'
