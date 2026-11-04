"""
Genesis 事件相机插件 — 顶层入口。

整合: GenesisInterpolator + DvsEmulator + GenesisEventRecorder

典型用法:

    import genesis as gs
    from genesis_event_plugin import GenesisEventPlugin

    gs.init(backend=gs.cpu)
    scene = gs.Scene(...)
    cam = scene.add_camera(...)
    cube = scene.add_entity(...)
    scene.build()

    plugin = GenesisEventPlugin(
        output_dir='output/',
    preset='real_v2',
        interpolation_mode='analytic',
        output_event_frames=True,
    )
    plugin.attach(scene, cam, [cube])

    plugin.start_episode()
    for step in range(1000):
        obs = get_observation(scene)
        action = policy(obs)
        scene.step()
        # 插件内部: scene.step → 渲染 → 插帧 → DVS → 事件
        events = plugin.capture(action)
    plugin.end_episode()
    plugin.close()
"""

import logging
import numpy as np
from dataclasses import asdict, replace
from typing import List, Optional, Dict, Union

from .interpolator import GenesisInterpolator
from .dvs_emulator import DvsEmulator
from .recorder import GenesisEventRecorder
from .noise_model import DvsNoiseConfig, get_preset, custom_config
from .camera_model import EventCameraProfile, get_camera_profile

logger = logging.getLogger(__name__)


class GenesisEventPlugin:
    """
    Genesis 事件相机插件。

    负责:
      1. 从 Genesis 场景获取帧 (渲染 + 插帧)
      2. DVS 事件生成
      3. 数据存储 (AER + 事件帧 + RGB 帧)
    """

    def __init__(
        self,
        output_dir: str = 'output/',
        # DVS 配置
        preset: str = 'real_v2',
        noise_config: DvsNoiseConfig = None,
        camera_profile: Union[str, EventCameraProfile, None] = None,
        strict_camera_profile: bool = False,
        # 插帧配置
        interpolation_mode: str = 'direct',
        U_fixed: int = 5,
        C_nom: float = 0.2,
        lambda_b: float = 0.5,
        min_U: int = 2,
        max_U: int = 100,
        # 输出配置
        output_event_frames: bool = False,
        event_frame_window: str = 'capture_interval',
        event_frame_window_n: int = 2000,
        event_frame_channels: int = 2,
        event_frame_normalization: str = 'log1p',
        output_rgb_frames: bool = False,
        rgb_exposure_times_ms: List[float] = None,
        compress: bool = True,
        # 其他
        device: str = None,
        fallback_dt_s: float = 0.001,
    ):
        """
        Args:
            output_dir: 输出根目录
            preset: DVS 噪声预设
            noise_config: 自定义 DVS 配置 (覆盖 preset)
            camera_profile: 目标真机/镜头档案或名称（如 'evk4'）
            strict_camera_profile: 是否要求原生分辨率和名义光学严格匹配
            interpolation_mode: 'direct' | 'analytic' | 'fixed'
            U_fixed: 固定插帧倍率
            C_nom, lambda_b: 自适应采样参数
            output_event_frames: 是否产出事件帧 HDF5
            event_frame_window: 'capture_interval' | 'fixed_count' | 'fixed_duration'
            event_frame_window_n: fixed_count 的 N events / fixed_duration 的 N ms
            event_frame_channels: 1/2
            event_frame_normalization: 'log1p' (recommended) | 'none' | 'max' (legacy)
            output_rgb_frames: 是否产出 RGB 帧
            rgb_exposure_times_ms: 多曝光时间列表
            compress: HDF5 gzip 压缩
            device: torch device
        """
        self.output_dir = output_dir

        # ── DVS 配置 ──
        if noise_config is not None:
            self.noise_config = replace(noise_config)
            self.noise_config.validate()
        else:
            self.noise_config = get_preset(preset)
        self.preset_name = "custom" if noise_config is not None else preset
        self.camera_profile = (
            get_camera_profile(camera_profile) if camera_profile is not None else None
        )
        self.strict_camera_profile = bool(strict_camera_profile)

        # ── 插帧参数 ──
        self.interpolation_mode = interpolation_mode
        self.U_fixed = U_fixed
        self.C_nom = C_nom
        self.lambda_b = lambda_b
        self.min_U = min_U
        self.max_U = max_U
        self.motion_blur_exposure_s = 0.0  # 0=关, 高速场景建议 0.01

        # ── 设备 ──
        if device is None:
            import torch
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.fallback_dt_s = float(fallback_dt_s)
        if self.fallback_dt_s <= 0:
            raise ValueError("fallback_dt_s must be positive")
        self._warned_fallback_time = False

        # ── 组件 (延迟初始化) ──
        self.interpolator: Optional[GenesisInterpolator] = None
        self.dvs_emulator: Optional[DvsEmulator] = None
        self.recorder: Optional[GenesisEventRecorder] = None

        # ── Genesis 绑定 ──
        self.scene = None
        self.cam = None
        self.moving_entities = []

        # ── 帧缓冲 ──
        self._prev_rgb = None
        self._prev_depth = None
        self._prev_seg = None
        self._prev_t = None
        self._initialized = False

        # ── 记录器参数 ──
        self._output_event_frames = output_event_frames
        self._event_frame_window = event_frame_window
        self._event_frame_window_n = event_frame_window_n
        self._event_frame_channels = event_frame_channels
        self._event_frame_normalization = event_frame_normalization
        self._output_rgb_frames = output_rgb_frames
        self._rgb_exposure_times_ms = rgb_exposure_times_ms
        self._compress = compress

        # ── 统计 ──
        self.total_events = 0
        self.total_frames = 0
        self.total_steps = 0

    # ════════════════════════════════════════════════════════
    # 绑定 Genesis
    # ════════════════════════════════════════════════════════

    def attach(
        self,
        scene,
        cam,
        moving_entities: List = None,
    ):
        """
        绑定 Genesis 场景和相机。

        Args:
            scene: gs.Scene 实例
            cam: gs.Camera 实例
            moving_entities: 运动实体列表 (entity.idx 必须可用)
        """
        self.scene = scene
        self.cam = cam
        self.moving_entities = moving_entities or []

        if self.camera_profile is not None:
            profile_messages = self.camera_profile.validate_genesis_camera(
                cam, strict=self.strict_camera_profile
            )
            for message in profile_messages:
                logger.warning("Camera profile check: %s", message)

        # 初始化插帧器
        K = cam.intrinsics
        self.interpolator = GenesisInterpolator(
            K=K,
            mode=self.interpolation_mode,
            U_fixed=self.U_fixed,
            C_nom=self.C_nom,
            lambda_b=self.lambda_b,
            min_U=self.min_U,
            max_U=self.max_U,
            device=self.device,
            motion_blur_exposure_s=self.motion_blur_exposure_s,
        )

        # 初始化 DVS 仿真器
        res = (cam.res[1], cam.res[0])  # (H, W)
        self.dvs_emulator = DvsEmulator(
            res=res,
            config=self.noise_config,
            device=self.device,
        )

        # 初始化记录器
        sensor_config = asdict(self.noise_config)
        sensor_config["preset_name"] = self.preset_name
        camera_config = {
            "render_resolution_wh": list(cam.res),
            "render_intrinsics": np.asarray(K, dtype=np.float64).tolist(),
            "render_vertical_fov_deg": (
                float(cam.fov) if getattr(cam, "fov", None) is not None else None
            ),
            "render_model": getattr(cam, "model", getattr(cam, "_model", None)),
        }
        if self.camera_profile is not None:
            camera_config["target_profile"] = self.camera_profile.to_metadata()

        self.recorder = GenesisEventRecorder(
            output_dir=self.output_dir,
            output_event_frames=self._output_event_frames,
            output_rgb_frames=self._output_rgb_frames,
            event_frame_window=self._event_frame_window,
            event_frame_window_n=self._event_frame_window_n,
            event_frame_channels=self._event_frame_channels,
            event_frame_normalization=self._event_frame_normalization,
            rgb_exposure_times_ms=self._rgb_exposure_times_ms,
            compress=self._compress,
            resolution=res,
            sensor_config=sensor_config,
            camera_config=camera_config,
        )

    # ════════════════════════════════════════════════════════
    # Episode 管理
    # ════════════════════════════════════════════════════════

    def start_episode(self):
        """开始新 episode，重置所有内部状态。"""
        self.recorder.start_episode()

        # 重置帧缓冲
        self._prev_rgb = None
        self._prev_depth = None
        self._prev_seg = None
        self._prev_t = None
        self._initialized = False

        if self.dvs_emulator is not None:
            self.dvs_emulator.reset()

        self.total_steps = 0

    def end_episode(self):
        """结束当前 episode。"""
        self.recorder.end_episode()

    # ════════════════════════════════════════════════════════
    # 主循环: capture()
    # ════════════════════════════════════════════════════════

    def capture(self, action: np.ndarray = None) -> np.ndarray:
        """
        捕获本步的事件。

        内部流程:
          1. 渲染当前帧 (RGB + Depth + Seg)
          2. 如果已有前一帧 → 插帧 → DVS 事件
          3. 存储 AER 事件 → 生成事件帧
          4. 更新帧缓冲

        Args:
            action: 当前动作 (可选，用于记录)

        Returns:
            events: (N, 4) float64 [t(s), x, y, p], 首帧返回空数组
        """
        if self.cam is None:
            raise RuntimeError("Plugin not attached. Call attach() first.")

        self.total_steps += 1

        # ── 渲染 ──
        needs_geometry = (
            self.interpolation_mode == 'analytic'
            or self.noise_config.depth_noise_scale > 0
        )
        rgb, depth, seg, _ = self.cam.render(
            rgb=True,
            depth=needs_geometry,
            segmentation=needs_geometry,
        )
        if hasattr(self.scene, 'cur_t'):
            t_current = float(self.scene.cur_t)
        else:
            t_current = self.total_steps * self.fallback_dt_s
            if not self._warned_fallback_time:
                logger.warning(
                    "Genesis scene has no cur_t; using explicit fallback_dt_s=%g. "
                    "Verify this equals the physics timestep before using timestamps.",
                    self.fallback_dt_s,
                )
                self._warned_fallback_time = True

        # ── 首次: 初始化 DVS + 存储帧 ──
        if not self._initialized:
            gray = self._to_gray(rgb)
            self.dvs_emulator.initialize(gray, t_current)
            self._prev_rgb = rgb
            self._prev_depth = depth
            self._prev_seg = seg
            self._prev_t = t_current
            self._initialized = True

            # 记录首帧 RGB
            if self._output_rgb_frames:
                self.recorder.record_rgb_frame(rgb, t_current)

            return np.empty((0, 4), dtype=np.float64)

        # ── 插帧 ──
        gray_frames, timestamps = self.interpolator.interpolate(
            self._prev_rgb, self._prev_depth, self._prev_seg,
            rgb, depth, seg,
            self.moving_entities,
            self._prev_t, t_current,
        )

        # ── DVS 事件生成 ──
        all_events_list = []
        for gray_frame, ts in zip(gray_frames, timestamps):
            events = self.dvs_emulator.generate_events(
                gray_frame,
                ts,
                # Direct depth-conditioned noise is retained only as an
                # explicitly requested legacy experiment.
                depth_map=self._prev_depth
                if self.noise_config.depth_noise_scale > 0 else None,
            )
            if len(events) > 0:
                all_events_list.append(events)

        if all_events_list:
            all_events = np.concatenate(all_events_list, axis=0)
            all_events = all_events[all_events[:, 0].argsort()]
        else:
            all_events = np.empty((0, 4), dtype=np.float64)

        # ── 存储 AER ──
        self.recorder.record_aer_events(all_events)

        # ── 存储事件帧 ──
        if self._output_event_frames:
            H, W = rgb.shape[:2]
            ef = self.recorder.build_event_frame_from_buffer(
                (H, W), current_time=t_current
            )
            # Preserve one observation per decision even when the interval is
            # silent; silence is a valid event-camera measurement.
            self.recorder.record_event_frame(ef, timestamp=t_current)

        # ── 存储 RGB ──
        if self._output_rgb_frames:
            self.recorder.record_rgb_frame(rgb, t_current)

        # ── 记录动作 ──
        if action is not None:
            self.recorder.record_action(action)

        # ── 更新缓冲 ──
        self._prev_rgb = rgb
        self._prev_depth = depth
        self._prev_seg = seg
        self._prev_t = t_current

        self.total_events += len(all_events)
        self.total_frames += len(gray_frames)

        return all_events

    # ════════════════════════════════════════════════════════
    # 设置
    # ════════════════════════════════════════════════════════

    def set_preset(self, preset: str):
        """Select a preset before attach(), preserving dataset provenance."""
        if self.dvs_emulator is not None or self.recorder is not None:
            raise RuntimeError(
                "set_preset() after attach() would invalidate sensor metadata "
                "and state. Create a new plugin instance for another sensor profile."
            )
        self.noise_config = get_preset(preset)
        self.preset_name = preset

    def get_stats(self) -> dict:
        """返回插件运行统计。"""
        stats = {
            'total_steps': self.total_steps,
            'total_events': self.total_events,
            'total_interp_frames': self.total_frames,
        }
        if self.dvs_emulator is not None:
            stats['dvs'] = self.dvs_emulator.get_stats()
        if self.interpolator is not None:
            stats['interpolator'] = self.interpolator.get_stats()
        return stats

    def close(self):
        """关闭插件，写入所有待存数据。"""
        if self.recorder is not None:
            self.recorder.close()
        logger.info(f"Plugin closed. Stats: {self.get_stats()}")

    # ════════════════════════════════════════════════════════
    # 内部
    # ════════════════════════════════════════════════════════

    @staticmethod
    def _to_gray(rgb: np.ndarray) -> np.ndarray:
        return (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1]
                + 0.114 * rgb[:, :, 2]).astype(np.uint8)
