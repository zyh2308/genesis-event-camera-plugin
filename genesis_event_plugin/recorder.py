"""
HDF5 数据记录器。

AER 格式 (必须):    aer_events.h5  → events: (N,4) uint64 [t_us, x, y, p]
事件帧 (可选):      event_frames.h5 → ep_N/frames: (T,H,W,2) float32
RGB 帧 (可选):      rgb_frames.h5   → ep_N/clean, ep_N/blur_Xms
"""

import h5py
import numpy as np
import os
import logging
import json
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)


class GenesisEventRecorder:
    """
    HDF5 记录器: AER + 事件帧 + RGB 帧。

    usage:
        rec = GenesisEventRecorder(output_dir, output_event_frames=True)
        rec.start_episode()
        rec.record_aer_events(events)
        rec.record_event_frame(frame_2ch)
        rec.record_action(action)
        rec.end_episode()
        rec.close()
    """

    def __init__(
        self,
        output_dir: str,
        output_event_frames: bool = False,
        output_rgb_frames: bool = False,
        event_frame_window: str = 'capture_interval',
        event_frame_window_n: int = 2000,
        event_frame_channels: int = 2,
        event_frame_normalization: str = 'log1p',
        rgb_exposure_times_ms: List[float] = None,
        compress: bool = True,
        resolution: Optional[tuple] = None,
        sensor_config: Optional[Dict] = None,
        camera_config: Optional[Dict] = None,
    ):
        """
        Args:
            output_dir: 输出目录
            output_event_frames: 是否额外产出事件帧 HDF5
            output_rgb_frames: 是否产出 RGB 帧 HDF5
            event_frame_window: 'capture_interval' | 'fixed_count' | 'fixed_duration'
            event_frame_window_n: fixed_count 的 N 个事件 / fixed_duration 的 N ms
            event_frame_channels: 1 (ON+OFF混) | 2 (ON/OFF分)
            event_frame_normalization: 'log1p' | 'none' | 'max' (legacy)
            rgb_exposure_times_ms: [2, 5, 10, 20] 等
            compress: 是否启用 gzip 压缩
        """
        self.output_dir = output_dir
        self.output_event_frames = output_event_frames
        self.output_rgb_frames = output_rgb_frames
        self.event_frame_window = event_frame_window
        self.event_frame_window_n = event_frame_window_n
        self.event_frame_channels = event_frame_channels
        self.event_frame_normalization = event_frame_normalization
        self.rgb_exposure_times_ms = rgb_exposure_times_ms or [2, 5, 10, 20]
        self.compress = compress
        self.resolution = resolution
        self.sensor_config = sensor_config or {}
        self.camera_config = camera_config or {}

        if event_frame_window not in {'fixed_count', 'fixed_duration', 'capture_interval'}:
            raise ValueError(
                "event_frame_window must be 'fixed_count', 'fixed_duration', "
                "or 'capture_interval'"
            )
        if event_frame_channels not in {1, 2}:
            raise ValueError("event_frame_channels must be 1 or 2")
        if event_frame_normalization not in {'none', 'log1p', 'max'}:
            raise ValueError(
                "event_frame_normalization must be 'none', 'log1p', or 'max'"
            )

        os.makedirs(output_dir, exist_ok=True)

        # ── AER 文件 (必须) ──
        aer_path = os.path.join(output_dir, 'aer_events.h5')
        self.aer_h5 = h5py.File(aer_path, 'w')
        self.aer_dataset = self.aer_h5.create_dataset(
            'events',
            shape=(0, 4),
            maxshape=(None, 4),
            # uint32 microseconds overflows after ~71.6 minutes, shorter than
            # many RL data-generation runs. uint64 keeps long episodes valid.
            dtype='uint64',
            compression='gzip' if compress else None,
        )
        self.aer_h5.attrs['schema_version'] = 'genesis-event-v2'
        self.aer_dataset.attrs['column_order'] = 't,x,y,p'
        self.aer_dataset.attrs['timestamp_unit'] = 'us'
        self.aer_dataset.attrs['polarity_encoding'] = '0=OFF,1=ON'
        if resolution is not None:
            self.aer_h5.attrs['resolution_hw'] = np.asarray(resolution, dtype=np.int32)
        self.aer_h5.attrs['sensor_config_json'] = json.dumps(
            self.sensor_config, sort_keys=True, ensure_ascii=False
        )
        self.aer_h5.attrs['camera_config_json'] = json.dumps(
            self.camera_config, sort_keys=True, ensure_ascii=False
        )
        self.episode_dataset = self.aer_h5.create_dataset(
            'episode_event_ranges',
            shape=(0, 3),
            maxshape=(None, 3),
            dtype='uint64',
        )
        self.episode_dataset.attrs['column_order'] = 'episode,start_inclusive,end_exclusive'
        self.aer_offset = 0  # 当前写入位置

        # ── 事件帧文件 (可选) ──
        if output_event_frames:
            ef_path = os.path.join(output_dir, 'event_frames.h5')
            self.ef_h5 = h5py.File(ef_path, 'w')
            self._write_root_metadata(self.ef_h5)
        else:
            self.ef_h5 = None

        # ── RGB 文件 (可选) ──
        if output_rgb_frames:
            rgb_path = os.path.join(output_dir, 'rgb_frames.h5')
            self.rgb_h5 = h5py.File(rgb_path, 'w')
            self._write_root_metadata(self.rgb_h5)
        else:
            self.rgb_h5 = None

        # ── 当前 episode 缓冲 ──
        self.ep_counter = -1
        self._ep_events_buffer = []         # [t, x, y, p] float32
        self._ep_event_frames = []          # (H, W, 2) float32
        self._ep_actions = []
        self._ep_rgb_frames = []            # (H, W, 3) uint8
        self._ep_timestamps = []            # 秒
        self._ep_event_timestamps = []      # 事件表示对应的决策时刻（秒）
        self._episode_event_start = 0
        self._last_event_time = None

    def _write_root_metadata(self, handle) -> None:
        """Keep sensor/optics provenance attached to every output product."""
        handle.attrs['schema_version'] = 'genesis-event-v2'
        if self.resolution is not None:
            handle.attrs['resolution_hw'] = np.asarray(
                self.resolution, dtype=np.int32
            )
        handle.attrs['sensor_config_json'] = json.dumps(
            self.sensor_config, sort_keys=True, ensure_ascii=False
        )
        handle.attrs['camera_config_json'] = json.dumps(
            self.camera_config, sort_keys=True, ensure_ascii=False
        )

    # ════════════════════════════════════════════════════════
    # Episode 管理
    # ════════════════════════════════════════════════════════

    def start_episode(self):
        """开始新 episode。"""
        self.ep_counter += 1
        self._ep_events_buffer = []
        self._ep_event_frames = []
        self._ep_actions = []
        self._ep_rgb_frames = []
        self._ep_timestamps = []
        self._ep_event_timestamps = []
        self._episode_event_start = self.aer_offset
        self._last_event_time = None

    def end_episode(self):
        """结束当前 episode，写入 HDF5。"""
        ep_idx = self.ep_counter

        # AER 是跨 episode 连续存储的，因此必须显式保存边界。
        n_ranges = len(self.episode_dataset)
        self.episode_dataset.resize(n_ranges + 1, axis=0)
        self.episode_dataset[n_ranges] = (
            ep_idx, self._episode_event_start, self.aer_offset
        )

        # ── 写入事件帧 ──
        if self.output_event_frames and self._ep_event_frames:
            ef_grp = self.ef_h5.create_group(f'ep_{ep_idx}')
            frames = np.stack(self._ep_event_frames, axis=0)  # (T, H, W, C)
            ef_grp.create_dataset(
                'frames', data=frames,
                compression='gzip' if self.compress else None,
            )
            if self._ep_actions:
                ef_grp.create_dataset(
                    'actions', data=np.array(self._ep_actions, dtype=np.float32),
                    compression='gzip' if self.compress else None,
                )
            if self._ep_event_timestamps:
                ef_grp.create_dataset(
                    'timestamps_s',
                    data=np.asarray(self._ep_event_timestamps, dtype=np.float64),
                )
            ef_grp.attrs['window'] = self.event_frame_window
            ef_grp.attrs['window_value'] = self.event_frame_window_n
            ef_grp.attrs['channels'] = 'ON,OFF' if self.event_frame_channels == 2 else 'signed'
            ef_grp.attrs['normalization'] = self.event_frame_normalization

        # ── 写入 RGB 帧 ──
        if self.output_rgb_frames and self._ep_rgb_frames:
            rgb_grp = self.rgb_h5.create_group(f'ep_{ep_idx}')
            rgb_array = np.stack(self._ep_rgb_frames, axis=0)  # (T, H, W, 3)
            rgb_grp.create_dataset(
                'clean', data=rgb_array,
                compression='gzip' if self.compress else None,
            )
            rgb_grp.create_dataset(
                'timestamps_s', data=np.asarray(self._ep_timestamps, dtype=np.float64)
            )

            # 运动模糊版本
            if self.rgb_exposure_times_ms:
                from .utils.motion_blur import generate_multi_exposure_rgb
                ts = np.array(self._ep_timestamps, dtype=np.float64)
                blurred = generate_multi_exposure_rgb(
                    rgb_array, ts, self.rgb_exposure_times_ms
                )
                for key, arr in blurred.items():
                    if key != 'clean':
                        rgb_grp.create_dataset(
                            key, data=arr,
                            compression='gzip' if self.compress else None,
                        )

    # ════════════════════════════════════════════════════════
    # 数据记录
    # ════════════════════════════════════════════════════════

    def record_aer_events(self, events: np.ndarray):
        """
        写入 AER 事件到 HDF5。

        Args:
            events: (N, 4) float64 [t(秒), x, y, p]
        """
        if len(events) == 0:
            return

        events = np.asarray(events, dtype=np.float64)
        if events.ndim != 2 or events.shape[1] != 4:
            raise ValueError(f"events must have shape (N,4), got {events.shape}")
        if not np.isfinite(events).all():
            raise ValueError("events contain NaN or infinite values")
        if (events[:, :3] < 0).any():
            # polarity is checked separately below; coordinates/time may not wrap
            # through the unsigned HDF5 representation.
            if (events[:, 0] < 0).any() or (events[:, 1:3] < 0).any():
                raise ValueError("event timestamps and coordinates must be non-negative")
        if not np.isin(events[:, 3], (-1.0, 1.0)).all():
            raise ValueError("event polarity must be -1 or +1")
        if not np.allclose(events[:, 1:3], np.rint(events[:, 1:3])):
            raise ValueError("event coordinates must be integer-valued")
        if np.any(np.diff(events[:, 0]) < 0):
            raise ValueError("events must be sorted by timestamp within each write")
        if self._last_event_time is not None and events[0, 0] < self._last_event_time:
            raise ValueError("event timestamps moved backwards within an episode")
        self._last_event_time = float(events[-1, 0])

        # 转为 uint64 格式 [t_us, x, y, p_idx]
        temp = events.copy()
        temp[:, 0] = np.rint(temp[:, 0] * 1e6)  # 秒 → 微秒（四舍五入）
        temp[:, 1:3] = np.rint(temp[:, 1:3])
        temp[temp[:, 3] < 0, 3] = 0     # OFF=0, ON=1
        temp = temp.astype(np.uint64)

        N = len(temp)
        self.aer_dataset.resize(self.aer_offset + N, axis=0)
        self.aer_dataset[self.aer_offset:self.aer_offset + N] = temp
        self.aer_offset += N

        # 同时缓冲给事件帧生成
        self._ep_events_buffer.append(events)

    def record_event_frame(self, on_off_frame: np.ndarray, timestamp: Optional[float] = None):
        """
        记录一帧事件帧。

        Args:
            on_off_frame: (H, W, 2) float32 [ON, OFF]
        """
        self._ep_event_frames.append(on_off_frame)
        if timestamp is not None:
            self._ep_event_timestamps.append(float(timestamp))

    def record_action(self, action: np.ndarray):
        """记录动作向量。"""
        self._ep_actions.append(action)

    def record_rgb_frame(self, rgb: np.ndarray, timestamp: float):
        """
        记录 RGB 帧和时间戳。

        Args:
            rgb: (H, W, 3) uint8
            timestamp: 秒
        """
        self._ep_rgb_frames.append(rgb)
        self._ep_timestamps.append(timestamp)

    def build_event_frame_from_buffer(
        self, frame_shape: tuple, current_time: Optional[float] = None
    ) -> np.ndarray:
        """
        从当前缓冲区的事件生成事件帧。

        Args:
            frame_shape: (H, W)

        Returns:
            event_frame: (H, W, C) float32 [ON, OFF] 或 [ON+OFF]
        """
        if self._ep_events_buffer:
            all_events = np.concatenate(self._ep_events_buffer, axis=0)
        else:
            all_events = np.empty((0, 4), dtype=np.float32)

        # 根据窗口策略选择事件
        if self.event_frame_window == 'fixed_count':
            selected = all_events[-self.event_frame_window_n:]
        elif self.event_frame_window == 'fixed_duration':
            if len(all_events) > 0:
                t_max = (
                    float(current_time) if current_time is not None
                    else float(all_events[-1, 0])
                )
                t_min = t_max - self.event_frame_window_n / 1000.0
                mask = all_events[:, 0] >= t_min
                selected = all_events[mask]
            else:
                selected = all_events
        else:  # capture_interval
            selected = all_events

        H, W = frame_shape
        img = np.zeros((H, W, self.event_frame_channels), dtype=np.float32)

        if len(selected) > 0:
            x = selected[:, 1].astype(np.int64)
            y = selected[:, 2].astype(np.int64)
            p = selected[:, 3]
            valid = (x >= 0) & (x < W) & (y >= 0) & (y < H)
            x, y, p = x[valid], y[valid], p[valid]
            if self.event_frame_channels == 2:
                channel = np.where(p > 0, 0, 1)
                np.add.at(img, (y, x, channel), 1.0)
            else:
                np.add.at(img[:, :, 0], (y, x), np.where(p > 0, 1.0, -1.0))

        # Representation scaling is explicit. log1p compresses the dynamic
        # range while preserving absolute event-rate differences; per-frame max
        # is retained only for legacy model compatibility.
        if self.event_frame_normalization == 'log1p':
            if self.event_frame_channels == 2:
                img = np.log1p(img)
            else:
                img = np.sign(img) * np.log1p(np.abs(img))
        elif self.event_frame_normalization == 'max':
            max_val = np.abs(img).max()
            if max_val > 0:
                img = img / max_val

        # capture_interval is non-overlapping and action-aligned. Fixed-count
        # and fixed-duration are rolling windows, so retain only their selected
        # history for the next decision.
        if self.event_frame_window == 'capture_interval' or len(selected) == 0:
            self._ep_events_buffer = []
        else:
            self._ep_events_buffer = [selected.copy()]

        return img

    # ════════════════════════════════════════════════════════
    # 清理
    # ════════════════════════════════════════════════════════

    def close(self):
        """关闭所有 HDF5 文件。"""
        if self.aer_h5 is not None:
            self.aer_h5.close()
        if self.ef_h5 is not None:
            self.ef_h5.close()
        if self.rgb_h5 is not None:
            self.rgb_h5.close()
        logger.info(f"Closed HDF5 files. AER total events: {self.aer_offset}")
