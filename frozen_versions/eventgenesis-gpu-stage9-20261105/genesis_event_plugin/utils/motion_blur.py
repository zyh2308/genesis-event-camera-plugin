"""
RGB 相机运动模糊模拟。

模拟标准相机在曝光时间内的运动拖影效应。
原理: 曝光窗口内帧序列的时间加权平均 (ESIM CameraSimulator 同款)。

曝光时间参考:
  Intel RealSense D435:  1-33ms (自动曝光)
  FLIR Blackfly S:       0.016-30ms
  GoPro Hero 120fps:     ~1ms
  USB 摄像头 30fps:       ~10ms (室内典型)

用法:
  blurred = apply_motion_blur(frames, timestamps, exposure_time_ms=10.0)
"""

import numpy as np
from typing import List


def apply_motion_blur(
    frames: np.ndarray,
    timestamps: np.ndarray,
    exposure_time_ms: float,
) -> np.ndarray:
    """
    对高帧率序列施加运动模糊，模拟标准相机。

    曝光窗口 [t - τ/2, t + τ/2] 内的所有帧做等权时间平均。

    Args:
        frames:         (N, H, W, C) 高帧率序列
        timestamps:     (N,) 每帧时间戳 (秒)
        exposure_time_ms: 曝光时间 (毫秒)

    Returns:
        blurred: (N, H, W, C) 模糊后的序列，与输入 shape 相同

    Raises:
        ValueError: 如果帧数太少无法覆盖曝光窗口
    """
    tau_s = exposure_time_ms / 1000.0
    N = len(frames)
    blurred = np.zeros_like(frames, dtype=np.float32)

    for i in range(N):
        t_center = timestamps[i]
        t_start = t_center - tau_s / 2
        t_end = t_center + tau_s / 2

        mask = (timestamps >= t_start) & (timestamps <= t_end)
        count = mask.sum()

        if count == 0:
            # 曝光窗口内无帧 -> 直接用当前帧
            blurred[i] = frames[i].astype(np.float32)
        else:
            # ESIM 风格: 加权时间平均 (等权)
            blurred[i] = frames[mask].astype(np.float32).mean(axis=0)

    # 转回原始 dtype
    if frames.dtype == np.uint8:
        blurred = np.clip(blurred, 0, 255).astype(np.uint8)

    return blurred


def generate_multi_exposure_rgb(
    frames: np.ndarray,
    timestamps: np.ndarray,
    exposure_times_ms: List[float],
) -> dict:
    """
    一次生成多个曝光时间的模糊 RGB，用于 ablation 对照实验。

    Args:
        frames:             (N, H, W, 3) uint8 高帧率 RGB
        timestamps:         (N,) 秒
        exposure_times_ms:  [1, 2, 5, 10, 20, 33] 等

    Returns:
        {
            'clean':   (N, H, W, 3) 无模糊原始帧,
            'blur_1ms': (N, H, W, 3),
            'blur_2ms': (N, H, W, 3),
            ...
        }
    """
    result = {'clean': frames}

    for et in exposure_times_ms:
        key = f'blur_{et}ms'
        result[key] = apply_motion_blur(frames, timestamps, et)

    return result
