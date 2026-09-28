"""
实验性解析光流计算：利用部分 Genesis 物理状态近似逐像素稠密光流。

核心公式:
  F(x,y) = -K · V(z) · Δt / Z
  其中 K = 相机内参, V(z) = 物体在深度 Z 处的 3D 速度投影

算法:
当前限制：只使用实体世界系线速度的 x/y 分量；尚未纳入世界到相机
坐标变换、相机 ego-motion、轴向运动、角速度和遮挡。因此该结果不能
称为 Genesis ground-truth optical flow，尤其不能用于 wrist camera。

正式实验应优先使用 direct 高频渲染，或由渲染器/完整刚体投影提供光流。
"""

import numpy as np
from typing import List, Tuple, Optional


def backproject_pixels(depth: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    像素坐标 → 3D 相机坐标。

    P_cam = K⁻¹ · [x·z, y·z, z]ᵀ

    Args:
        depth: (H, W) float32 深度图
        K: (3, 3) float64 相机内参矩阵

    Returns:
        X, Y, Z: 各 (H, W) float32, 3D 相机坐标
    """
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    yy, xx = np.meshgrid(np.arange(H, dtype=np.float32),
                         np.arange(W, dtype=np.float32), indexing='ij')

    Z = depth.astype(np.float32)
    X = (xx - cx) * Z / fx
    Y = (yy - cy) * Z / fy
    return X, Y, Z


def project_velocity_to_2d(vel_3d: np.ndarray, Z: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    3D 速度 → 2D 光流分量。

    vx_2d = -fx · vx / Z
    vy_2d = -fy · vy / Z

    负号: 物体向右运动 → 像素坐标向左变化（相机坐标系约定）。

    Args:
        vel_3d: (3,) 线速度 [vx, vy, vz] m/s
        Z: (N,) 该物体覆盖像素的深度值
        K: (3, 3) 相机内参

    Returns:
        (vx_2d, vy_2d): 各 (N,) float32, 像素/秒
    """
    fx, fy = K[0, 0], K[1, 1]
    # 只考虑 x,y 方向的线速度（z方向速度对 2D 投影影响为二阶）
    safe_Z = np.where(Z < 1e-6, 1e-6, Z)  # 防止除零
    vx_2d = -fx * vel_3d[0] / safe_Z
    vy_2d = -fy * vel_3d[1] / safe_Z
    # 深度为0的像素置零流 (背景/天空)
    zero_mask = Z < 1e-6
    vx_2d[zero_mask] = 0.0
    vy_2d[zero_mask] = 0.0
    return vx_2d, vy_2d


def compute_analytic_flow(
    depth: np.ndarray,
    seg: np.ndarray,
    K: np.ndarray,
    moving_entities: List,
    dt: float = 1.0,
) -> np.ndarray:
    """
    Genesis 解析光流：全帧一次性计算。

    对每个运动实体:
      1. 获取 seg mask → 确定像素属于哪个物体
      2. 获取物体线速度 get_vel()
      3. 投影为 2D 光流分量
      4. 广播到 mask 区域

    当前版本仅处理线速度（刚体平移）。
    角速度支持待后续添加。

    Args:
        depth: (H, W) float32 深度图 (cam.render(depth=True))
        seg:   (H, W) int64 分割图 (cam.render(segmentation=True))
        K:     (3, 3) 相机内参矩阵
        moving_entities: 运动实体列表，每个必须有 .idx, .get_vel()
        dt:    帧间隔 (秒), 默认为 1.0 (返回光流 px/s)

    Returns:
        F_0_1: (2, H, W) float32 正向光流 [vx, vy] px/s
    """
    # 展开铰接体为独立 links (每 link 有自己的 seg ID 和速度)
    expanded = []
    for ent in moving_entities:
        if hasattr(ent, 'links') and len(ent.links) > 1:
            expanded.extend(ent.links[1:])  # 跳过 link0 (固定基座)
        else:
            expanded.append(ent)

    H, W = depth.shape
    F = np.zeros((2, H, W), dtype=np.float32)

    for entity in expanded:
        mask = (seg == entity.idx + 1)  # Genesis seg = sim_idx + 1
        if mask.sum() == 0:
            continue

        vel = entity.get_vel().cpu().numpy()  # (3,)
        Z = depth[mask].astype(np.float32)

        vx_2d, vy_2d = project_velocity_to_2d(vel, Z, K)
        F[0][mask] = vx_2d * dt
        F[1][mask] = vy_2d * dt

    return F


def compute_adaptive_upsample_factor(
    gray_frame: np.ndarray,
    F_0_1: np.ndarray,
    C_nom: float = 0.2,
    lambda_b: float = 0.5,
    min_U: int = 2,
    max_U: int = 300,
    flow_percentile: float = 100.0,
) -> int:
    """
    自适应插帧率 (ESIM 对比度变化上界)。

    M = max_x |V(x) · ∇log L(x)| → U。默认使用最大值以维持全像素
    对比度变化上界；分位数只保留为显式消融参数，不能宣称满足 ESIM
    的全图误差界。

    核心: 高速多插、低速少插。白墙(∇L≈0)→U=min_U。

    Args:
        gray_frame: (H, W) float32 灰度帧 [0,1]
        F_0_1:      (2, H, W) float32 正向光流 px/s
        C_nom:      对比度阈值 (默认 0.2, 对数域)
        lambda_b:   安全系数 (默认 0.5, ESIM 推荐)
        min_U:      最小插帧倍率
        max_U:      最大插帧倍率 (v2: 100→300, 适应高速)
        flow_percentile: 分位数 (默认 100，即最大值)

    Returns:
        U: int, 上采样倍率
    """
    eps = 1e-6
    log_frame = np.log(gray_frame.astype(np.float32) + eps)

    # 图像梯度 (对数域)
    gy, gx = np.gradient(log_frame)
    grad_log = np.stack([gx, gy], axis=0)  # (2, H, W)

    # M = max(|V · ∇logL|); percentile<100 is an explicit approximation.
    dLdt = np.abs((F_0_1 * grad_log).sum(axis=0))
    # 过滤 NaN/Inf (来自深度零和 log(0))
    dLdt = np.nan_to_num(dLdt, nan=0.0, posinf=0.0, neginf=0.0)
    nonzero = dLdt[dLdt > 1e-8]
    if len(nonzero) == 0:
        return min_U
    M = float(np.percentile(nonzero, flow_percentile))

    h = lambda_b * C_nom / M
    U = max(min_U, min(max_U, int(np.ceil(1.0 / h))))
    return U
