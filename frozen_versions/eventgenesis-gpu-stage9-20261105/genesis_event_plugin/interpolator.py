"""Genesis 时间采样/历史插帧后端。

``direct`` 逐次使用真实渲染端点，是当前可信主基线。``analytic`` 只实现
部分实体 x/y 平移，缺相机 ego-motion、轴向/角运动与遮挡，仅供实验；
``fixed`` 是可能制造非物理事件的交叉淡化历史基线。
"""

import logging
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional

from .utils.flow_utils import compute_analytic_flow, compute_adaptive_upsample_factor

logger = logging.getLogger(__name__)


class GenesisInterpolator:
    """
    插帧器: 从 Genesis 源帧生成高帧率序列。

    usage:
        interp = GenesisInterpolator(K=cam.intrinsics)
        gray_frames, timestamps = interp.interpolate(
            rgb0, depth0, seg0,
            rgb1, depth1, seg1,
            moving_entities, t0, t1
        )
    """

    def __init__(
        self,
        K: np.ndarray,
        mode: str = 'direct',
        U_fixed: int = 5,
        C_nom: float = 0.2,
        lambda_b: float = 0.5,
        min_U: int = 2,
        max_U: int = 300,  # v2: 100→300, 适应 3×+ 高速场景
        motion_blur_exposure_s: float = 0.0,
        device: str = None,
    ):
        """
        Args:
            K:         (3,3) 相机内参矩阵
            mode:      'direct' (逐物理步直采) | 'analytic' (实验性解析流)
                       | 'fixed' (旧版交叉淡化，仅用于复现)
            U_fixed:   固定插帧倍率 (mode='fixed' 时使用)
            C_nom:     对比度阈值 (自适应采样参数)
            lambda_b:  安全系数 (ESIM 推荐 0.5)
            min_U:     最小插帧倍率
            max_U:     最大插帧倍率
            device:    torch device
        """
        self.K = K
        self.mode = mode
        if mode not in {'direct', 'analytic', 'fixed'}:
            raise ValueError("mode must be 'direct', 'analytic', or 'fixed'")
        if mode == 'analytic':
            logger.warning(
                "analytic interpolation is experimental: current Genesis state "
                "projection omits camera ego-motion, axial motion and angular "
                "velocity. Do not use it for wrist-camera claims."
            )
        elif mode == 'fixed':
            logger.warning(
                "fixed interpolation cross-fades frames and may generate "
                "non-physical events; use only as a legacy baseline."
            )
        self.U_fixed = U_fixed
        self.C_nom = C_nom
        self.lambda_b = lambda_b
        self.min_U = min_U
        self.max_U = max_U
        if motion_blur_exposure_s != 0:
            raise ValueError(
                "motion_blur_exposure_s must be 0 for the event signal; model "
                "finite photoreceptor bandwidth instead. RGB comparison blur "
                "belongs in the recorder/output branch."
            )
        self.motion_blur_exposure_s = 0.0

        if device is None:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        # 统计
        self.total_interpolations = 0
        self.avg_U = 0.0
        self.last_flow = None  # 最近一次计算的解析光流 (2,H,W)

    def interpolate(
        self,
        rgb0: np.ndarray, depth0: np.ndarray, seg0: np.ndarray,
        rgb1: np.ndarray, depth1: np.ndarray, seg1: np.ndarray,
        moving_entities: List,
        t0: float, t1: float,
    ) -> Tuple[List[np.ndarray], List[float]]:
        """
        在一对帧之间插值。

        Args:
            rgb0/1:   (H, W, 3) uint8 RGB
            depth0/1: (H, W) float32 深度
            seg0/1:   (H, W) int64 分割
            moving_entities: 运动实体列表
            t0/t1:    帧时间戳 (秒)

        Returns:
            gray_frames: [U] 个 (H, W) uint8 灰度帧
            timestamps:  [U] 个时间戳 (秒)
        """
        H, W = rgb0.shape[:2]

        # ── 转灰度 ──
        gray0 = self._to_gray(rgb0).astype(np.float32) / 255.0
        gray1 = self._to_gray(rgb1).astype(np.float32) / 255.0

        # Preferred Genesis-native path: when capture() is called at a cadence
        # that resolves contrast change (typically every physics/render substep),
        # feed the true rendered endpoint to the sensor model. This avoids
        # cross-dissolve/warp ghosts and makes the timestamp exactly observable.
        if self.mode == 'direct':
            self.last_flow = np.zeros((2, H, W), dtype=np.float32)
            n = self.total_interpolations
            self.avg_U = (self.avg_U * n + 1) / (n + 1)
            self.total_interpolations += 1
            return [(gray1 * 255).astype(np.uint8)], [float(t1)]

        # ── Phase 1: 解析光流 ──
        dt = t1 - t0
        if self.mode == 'analytic':
            if depth0 is None or seg0 is None:
                raise ValueError("analytic interpolation requires depth and segmentation")
            F_0_1 = compute_analytic_flow(depth0, seg0, self.K, moving_entities, dt=dt)
            self.last_flow = F_0_1
        else:
            # 固定模式: 无光流，跳过所有 warp
            F_0_1 = np.zeros((2, H, W), dtype=np.float32)
            self.last_flow = F_0_1

        # ── Phase 2: 自适应插帧率 (v2: 95分位数 + max_U=300) ──
        if self.mode == 'analytic':
            U = compute_adaptive_upsample_factor(
                gray0, F_0_1, C_nom=self.C_nom,
                lambda_b=self.lambda_b, min_U=self.min_U, max_U=self.max_U,
                flow_percentile=100.0,
            )
        elif self.mode == 'fixed':
            U = self.U_fixed
        else:
            U = self.U_fixed

        # 更新统计
        n = self.total_interpolations
        self.avg_U = (self.avg_U * n + U) / (n + 1)
        self.total_interpolations += 1

        # ── 转 torch ──
        I0 = torch.from_numpy(gray0).float().unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,H,W)
        I1 = torch.from_numpy(gray1).float().unsqueeze(0).unsqueeze(0).to(self.device)
        F01 = torch.from_numpy(F_0_1).float().unsqueeze(0).to(self.device)  # (1,2,H,W)
        F10 = -F01  # 反向光流 ≈ -正向 (刚体近似)

        gray_frames = []
        timestamps = []

        for k in range(U):
            t = (k + 0.5) / U
            ts = t0 + t * dt
            timestamps.append(ts)

            if U <= 1 or self.mode not in ('analytic', 'fixed'):
                frame = gray0
            elif self.mode == 'fixed':
                blended = ((1 - t) * gray0 + t * gray1)
                frame = (blended * 255).astype(np.uint8)
            else:
                # ── Phase 3: 三次多项式混合光流 ──
                c = -t * (1 - t)
                F_t_0 = c * F01 + t * t * F10        # t → 0
                F_t_1 = (1 - t) * (1 - t) * F01 + c * F10  # t → 1

                # ── Phase 4: 双端 warp ──
                grid_t_0 = self._flow_to_grid(F_t_0)
                grid_t_1 = self._flow_to_grid(F_t_1)

                warped_I0 = F.grid_sample(I0, grid_t_0, mode='bilinear',
                                          padding_mode='border', align_corners=True)
                warped_I1 = F.grid_sample(I1, grid_t_1, mode='bilinear',
                                          padding_mode='border', align_corners=True)

                # ── Phase 5+6: 时间加权融合 ──
                # 注意: 深度遮挡在边界因双线性插值会误判(cube: 368px被错误强制用I1)
                # 改为简单时间加权: t→0时偏I0, t→1时偏I1, 中间平滑过渡
                # 对于真正需要遮挡的场景(两物体交叠), 未来可加回深度判据
                Ft_p = (1 - t) * warped_I0 + t * warped_I1

                # ── 转 numpy ──
                frame = Ft_p.squeeze().cpu().numpy()
                frame = np.clip(frame * 255, 0, 255).astype(np.uint8)

            gray_frames.append(frame)

        return gray_frames, timestamps

    # ════════════════════════════════════════════════════════
    # 内部工具
    # ════════════════════════════════════════════════════════

    @staticmethod
    def _to_gray(rgb: np.ndarray) -> np.ndarray:
        """RGB → 灰度 (亮度加权)。"""
        return (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1]
                + 0.114 * rgb[:, :, 2]).astype(np.uint8)

    @staticmethod
    def _flow_to_grid(flow: torch.Tensor) -> torch.Tensor:
        """
        光流 → grid_sample 坐标网格。

        grid_sample 期望 [-1, 1] 归一化坐标。

        Args:
            flow: (1, 2, H, W) 光流 [vx, vy] 单位: 像素

        Returns:
            grid: (1, H, W, 2) 归一化坐标
        """
        _, _, H, W = flow.shape

        # 基础网格: 每个像素的 (x, y) 坐标
        yy, xx = torch.meshgrid(
            torch.arange(H, dtype=torch.float32, device=flow.device),
            torch.arange(W, dtype=torch.float32, device=flow.device),
            indexing='ij',
        )

        # 加上光流
        x_warped = xx + flow[0, 0]  # vx
        y_warped = yy + flow[0, 1]  # vy

        # 归一化到 [-1, 1]
        x_norm = 2.0 * x_warped / (W - 1) - 1.0
        y_norm = 2.0 * y_warped / (H - 1) - 1.0

        grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0)
        return grid


    def _apply_motion_blur(self, gray: np.ndarray, seg: np.ndarray, depth: np.ndarray,
                           entities: List, K: np.ndarray) -> np.ndarray:
        """Genesis 专属: 分割驱动的物体级运动模糊。

        思路: 真实相机快门期间, 运动物体在像素面上拖出轨迹。
        V2E 只能对全图做 box blur (无法区分动静),
        我们利用 seg 对每物体独立施加速度方向模糊。

        Args:
            gray: (H,W) uint8 灰度帧
            seg:  (H,W) int64 分割图
            entities: 运动实体列表
            delta_t: 插帧间隔 (用于换算模糊量, s)
            K: (3,3) 相机内参

        Returns:
            blurred: (H,W) uint8
        """
        blurred = gray.astype(np.float64)
        fx, fy = K[0, 0], K[1, 1]

        for entity in entities:
            mask = (seg == entity.idx + 1)  # Genesis seg = sim_idx + 1
            if mask.sum() == 0:
                continue

            # 获取实体 3D 速度 (m/s) → 2D 速度 (px/s)
            vel_3d = entity.get_vel().cpu().numpy()  # (3,)

            # 实体平均深度
            entity_z = depth[mask].mean()
            vx_px = -fx * vel_3d[0] / max(entity_z, 0.01)  # px/s
            vy_px = -fy * vel_3d[1] / max(entity_z, 0.01)

            # 模糊量 = 速度 × 快门时间
            blur_x = abs(vx_px) * self.motion_blur_exposure_s
            blur_y = abs(vy_px) * self.motion_blur_exposure_s

            if blur_x < 0.5 and blur_y < 0.5:
                continue  # 不足半像素, 跳过

            # 归一化卷积: 只在 mask 内模糊, 边界正确处理
            mask_f = mask.astype(np.float64)
            region = blurred * mask_f

            if blur_x >= 0.5:
                kx = int(np.ceil(blur_x))
                kernel = np.ones(kx) / kx
                region_b = np.apply_along_axis(
                    lambda r: np.convolve(r, kernel, mode='same'), 1, region)
                mask_b = np.apply_along_axis(
                    lambda r: np.convolve(r, kernel, mode='same'), 1, mask_f)
                valid = mask_b > 0
                region[valid] = region_b[valid] / mask_b[valid]

            if blur_y >= 0.5:
                ky = int(np.ceil(blur_y))
                kernel = np.ones(ky) / ky
                region_b = np.apply_along_axis(
                    lambda r: np.convolve(r, kernel, mode='same'), 0, region)
                mask_b = np.apply_along_axis(
                    lambda r: np.convolve(r, kernel, mode='same'), 0, mask_f)
                valid = mask_b > 0
                region[valid] = region_b[valid] / mask_b[valid]

            # 写回: 只更新该 entity 的区域
            blurred = np.where(mask, region, blurred)

        # 保留输入类型: uint8→uint8, float→float
        if gray.dtype == np.uint8 or gray.max() > 1.0:
            return np.clip(blurred, 0, 255).astype(np.uint8)
        else:
            return np.clip(blurred, 0.0, 1.0).astype(np.float32)

    def get_stats(self) -> dict:
        """返回插帧统计。"""
        return {
            'mode': self.mode,
            'total_interpolations': self.total_interpolations,
            'avg_upsample_factor': round(self.avg_U, 1),
        }
