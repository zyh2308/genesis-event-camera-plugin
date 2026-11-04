"""
统一评估框架：Genesis 事件仿真器 × PECS 方法论
===================================================
整合:
  1. Genesis 六项内部质量指标（边缘率、GT/rnd、暗角率、ON/OFF平衡、Q2、流入噪声）
  2. PECS CD/GD 外部对齐指标（与真实事件对比）
  3. PECS 论文 baseline 对比

用法:
  eval = EventSimEvaluator()
  # 仅内部指标
  report = eval.evaluate_internal(genesis_events, rgb_frame)
  # 与真实事件对比
  report = eval.evaluate_vs_real(genesis_events, real_events)
  # 全量评估
  report = eval.full_evaluation(genesis_events, real_events, rgb_frame)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import json
import time
import logging

from .event_schema import canonicalize_events

from .pecs_metrics import (
    chamfer_distance, gaussian_distance, compute_both,
    compare_to_baselines, normalize_events,
    PECS_BASELINES, V2E_BASELINES, ESIM_BASELINES
)


logger = logging.getLogger(__name__)


@dataclass
class InternalMetrics:
    """Genesis 插件六项内部质量指标"""
    # Q1: 空间质量
    edge_fraction: float = float('nan')  # 事件落在 RGB 边缘邻域的比例
    gt_rnd_ratio: float = float('nan')   # 边缘事件比例 / 边缘面积比例
    
    # Q2: 覆盖与平衡
    dark_pixel_rate: float = 0.0     # 无事件像素比例
    on_off_balance: float = 0.0      # ON/OFF 比例 (理想=1.0)
    
    # Q3: 亮度保真度
    q2_ratio: float = float('nan')   # 事件密度与 RGB 梯度的空间相关系数
    
    # Q4: 噪声
    static_noise_density: float = float('nan')  # 显式静止 mask 内 ev/(px*s)
    
    # 统计
    n_events: int = 0
    n_frames: int = 0
    
    def to_dict(self) -> dict:
        return {
            'edge_fraction': round(self.edge_fraction, 4),
            'gt_rnd_ratio': round(self.gt_rnd_ratio, 1),
            'dark_pixel_rate': round(self.dark_pixel_rate, 4),
            'on_off_balance': round(self.on_off_balance, 4),
            'q2_ratio': round(self.q2_ratio, 4),
            'static_noise_density': round(self.static_noise_density, 6),
            'n_events': self.n_events,
            'n_frames': self.n_frames,
        }


@dataclass
class ExternalMetrics:
    """PECS CD/GD 外部对齐指标"""
    cd: float = float('inf')
    gd: float = float('inf')
    n_sim: int = 0
    n_real: int = 0
    
    # vs baselines
    cd_vs_pecs: float = float('nan')
    gd_vs_pecs: float = float('nan')
    cd_vs_v2e: float = float('nan')
    gd_vs_v2e: float = float('nan')
    
    scene: str = ""
    
    def to_dict(self) -> dict:
        return {
            'cd': round(self.cd, 4),
            'gd': round(self.gd, 4),
            'cd_vs_pecs': round(self.cd_vs_pecs, 2),
            'gd_vs_pecs': round(self.gd_vs_pecs, 2),
            'cd_vs_v2e': round(self.cd_vs_v2e, 2),
            'gd_vs_v2e': round(self.gd_vs_v2e, 2),
            'n_sim': self.n_sim,
            'n_real': self.n_real,
            'scene': self.scene,
        }


@dataclass
class EvaluationReport:
    """完整评估报告"""
    preset: str = ""
    scene_name: str = ""
    timestamp: str = ""
    runtime_s: float = 0.0
    
    internal: Optional[InternalMetrics] = None
    external: Optional[ExternalMetrics] = None
    
    def summary(self) -> str:
        lines = []
        lines.append("=" * 70)
        lines.append(f"  Genesis Event Simulator — Evaluation Report")
        lines.append(f"  Preset: {self.preset}  |  Scene: {self.scene_name}")
        lines.append(f"  Time: {self.timestamp}  |  Runtime: {self.runtime_s:.1f}s")
        lines.append("=" * 70)
        
        if self.internal:
            m = self.internal
            lines.append("\n── Internal Quality Metrics ──")
            lines.append(f"  Events: {m.n_events:,}  |  Frames: {m.n_frames}")
            lines.append(f"  Edge Fraction:     {m.edge_fraction:.3%}")
            lines.append(f"  GT/Rnd Ratio:      {m.gt_rnd_ratio:.1f}")
            lines.append(f"  Dark Pixel Rate:   {m.dark_pixel_rate:.3%}")
            lines.append(f"  ON/OFF Ratio:      {m.on_off_balance:.3f} (scene-dependent)")
            lines.append(f"  Edge Correlation:  {m.q2_ratio:.3f}")
            lines.append(f"  Static Noise:      {m.static_noise_density:.6f} ev/(px*s)")
            lines.append("  Note: diagnostics require matched-scene controls; no universal pass threshold.")
        
        if self.external:
            m = self.external
            lines.append(f"\n── PECS External Alignment ({m.scene}) ──")
            lines.append(f"  Sim events: {m.n_sim:,}  |  Real events: {m.n_real:,}")
            lines.append(f"  Chamfer Distance:  {m.cd:.4f}")
            lines.append(f"  Gaussian Distance: {m.gd:.4f}")
            if np.isfinite(m.cd_vs_pecs):
                lines.append(f"  vs PECS baseline:  CD={m.cd_vs_pecs:.2f}×  GD={m.gd_vs_pecs:.2f}×")
                lines.append(f"  vs V2E baseline:   CD={m.cd_vs_v2e:.2f}×  GD={m.gd_vs_v2e:.2f}×")
            else:
                lines.append("  Published baseline comparison disabled (requires exact PECS protocol).")
        
        lines.append("=" * 70)
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        d = {
            'preset': self.preset,
            'scene': self.scene_name,
            'timestamp': self.timestamp,
            'runtime_s': self.runtime_s,
        }
        if self.internal:
            d['internal'] = self.internal.to_dict()
        if self.external:
            d['external'] = self.external.to_dict()
        return d


class EventSimEvaluator:
    """Genesis 事件仿真器统一评估器"""
    
    def __init__(self, subsample: int = 5000):
        """
        Args:
            subsample: max events for CD/GD computation (speed/accuracy tradeoff)
        """
        self.subsample = subsample
    
    # ═══════════════════════════════════════════════
    # 内部指标 (六项)
    # ═══════════════════════════════════════════════
    
    def evaluate_internal(
        self,
        events: np.ndarray,
        rgb_frame: Optional[np.ndarray] = None,
        depth_frame: Optional[np.ndarray] = None,
        static_mask: Optional[np.ndarray] = None,
        event_layout: str = 'txyp',
        timestamp_unit: str = 's',
    ) -> InternalMetrics:
        """
        Compute Genesis internal quality metrics from raw events.
        
        Args:
            events: (N, 4), layout declared by ``event_layout``
            rgb_frame: (H, W, 3) optional, for image-edge diagnostics
            depth_frame: deprecated; a single depth image cannot identify motion
            static_mask: (H,W) bool ground-truth mask known to have no image motion
        """
        events = canonicalize_events(
            events,
            layout=event_layout,
            timestamp_unit=timestamp_unit,
            sort=True,
        )
        m = InternalMetrics()
        m.n_events = len(events)
        m.n_frames = 1
        
        if len(events) == 0:
            return m
        
        # Determine resolution
        if rgb_frame is not None:
            H, W = rgb_frame.shape[:2]
        else:
            H = int(events[:, 1].max()) + 1
            W = int(events[:, 0].max()) + 1
        
        x = np.clip(events[:, 0].astype(int), 0, W - 1)
        y = np.clip(events[:, 1].astype(int), 0, H - 1)
        p = events[:, 2]
        
        # ── Event histogram ──
        hist = np.zeros((H, W), dtype=np.float32)
        np.add.at(hist, (y, x), 1)
        
        # ── Q1/Q2: RGB-edge association ──
        # The previous implementation derived edges from the event histogram
        # itself, making the metric circular. Here the reference is independent.
        if rgb_frame is not None and len(events) > 0:
            edge_mask, gradient = self._rgb_edge_reference(rgb_frame)
            total_events = float(hist.sum())
            m.edge_fraction = float(hist[edge_mask].sum() / total_events)
            random_fraction = float(edge_mask.mean())
            m.gt_rnd_ratio = (
                m.edge_fraction / random_fraction if random_fraction > 0 else float('nan')
            )
            h = np.log1p(hist).ravel()
            g = gradient.ravel()
            if h.std() > 0 and g.std() > 0:
                m.q2_ratio = float(np.corrcoef(h, g)[0, 1])
        
        # ── Q3: Dark Pixel Rate ──
        dark_pixels = (hist == 0).sum()
        m.dark_pixel_rate = dark_pixels / (H * W)
        
        # ── Q4: ON/OFF Balance ──
        n_on = (p > 0).sum()
        n_off = (p < 0).sum()
        total = n_on + n_off
        m.on_off_balance = n_on / n_off if n_off > 0 else float('inf')
        
        # ── Q6: Static Noise Density ──
        if depth_frame is not None and static_mask is None:
            logger.warning(
                "depth_frame no longer infers static regions: spatial depth "
                "differences are not temporal motion. Pass an explicit static_mask."
            )
        if static_mask is not None:
            mask = np.asarray(static_mask, dtype=bool)
            if mask.shape != (H, W):
                raise ValueError(f"static_mask shape {mask.shape} != {(H, W)}")
            duration = float(events[-1, 3] - events[0, 3])
            if mask.any() and duration > 0:
                m.static_noise_density = float(
                    hist[mask].sum() / (mask.sum() * duration)
                )
        
        return m
    
    # ═══════════════════════════════════════════════
    # 外部指标 (PECS CD/GD)
    # ═══════════════════════════════════════════════
    
    def evaluate_vs_real(
        self,
        sim_events: np.ndarray,
        real_events: np.ndarray,
        scene: str = "",
        sim_layout: str = 'txyp',
        real_layout: str = 'xypt',
        sim_timestamp_unit: str = 's',
        real_timestamp_unit: str = 's',
        compare_published_baselines: bool = False,
    ) -> ExternalMetrics:
        """
        Compute PECS CD/GD between simulated and real events.
        
        Args:
            sim_events: Genesis-generated array; layout/unit must be declared
            real_events: real-camera array; layout/unit must be declared
            scene: scene name for baseline comparison
        """
        sim_events = canonicalize_events(
            sim_events, layout=sim_layout,
            timestamp_unit=sim_timestamp_unit, sort=True,
        )
        real_events = canonicalize_events(
            real_events, layout=real_layout,
            timestamp_unit=real_timestamp_unit, sort=True,
        )
        m = ExternalMetrics()
        m.n_sim = len(sim_events)
        m.n_real = len(real_events)
        m.scene = scene
        
        if len(sim_events) == 0 or len(real_events) == 0:
            return m
        
        result = compute_both(
            sim_events, real_events,
            normalize=True,
            subsample=self.subsample,
            label=scene,
        )
        m.cd = result['cd']
        m.gd = result['gd']
        
        # Published PECS numbers are meaningful only for its calibrated optical
        # rig, sequences, crop and normalization protocol. Never compare merely
        # because a free-form scene string resembles a table row.
        if compare_published_baselines:
            if scene not in PECS_BASELINES:
                raise ValueError(
                    "published baseline comparison requested for unknown PECS "
                    f"scene {scene!r}"
                )
            comp = compare_to_baselines(m.gd, m.cd, scene)
            m.cd_vs_pecs = comp['cd_vs_pecs']
            m.gd_vs_pecs = comp['gd_vs_pecs']
            m.cd_vs_v2e = comp['cd_vs_v2e']
            m.gd_vs_v2e = comp['gd_vs_v2e']
        
        return m
    
    # ═══════════════════════════════════════════════
    # 全量评估
    # ═══════════════════════════════════════════════
    
    def full_evaluation(
        self,
        sim_events: np.ndarray,
        real_events: np.ndarray,
        rgb_frame: Optional[np.ndarray] = None,
        depth_frame: Optional[np.ndarray] = None,
        preset: str = "",
        scene: str = "",
        sim_layout: str = 'txyp',
        real_layout: str = 'xypt',
        compare_published_baselines: bool = False,
    ) -> EvaluationReport:
        """Run all metrics and produce a complete report."""
        t0 = time.time()
        
        report = EvaluationReport(
            preset=preset,
            scene_name=scene,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        
        report.internal = self.evaluate_internal(
            sim_events, rgb_frame, depth_frame, event_layout=sim_layout
        )
        report.external = self.evaluate_vs_real(
            sim_events,
            real_events,
            scene,
            sim_layout=sim_layout,
            real_layout=real_layout,
            compare_published_baselines=compare_published_baselines,
        )
        report.runtime_s = time.time() - t0
        
        return report
    
    # ═══════════════════════════════════════════════
    # Helpers
    # ═══════════════════════════════════════════════
    
    @staticmethod
    def _rgb_edge_reference(rgb_frame):
        from scipy.ndimage import sobel, binary_dilation

        gray = (
            EventSimEvaluator._to_gray(rgb_frame)
            if rgb_frame.ndim == 3 else np.asarray(rgb_frame)
        ).astype(np.float64)
        gx = sobel(gray, axis=1)
        gy = sobel(gray, axis=0)
        gradient = np.hypot(gx, gy)
        positive = gradient[gradient > 0]
        if len(positive) == 0:
            return np.zeros(gray.shape, dtype=bool), gradient
        threshold = np.percentile(positive, 80)
        # One-pixel tolerance accounts for motion between the reference frame
        # and event timestamps without deriving the target from events.
        edge_mask = binary_dilation(gradient >= threshold, iterations=1)
        return edge_mask, gradient
    
    @staticmethod
    def _to_gray(rgb):
        return (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2])


# ═══════════════════════════════════════════════
# Batch Evaluation
# ═══════════════════════════════════════════════

def batch_evaluate(
    scenes: List[dict],
    evaluator: Optional[EventSimEvaluator] = None,
) -> List[EvaluationReport]:
    """
    Batch evaluate multiple scenes.
    
    Args:
        scenes: list of dicts with keys:
            - sim_events: path to .npy or np.ndarray
            - real_events: path to .npy or np.ndarray
            - rgb_frame: optional
            - preset: str
            - scene: str
    
    Returns:
        list of EvaluationReport
    """
    if evaluator is None:
        evaluator = EventSimEvaluator()
    
    reports = []
    for i, s in enumerate(scenes):
        sim = _load_events(s['sim_events'])
        real = _load_events(s['real_events'])
        rgb = s.get('rgb_frame')
        depth = s.get('depth_frame')
        
        print(f"[{i+1}/{len(scenes)}] {s.get('scene', 'unknown')}...")
        report = evaluator.full_evaluation(
            sim, real, rgb, depth,
            preset=s.get('preset', ''),
            scene=s.get('scene', ''),
        )
        reports.append(report)
        print(report.summary())
    
    return reports


def _load_events(path_or_array):
    if isinstance(path_or_array, np.ndarray):
        return path_or_array
    if isinstance(path_or_array, str):
        if path_or_array.endswith('.npy'):
            return np.load(path_or_array)
        elif path_or_array.endswith('.npz'):
            return np.load(path_or_array)['events']
        else:
            raise ValueError(f"Unknown format: {path_or_array}")
    raise TypeError(f"Expected ndarray or str, got {type(path_or_array)}")
