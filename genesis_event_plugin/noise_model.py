"""
噪声模型配置与预设系统。

预设身份:
  clean                 — 理想/调试基线
  moderate/noisy        — 通用工程/V2E-compatible 基线，未绑定真机
  real_v2               — legacy project noise-map profile; not a complete EVK4 calibration
  evk4_nominal_1klux    — IMX636 公开 1-klux 名义点，仍非单机标定
  low_light/high_speed/overexposure — legacy stress profiles，仅供消融
"""

from dataclasses import dataclass, replace
import logging
import os
from pathlib import Path
from typing import Dict, Optional


logger = logging.getLogger(__name__)


def _real_v2_noise_map_path() -> str:
    """Resolve the calibrated map without binding the package to one host.

    ``real_v2`` is a project-level calibration artifact.  Prefer an explicit
    environment override for installed/copied deployments, then resolve it
    relative to the repository that contains this plugin.  The former version
    embedded the workstation path ``/home/科研/...`` in the preset, which made
    a remote training run silently fall back or fail even when the calibration
    file had been deployed alongside the project.
    """
    override = os.environ.get("GENESIS_EVENT_PLUGIN_REAL_V2_MAP")
    if override:
        return str(Path(override).expanduser())
    project_root = Path(__file__).resolve().parents[3]
    return str(
        project_root
        / "validation"
        / "simulator-quality"
        / "real_v2"
        / "calibration"
        / "evk4_real_noise_map_v2.npz"
    )


@dataclass
class DvsNoiseConfig:
    """DVS 像素模型噪声参数集。"""

    # ── 对比度阈值 (对数域) ──
    pos_thres: float = 0.2        # ON 阈值
    neg_thres: float = 0.2        # OFF 阈值
    sigma_thres: float = 0.02     # 阈值 mismatch (高斯 σ)

    # ── 带宽模型 ──
    cutoff_hz: float = 0.0        # 光感受器 -3dB 带宽 (0=无限)
    leak_rate_hz: float = 0.0     # 泄漏电流速率 Hz
    leak_jitter_fraction: float = 0.0  # 泄漏抖动

    # ── 散粒噪声 ──
    shot_noise_rate_hz: float = 0.0  # 散粒噪声速率 Hz
    shot_noise_inten_factor: float = 0.25  # V2E SHOT_NOISE_INTEN_FACTOR

    # ── 实验性深度噪声调制 ──
    # 距离只能通过渲染得到的辐照度间接影响光子噪声；直接按深度放大噪声
    # 没有传感器物理依据。保留该字段仅用于复现实验，所有正式预设均关闭。
    depth_noise_scale: float = 0.0

    # ── 光电噪声 ──
    photoreceptor_noise: bool = False  # 是否启用光电噪声校准
    noise_rate_cov_decades: float = 0.0

    # ── 不应期 ──
    refractory_period_s: float = 0.0  # 不应期 (秒)

    # ── 热像素 ──
    hot_pixel_rate_hz: float = 0.0   # 热像素速率 Hz

    # ── 实测逐像素噪声扩展 ──
    # Optional .npz map with a calibrated total background event rate per
    # pixel. Keeping this outside JSON avoids embedding a 1280x720 array in
    # every output metadata record.
    pixel_noise_map_path: Optional[str] = None
    # AR(1) modulation of the calibrated map for measured short-term
    # activity correlation. A zero value preserves the legacy path.
    correlated_noise_alpha: float = 0.0
    correlated_noise_scale: float = 0.0

    # ── 其他 ──
    log_input: bool = False          # 输入是否为对数 HDR
    log_eps: float = 1e-6            # 对数变换的微小偏移
    # ── 线性 HDR/radiance 输入 ──
    # radiance_input 保留输入的线性动态范围；radiance_scale 是相机增益，
    # radiance_white_level 用于带宽模型的相对强度归一化。
    radiance_input: bool = False
    radiance_scale: float = 1.0
    radiance_white_level: float = 1.0
    radiance_log_threshold: float = 20.0
    # Explicit radiometric transfer.  ``video_linlog`` preserves the legacy
    # V2E-compatible DN=20 knee; physics/radiance v3 uses ``physical_log``.
    response_mode: str = "video_linlog"
    radiometric_gain: float = 1.0
    radiometric_offset: float = 0.0
    radiometric_floor: float = 1e-6
    radiometric_calibration_status: str = "uncalibrated"
    radiometric_calibration_id: str = ""
    calibrated_transfer_path: Optional[str] = None
    # Numerical IIR convergence target; this is not an EVK4 parameter.
    sensor_eps_target: float = 0.05
    # A pluggable readout hook.  ``ideal_readout`` is intentionally the only
    # default until measured latency/refractory/dead-time data exists.
    readout_model: str = "ideal_readout"
    seed: int = 42                   # 固定传感器实例/噪声流，保证可复现

    def validate(self) -> None:
        """Validate physical and numerical parameter ranges."""
        if self.pos_thres <= 0 or self.neg_thres <= 0:
            raise ValueError("pos_thres and neg_thres must be positive")
        if self.sigma_thres < 0:
            raise ValueError("sigma_thres must be non-negative")
        if self.cutoff_hz < 0 or self.leak_rate_hz < 0:
            raise ValueError("cutoff_hz and leak_rate_hz must be non-negative")
        if self.shot_noise_rate_hz < 0 or self.refractory_period_s < 0:
            raise ValueError("noise rate and refractory period must be non-negative")
        if self.hot_pixel_rate_hz != 0:
            raise NotImplementedError(
                "hot_pixel_rate_hz needs a calibrated hot-pixel fraction/map and "
                "polarity process; non-zero values are not implemented"
            )
        if not 0 <= self.leak_jitter_fraction <= 1:
            raise ValueError("leak_jitter_fraction must be in [0, 1]")
        if self.depth_noise_scale < 0:
            raise ValueError("depth_noise_scale must be non-negative")
        if not 0 <= self.correlated_noise_alpha < 1:
            raise ValueError("correlated_noise_alpha must be in [0, 1)")
        if self.correlated_noise_scale < 0:
            raise ValueError("correlated_noise_scale must be non-negative")
        if self.log_input and self.radiance_input:
            raise ValueError("log_input and radiance_input are mutually exclusive")
        if self.radiance_scale <= 0 or self.radiance_white_level <= 0:
            raise ValueError("radiance scale and white level must be positive")
        if self.radiance_log_threshold <= 0:
            raise ValueError("radiance_log_threshold must be positive")
        if self.response_mode not in {"video_linlog", "physical_log", "calibrated_transfer"}:
            raise ValueError("unknown response_mode")
        if self.radiometric_gain <= 0 or self.radiometric_floor <= 0:
            raise ValueError("radiometric_gain and radiometric_floor must be positive")
        if self.radiometric_calibration_status not in {"uncalibrated", "effective", "calibrated"}:
            raise ValueError("invalid radiometric_calibration_status")
        if self.response_mode == "calibrated_transfer":
            if not self.calibrated_transfer_path:
                raise ValueError("calibrated_transfer requires calibrated_transfer_path")
            if self.radiometric_calibration_status == "uncalibrated":
                raise ValueError("calibrated_transfer cannot be marked uncalibrated")
        if self.sensor_eps_target <= 0:
            raise ValueError("sensor_eps_target must be positive")
        if self.readout_model not in {"ideal_readout", "none"}:
            raise ValueError(
                "only ideal_readout/none is available before measured readout calibration"
            )
        if self.pixel_noise_map_path is not None and not isinstance(
            self.pixel_noise_map_path, str
        ):
            raise ValueError("pixel_noise_map_path must be a string or None")


# ═══════════════════════════════════════════════════════════════
# 预设定义
# ═══════════════════════════════════════════════════════════════

PRESETS: Dict[str, DvsNoiseConfig] = {
    # ── legacy project noise-map profile ──
    # The external map may originate from a lab recording, but this preset
    # does not calibrate absolute radiometry, pixel response or readout and
    # must not be reported as a complete EVK4 model.
    'real_v2': DvsNoiseConfig(
        pos_thres=0.205,
        neg_thres=0.195,
        sigma_thres=0.04,
        cutoff_hz=50.0,
        leak_rate_hz=0.0,
        leak_jitter_fraction=0.0,
        shot_noise_rate_hz=0.0,
        photoreceptor_noise=False,
        noise_rate_cov_decades=0.0,
        refractory_period_s=0.0,
        depth_noise_scale=0.0,
        pixel_noise_map_path=_real_v2_noise_map_path(),
        correlated_noise_alpha=0.701838977169265,
        correlated_noise_scale=0.016,
    ),

    # ── 完美传感器 (用于调试/对比) ──
    # ── 以下 clean/noisy 严格对齐 V2E emulator.py set_dvs_params() ──
    'clean': DvsNoiseConfig(
        pos_thres=0.2,
        neg_thres=0.2,
        sigma_thres=0.02,
        cutoff_hz=0,
        leak_rate_hz=0,
        leak_jitter_fraction=0,
        noise_rate_cov_decades=0,
        shot_noise_rate_hz=0,
        refractory_period_s=0,
    ),

    # ── V2E-compatible 通用噪声基线（不代表任意具体真实传感器）──
    'noisy': DvsNoiseConfig(
        pos_thres=0.2,
        neg_thres=0.2,
        sigma_thres=0.05,
        cutoff_hz=30,                # V2E 原值
        leak_rate_hz=0.1,
        leak_jitter_fraction=0.1,
        shot_noise_rate_hz=5.0,      # V2E 原值 (5 Hz/px temporal noise)
        photoreceptor_noise=False,   # V2E 默认: Poisson 噪声, 非 photoreceptor 注入
        noise_rate_cov_decades=0.1,
        refractory_period_s=0,       # V2E 原值: 0
        depth_noise_scale=0.0,
    ),

    # ── 中等噪声 (日常开发用，介于 clean 和 noisy 之间) ──
    'moderate': DvsNoiseConfig(
        pos_thres=0.2,
        neg_thres=0.2,
        sigma_thres=0.03,
        cutoff_hz=100,                # 带宽介于 clean(∞) 和 noisy(30) 之间
        leak_rate_hz=0.05,
        leak_jitter_fraction=0.05,
        shot_noise_rate_hz=1.0,       # 噪声率 ~20-25%, 信号可见
        photoreceptor_noise=False,
        noise_rate_cov_decades=0.05,
        refractory_period_s=0,
        depth_noise_scale=0.0,
    ),

    # ── Prophesee EVK4-HD / Sony IMX636 名义工作点 ──
    # Sony 给出的标准设置标称阈值为 25%（自然对数定义），CTNU 最大 6%。
    # 这里将 6% 解释为阈值分布的保守相对 sigma 上界：0.25 * 0.06=0.015。
    # 1 klux 背景率上限 0.1 Hz/px 用作均匀一阶 background activity；
    # cutoff/refractory/latency 不可由公开 bias offset 唯一换算，因此不猜测。
    'evk4_nominal_1klux': DvsNoiseConfig(
        pos_thres=0.25,
        neg_thres=0.25,
        sigma_thres=0.015,
        cutoff_hz=0.0,
        leak_rate_hz=0.0,
        leak_jitter_fraction=0.0,
        shot_noise_rate_hz=0.1,
        shot_noise_inten_factor=1.0,
        photoreceptor_noise=False,
        noise_rate_cov_decades=0.0,
        refractory_period_s=0.0,
        depth_noise_scale=0.0,
    ),

    # ── 低光照场景 ──
    # legacy 场景条件预设：它同时改传感器参数，不能作为“只改变照度”的实验。
    'low_light': DvsNoiseConfig(
        pos_thres=0.1,
        neg_thres=0.1,
        sigma_thres=0.08,
        cutoff_hz=10.0,
        leak_rate_hz=0.05,
        shot_noise_rate_hz=0.2,       # 暗光背景噪声 ~10x 正常 (0.02→0.2)
        photoreceptor_noise=False,   # 直接 Poisson
        noise_rate_cov_decades=0.2,
        depth_noise_scale=0.0,
    ),

    # ── 高速场景 ──
    # legacy 场景条件预设：只供历史复现/压力测试。
    'high_speed': DvsNoiseConfig(
        pos_thres=0.3,
        neg_thres=0.3,
        sigma_thres=0.02,
        cutoff_hz=300.0,
        leak_rate_hz=0.0,
        shot_noise_rate_hz=0.0,
        refractory_period_s=0.0005,
    ),

    # ── 过曝光场景 ──
    # legacy 场景条件预设；降低阈值不能恢复渲染 clipping 已丢失的信息。
    'overexposure': DvsNoiseConfig(
        pos_thres=0.08,
        neg_thres=0.08,
        sigma_thres=0.1,
        cutoff_hz=0.0,
        leak_rate_hz=0.0,
        shot_noise_rate_hz=0.2,       # was 2.0, 过曝下适度噪声
    ),
}


def get_preset(name: str) -> DvsNoiseConfig:
    """获取预设配置。

    Args:
        name: 预设名称，例如 'real_v2' 或 'evk4_nominal_1klux'

    Returns:
        DvsNoiseConfig

    Raises:
        ValueError: 未知预设名
    """
    if name not in PRESETS:
        raise ValueError(
            f"Unknown preset '{name}'. "
            f"Available: {list(PRESETS.keys())}"
        )
    cfg = replace(PRESETS[name])
    cfg.validate()
    if name in {'low_light', 'high_speed', 'overexposure'}:
        logger.warning(
            "Preset '%s' is a legacy scene-conditioned stress profile, not a "
            "calibrated physical sensor. Do not compare it against another "
            "scene preset as if only illumination or motion had changed.",
            name,
        )
    elif name == 'evk4_nominal_1klux':
        logger.warning(
            "Preset 'evk4_nominal_1klux' only applies public IMX636 nominal "
            "threshold/CTNU and the 1-klux background-rate limit. Bandwidth, "
            "latency, refractory period, lens distortion and unit variation "
            "still require physical EVK4 calibration."
        )
    return cfg


def custom_config(**kwargs) -> DvsNoiseConfig:
    """从关键字参数构建自定义配置。

    Example:
        cfg = custom_config(pos_thres=0.15, cutoff_hz=50, shot_noise_rate_hz=3.0)
    """
    cfg = DvsNoiseConfig(**kwargs)
    cfg.validate()
    return cfg
