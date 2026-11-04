"""
DVS 像素模型仿真器。

封装 V2E emulator_utils 的函数为有状态的 EventEmulator 类。

管线:
  new_frame → rescale → lin_log → low_pass → [diff] → event_map
  → [refractory] → shot_noise → base_log_update → output

References:
  Delbruck, Hu, He. "V2E: Video to Events." CVPRW 2021.
  Graca & Delbruck. "Intensity-Dependent DVS Pixel Noise." arXiv 2021.
"""

import logging
import math
import numpy as np
import torch
from dataclasses import replace

from .utils.dvs_core import (
    lin_log,
    rescale_intensity_frame,
    low_pass_filter,
    compute_event_map,
    subtract_leak_current,
    generate_shot_noise,
    compute_photoreceptor_noise_voltage,
)
from .noise_model import DvsNoiseConfig, get_preset
from .radiometry import load_calibrated_transfer
from .readout import build_readout_model

logger = logging.getLogger(__name__)


class DvsEmulator:
    """
    DVS 像素模型: 灰度帧 → 事件流 [t, x, y, p]。

    usage:
        emu = DvsEmulator(res=(240, 320), preset='real_v2')
        emu.initialize(first_frame, t=0.0)
        for frame, t in frames:
            events = emu.generate_events(frame, t)
    """

    def __init__(
        self,
        res: tuple = (240, 320),
        config: DvsNoiseConfig = None,
        preset: str = 'real_v2',
        device: str = None,
    ):
        """
        Args:
            res: (H, W) 图像分辨率
            config: DvsNoiseConfig 实例 (若提供则覆盖 preset)
            preset: 预设名，如 'clean'、'noisy'、'evk4_nominal_1klux'
            device: torch device ('cpu' | 'cuda:0'), None=auto
        """
        if device is None:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.height, self.width = res

        if config is not None:
            self.cfg = replace(config)
        else:
            self.cfg = get_preset(preset)
        self.cfg.validate()
        self.rng = torch.Generator(device=self.device).manual_seed(int(self.cfg.seed))
        self.readout_model = build_readout_model(self.cfg.readout_model)
        self.transfer_input = None
        self.transfer_log_response = None
        if self.cfg.response_mode == "calibrated_transfer":
            x, y = load_calibrated_transfer(self.cfg.calibrated_transfer_path)
            self.transfer_input = torch.as_tensor(x, dtype=torch.float64, device=self.device)
            self.transfer_log_response = torch.as_tensor(y, dtype=torch.float64, device=self.device)

        # ── 内部状态 ──
        self.base_log_frame = None       # 参考对数强度
        self.lp_log_frame = None         # 低通滤波输出
        self.new_frame = None            # 当前帧 (torch)
        self.log_new_frame = None        # 对数帧
        self.diff_frame = None           # 差分
        self.photoreceptor_noise_arr = None
        self.photoreceptor_noise_vrms = None
        self.noise_rate_array = None

        # ── 不应期状态 ──
        self.timestamp_mem = None        # 上次事件时间戳 (H,W)

        # ── 阈值数组 (逐像素 mismatch) ──
        self.pos_thres = None
        self.neg_thres = None

        # ── 统计 ──
        self.t_previous = None
        self.t_start = None
        self.num_events_total = 0
        self.num_events_on = 0
        self.num_events_off = 0
        self.frame_counter = 0
        self._warned_depth_noise = False

        self._initialized = False
        self.calibrated_noise_rate_map = None
        self.calibrated_noise_on_probability = None
        self.calibrated_hot_pixel_mask = None
        self.correlated_noise_state = None
        self._load_calibrated_noise_map()

    def _load_calibrated_noise_map(self):
        """Load optional measured per-pixel background activity.

        The map is deliberately an external ``.npz`` artifact so a normal
        preset remains lightweight and backward compatible.  ``rate_hz`` is
        the total measured background event rate per pixel; polarity is
        supplied independently through ``on_probability`` when available.
        """
        self.calibrated_noise_rate_map = None
        self.calibrated_noise_on_probability = None
        self.calibrated_hot_pixel_mask = None
        self.correlated_noise_state = None
        path = self.cfg.pixel_noise_map_path
        if path is None:
            return

        with np.load(path) as data:
            if "rate_hz" not in data:
                raise ValueError(f"calibrated noise map {path!r} lacks 'rate_hz'")
            rate = np.asarray(data["rate_hz"], dtype=np.float64)
            if rate.shape != (self.height, self.width):
                src_h, src_w = rate.shape
                if src_h % self.height or src_w % self.width:
                    raise ValueError(
                        f"calibrated noise map shape {rate.shape} cannot be "
                        f"area-reduced to sensor {(self.height, self.width)}"
                    )
                sy, sx = src_h // self.height, src_w // self.width
                rate = rate.reshape(self.height, sy, self.width, sx).mean(axis=(1, 3))
                logger.info(
                    "Area-reduced calibrated noise map from %s to sensor %s",
                    (src_h, src_w), (self.height, self.width),
                )
            if not np.isfinite(rate).all() or (rate < 0).any():
                raise ValueError("calibrated noise rate map must be finite and non-negative")
            self.calibrated_noise_rate_map = torch.as_tensor(
                rate, dtype=torch.float64, device=self.device
            )

            if "on_probability" in data:
                on_probability = np.asarray(data["on_probability"], dtype=np.float64)
                source_shape = data["on_probability"].shape
                if on_probability.shape != source_shape:
                    raise ValueError("on_probability has an invalid shape")
                if on_probability.shape != (self.height, self.width):
                    src_h, src_w = on_probability.shape
                    sy, sx = src_h // self.height, src_w // self.width
                    on_probability = on_probability.reshape(
                        self.height, sy, self.width, sx
                    )
                    # Weight polarity by the original activity rate so that
                    # high-rate pixels dominate each reduced display pixel.
                    source_rate = np.asarray(data["rate_hz"], dtype=np.float64)
                    source_rate = source_rate.reshape(
                        self.height, sy, self.width, sx
                    )
                    numerator = (source_rate * on_probability).mean(axis=(1, 3))
                    denominator = source_rate.mean(axis=(1, 3))
                    on_probability = np.divide(
                        numerator, denominator, out=np.full_like(numerator, 0.5),
                        where=denominator > 0,
                    )
                if not np.isfinite(on_probability).all():
                    raise ValueError("on_probability must be finite")
                self.calibrated_noise_on_probability = torch.as_tensor(
                    np.clip(on_probability, 0.0, 1.0),
                    dtype=torch.float64,
                    device=self.device,
                )

            if "hot_pixel_mask" in data:
                hot_mask = np.asarray(data["hot_pixel_mask"], dtype=bool)
                if hot_mask.shape != data["hot_pixel_mask"].shape:
                    raise ValueError("hot_pixel_mask has an invalid shape")
                if hot_mask.shape != (self.height, self.width):
                    src_h, src_w = hot_mask.shape
                    sy, sx = src_h // self.height, src_w // self.width
                    hot_mask = hot_mask.reshape(
                        self.height, sy, self.width, sx
                    ).any(axis=(1, 3))
                self.calibrated_hot_pixel_mask = torch.as_tensor(
                    hot_mask, dtype=torch.bool, device=self.device
                )

    # ════════════════════════════════════════════════════════
    # 初始化
    # ════════════════════════════════════════════════════════

    def _prepare_input(self, frame: np.ndarray) -> np.ndarray:
        """Convert supported intensity inputs without dtype-dependent scaling.

        Non-log inputs may be uint/integer ``[0,255]``, float ``[0,1]`` or
        float ``[0,255]``.  ``log_input=True`` means the caller already supplies
        log irradiance and no scaling is applied.
        """
        arr = np.asarray(frame)
        if arr.shape != (self.height, self.width):
            raise ValueError(
                f"frame shape {arr.shape} does not match sensor {(self.height, self.width)}"
            )
        if not np.isfinite(arr).all():
            raise ValueError("frame contains NaN or infinite values")
        out = arr.astype(np.float64, copy=False)
        if self.cfg.log_input:
            return out
        if out.min(initial=0.0) < 0:
            raise ValueError("linear intensity input must be non-negative")
        if self.cfg.radiance_input:
            # Keep HDR values in the linear domain.  The gain maps physical
            # radiance units into the calibrated sensor-response units used by
            # the lin-log model; no 8-bit clipping is allowed here.
            return out * float(self.cfg.radiance_scale)
        max_value = float(out.max(initial=0.0))
        if np.issubdtype(arr.dtype, np.floating) and max_value <= 1.0 + 1e-6:
            out = out * 255.0
        elif max_value > 255.0 + 1e-6:
            raise ValueError(
                "non-log intensity exceeds 255; normalize linear radiance or "
                "set log_input=True for precomputed log irradiance"
            )
        return out

    def _to_log_input(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Map the selected input domain into the sensor log-response domain."""
        if self.cfg.log_input:
            return pixel_values
        if self.cfg.response_mode == "physical_log":
            response = (
                float(self.cfg.radiometric_gain) * pixel_values
                + float(self.cfg.radiometric_offset)
            )
            return torch.log(torch.clamp(response, min=float(self.cfg.radiometric_floor)))
        if self.cfg.response_mode == "calibrated_transfer":
            if self.transfer_input is None or self.transfer_log_response is None:
                raise RuntimeError("calibrated transfer table was not loaded")
            if bool((pixel_values < self.transfer_input[0]).any()) or bool(
                (pixel_values > self.transfer_input[-1]).any()
            ):
                raise RuntimeError(
                    "input is outside calibrated transfer range; refusing to clip"
                )
            idx = torch.searchsorted(self.transfer_input, pixel_values, right=True).clamp(
                1, self.transfer_input.numel() - 1
            )
            x0 = self.transfer_input[idx - 1]
            x1 = self.transfer_input[idx]
            y0 = self.transfer_log_response[idx - 1]
            y1 = self.transfer_log_response[idx]
            weight = (pixel_values - x0) / torch.clamp(x1 - x0, min=1e-12)
            return y0 + weight * (y1 - y0)
        if self.cfg.radiance_input:
            return lin_log(
                pixel_values,
                threshold=float(self.cfg.radiance_log_threshold),
            )
        return lin_log(pixel_values)

    def _intensity_fraction(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Relative linear intensity for bandwidth/noise modulation."""
        if self.cfg.radiance_input:
            return torch.clamp(
                pixel_values / float(self.cfg.radiance_white_level), 0.0, 1.0
            )
        return rescale_intensity_frame(pixel_values)

    def initialize(self, first_frame: np.ndarray, t: float = 0.0):
        """
        用第一帧初始化所有内部状态。不产生事件。

        Args:
            first_frame: (H, W) uint8 或 float32 灰度帧
            t: 时间戳 (秒)
        """
        pixel_vals = self._prepare_input(first_frame)

        self.new_frame = torch.tensor(pixel_vals, dtype=torch.float64, device=self.device)
        self.log_new_frame = self._to_log_input(self.new_frame)
        self.lp_log_frame = self.log_new_frame.clone()
        self.base_log_frame = self.lp_log_frame.clone()

        # 不应期状态
        self.timestamp_mem = torch.full(
            (self.height, self.width),
            float(t) - self.cfg.refractory_period_s,
            dtype=torch.float64,
            device=self.device,
        )

        # 生成随机阈值 (mismatch)
        self._init_thresholds()

        # Fixed-pattern leak-rate mismatch, matching v2e's log-normal model.
        if self.cfg.leak_rate_hz > 0 and self.cfg.noise_rate_cov_decades > 0:
            log_rate = torch.randn(
                (self.height, self.width),
                dtype=torch.float64,
                device=self.device,
                generator=self.rng,
            )
            self.noise_rate_array = torch.exp(
                math.log(10.0) * self.cfg.noise_rate_cov_decades * log_rate
            )
        else:
            self.noise_rate_array = torch.ones_like(self.base_log_frame)

        # 光电噪声初始化
        self.photoreceptor_noise_arr = torch.zeros_like(self.lp_log_frame)
        if self.calibrated_noise_rate_map is not None:
            # One scalar AR(1) state models the short, shared activity bursts
            # seen in the static white-wall recording.  A per-pixel state would
            # average out over 921,600 pixels and fail to reproduce that
            # measured frame-to-frame correlation.
            self.correlated_noise_state = torch.zeros(
                (), dtype=torch.float64, device=self.device
            )
        self.t_previous = t
        self.t_start = t
        self.frame_counter = 1
        self._initialized = True

    def _init_thresholds(self):
        """生成逐像素对比度阈值 (高斯 mismatch)。"""
        shape = (self.height, self.width)

        self.pos_thres_nominal = self.cfg.pos_thres  # V2E: 保存 nominal 用于 pre_prob
        self.neg_thres_nominal = self.cfg.neg_thres
        if self.cfg.sigma_thres > 0:
            self.pos_thres = torch.normal(
                self.cfg.pos_thres, self.cfg.sigma_thres,
                size=shape, generator=self.rng, device=self.device, dtype=torch.float64
            )
            self.neg_thres = torch.normal(
                self.cfg.neg_thres, self.cfg.sigma_thres,
                size=shape, generator=self.rng, device=self.device, dtype=torch.float64
            )
        else:
            self.pos_thres = torch.full(shape, self.cfg.pos_thres, dtype=torch.float64, device=self.device)
            self.neg_thres = torch.full(shape, self.cfg.neg_thres, dtype=torch.float64, device=self.device)

        # 确保阈值 > 0
        self.pos_thres = torch.clamp(self.pos_thres, min=0.01)
        self.neg_thres = torch.clamp(self.neg_thres, min=0.01)

    def reset(self):
        """重置所有内部状态，准备处理新视频/场景。"""
        self.base_log_frame = None
        self.lp_log_frame = None
        self.timestamp_mem = None
        self.noise_rate_array = None
        self.calibrated_noise_rate_map = None
        self.calibrated_noise_on_probability = None
        self.calibrated_hot_pixel_mask = None
        self.correlated_noise_state = None
        self.t_previous = None
        self.t_start = None
        self.num_events_total = 0
        self.num_events_on = 0
        self.num_events_off = 0
        self.frame_counter = 0
        self._warned_depth_noise = False
        self.rng = torch.Generator(device=self.device).manual_seed(int(self.cfg.seed))
        self._load_calibrated_noise_map()
        self._initialized = False

    # ════════════════════════════════════════════════════════
    # 事件生成 (主循环)
    # ════════════════════════════════════════════════════════

    def generate_events(self, frame: np.ndarray, t: float, depth_map: np.ndarray = None) -> np.ndarray:
        """
        单帧 → 事件流。

        Args:
            frame: (H, W) uint8/float32 灰度帧
            t: 时间戳 (秒)

        Returns:
            events: (N, 4) float64 [t, x, y, p], p∈{+1,-1}
                    若无事件返回形状为 (0,4) 的空数组

        Raises:
            RuntimeError: 若未调用 initialize()
        """
        if not self._initialized:
            raise RuntimeError("DvsEmulator not initialized. Call initialize() first.")

        if t <= self.t_previous:
            raise ValueError(f"Time must increase strictly: {t} <= {self.t_previous}")

        # ── 预处理 ──
        pixel_vals = self._prepare_input(frame)
        self.new_frame = torch.tensor(pixel_vals, dtype=torch.float64, device=self.device)

        # ── Step 1: lin-log 映射 ──
        self.log_new_frame = self._to_log_input(self.new_frame)

        delta_time = t - self.t_previous
        signal_start = self.lp_log_frame + self.photoreceptor_noise_arr
        base_start = self.base_log_frame.clone()

        # ── Step 2: 强度缩放 ──
        inten01 = None
        if (
            self.cfg.cutoff_hz > 0
            or self.cfg.shot_noise_rate_hz > 0
            or self.calibrated_noise_rate_map is not None
        ):
            inten01 = self._intensity_fraction(self.new_frame.clone().detach())

        # ── Step 3: 低通滤波 ──
        self.lp_log_frame = low_pass_filter(
            log_new_frame=self.log_new_frame,
            lp_log_frame=self.lp_log_frame,
            inten01=inten01,
            delta_time=delta_time,
            cutoff_hz=self.cfg.cutoff_hz,
        )

        # ── Step 4: 光电噪声 ──
        if self.cfg.photoreceptor_noise:
            if self.cfg.cutoff_hz <= 0 or self.cfg.shot_noise_rate_hz <= 0:
                raise ValueError(
                    "photoreceptor_noise requires positive cutoff_hz and "
                    "shot_noise_rate_hz"
                )
            self.photoreceptor_noise_vrms = compute_photoreceptor_noise_voltage(
                shot_noise_rate_hz=self.cfg.shot_noise_rate_hz,
                f3db=self.cfg.cutoff_hz,
                sample_rate_hz=1.0 / delta_time,
                pos_thr=self.cfg.pos_thres,
                neg_thr=self.cfg.neg_thres,
                sigma_thr=self.cfg.sigma_thres,
            )
            noise = self.photoreceptor_noise_vrms * torch.randn(
                self.log_new_frame.shape, dtype=torch.float32,
                device=self.device, generator=self.rng,
            )
            self.photoreceptor_noise_arr = low_pass_filter(
                noise, self.photoreceptor_noise_arr, None, delta_time, self.cfg.cutoff_hz
            )

        # ── Step 5: 泄漏电流 ──
        if self.cfg.leak_rate_hz > 0:
            self.base_log_frame = subtract_leak_current(
                base_log_frame=self.base_log_frame,
                leak_rate_hz=self.cfg.leak_rate_hz,
                delta_time=delta_time,
                pos_thres=self.pos_thres,
                leak_jitter_fraction=self.cfg.leak_jitter_fraction,
                noise_rate_array=self.noise_rate_array,
                generator=self.rng,
            )

        # ── Step 6: 差分 → 事件量化 ──
        signal_end = self.lp_log_frame + self.photoreceptor_noise_arr
        self.diff_frame = signal_end - self.base_log_frame
        relative_start = signal_start - base_start
        relative_end = self.diff_frame

        pos_evts_frame, neg_evts_frame = compute_event_map(
            self.diff_frame, self.pos_thres, self.neg_thres
        )

        max_events = max(pos_evts_frame.max().item(), neg_evts_frame.max().item(), 1)

        # ── Step 7: 展开事件坐标 + 不应期过滤 ──
        events_list = []
        final_pos_evts_frame = torch.zeros_like(pos_evts_frame)
        final_neg_evts_frame = torch.zeros_like(neg_evts_frame)
        relative_delta = relative_end - relative_start
        stable_delta = torch.where(
            torch.abs(relative_delta) > 1e-12,
            relative_delta,
            torch.ones_like(relative_delta),
        )

        for i in range(max_events):
            pos_cord = (pos_evts_frame >= i + 1)
            neg_cord = (neg_evts_frame >= i + 1)

            # Localize each pixel's threshold crossing on its own continuous
            # interval trajectory. This removes the global timestamp layers
            # produced by assigning every pixel the same i/max_events time.
            pos_target = (i + 1) * self.pos_thres
            neg_target = -(i + 1) * self.neg_thres
            pos_alpha = torch.clamp(
                (pos_target - relative_start) / stable_delta, 0.0, 1.0
            )
            neg_alpha = torch.clamp(
                (neg_target - relative_start) / stable_delta, 0.0, 1.0
            )
            fallback_alpha = float(i + 1) / max_events
            pos_alpha = torch.where(
                torch.abs(relative_delta) > 1e-12,
                pos_alpha,
                torch.full_like(pos_alpha, fallback_alpha),
            )
            neg_alpha = torch.where(
                torch.abs(relative_delta) > 1e-12,
                neg_alpha,
                torch.full_like(neg_alpha, fallback_alpha),
            )
            pos_ts = self.t_previous + delta_time * pos_alpha
            neg_ts = self.t_previous + delta_time * neg_alpha

            # 不应期
            if self.cfg.refractory_period_s > 0:
                pos_cord = pos_cord & (
                    pos_ts - self.timestamp_mem >= self.cfg.refractory_period_s
                )
                neg_cord = neg_cord & (
                    neg_ts - self.timestamp_mem >= self.cfg.refractory_period_s
                )

                self.timestamp_mem = torch.where(pos_cord, pos_ts, self.timestamp_mem)
                self.timestamp_mem = torch.where(neg_cord, neg_ts, self.timestamp_mem)

            final_pos_evts_frame += pos_cord.to(torch.int32)
            final_neg_evts_frame += neg_cord.to(torch.int32)

            # 生成事件坐标
            if pos_cord.any():
                ys, xs = pos_cord.nonzero(as_tuple=True)
                pos_events = torch.stack([
                    pos_ts[pos_cord],
                    xs.double(), ys.double(),
                    torch.ones_like(xs, dtype=torch.float64),
                ], dim=1)
                events_list.append(pos_events)

            if neg_cord.any():
                ys, xs = neg_cord.nonzero(as_tuple=True)
                neg_events = torch.stack([
                    neg_ts[neg_cord],
                    xs.double(), ys.double(),
                    -torch.ones_like(xs, dtype=torch.float64),
                ], dim=1)
                events_list.append(neg_events)

        # ── Step 8: 散粒/背景活动噪声 ──
        accepted_shot_on = torch.zeros_like(pos_evts_frame, dtype=torch.bool)
        accepted_shot_off = torch.zeros_like(neg_evts_frame, dtype=torch.bool)
        if (
            (self.cfg.shot_noise_rate_hz > 0 or self.calibrated_noise_rate_map is not None)
            and not self.cfg.photoreceptor_noise
        ):
            if self.calibrated_noise_rate_map is not None:
                effective_rate_map = self.calibrated_noise_rate_map
                if (
                    self.cfg.correlated_noise_alpha > 0
                    and self.cfg.correlated_noise_scale > 0
                ):
                    alpha = self.cfg.correlated_noise_alpha
                    innovation = torch.randn(
                        (), dtype=torch.float64, device=self.device, generator=self.rng
                    )
                    self.correlated_noise_state = (
                        alpha * self.correlated_noise_state
                        + math.sqrt(1.0 - alpha * alpha) * innovation
                    )
                    scale = self.cfg.correlated_noise_scale
                    gain = torch.exp(scale * self.correlated_noise_state - 0.5 * scale * scale)
                    effective_rate_map = effective_rate_map * gain
                # The legacy Bernoulli approximation emits at most one event
                # per pixel and interval. Keep its probability below one even
                # for a measured hot pixel or a positive correlation burst.
                effective_rate_map = torch.clamp(
                    effective_rate_map, min=0.0, max=0.9 / max(delta_time, 1e-12)
                )
            else:
                effective_rate = self.cfg.shot_noise_rate_hz
                if depth_map is not None and self.cfg.depth_noise_scale > 0:
                    if not self._warned_depth_noise:
                        logger.warning(
                            "depth_noise_scale is an unsupported legacy ablation: "
                            "distance should affect noise through rendered irradiance, "
                            "not an extra depth multiplier."
                        )
                        self._warned_depth_noise = True
                    depth_t = torch.from_numpy(depth_map.astype(np.float32)).to(self.device)
                    d_max = depth_t.max()
                    if d_max > 0:
                        d_norm = depth_t / d_max  # [0, 1]
                        # 远处噪声 = base × (1 + scale × d_norm), 近处=base
                        effective_rate_map = self.cfg.shot_noise_rate_hz * (
                            1.0 + self.cfg.depth_noise_scale * d_norm)
                    else:
                        effective_rate_map = torch.full_like(depth_t, self.cfg.shot_noise_rate_hz)
                else:
                    effective_rate_map = effective_rate  # 标量, generate_shot_noise 会广播

            shot_on, shot_off = generate_shot_noise(
                shot_noise_rate_hz=effective_rate_map,
                delta_time=delta_time,
                shot_noise_inten_factor=self.cfg.shot_noise_inten_factor,
                inten01=inten01 if inten01 is not None else torch.ones_like(self.lp_log_frame),
                pos_thres_pre_prob=torch.div(self.pos_thres_nominal, self.pos_thres),
                neg_thres_pre_prob=torch.div(self.neg_thres_nominal, self.neg_thres),
                generator=self.rng,
            )

            if self.calibrated_noise_on_probability is not None:
                any_shot = shot_on | shot_off
                polarity_draw = torch.rand(
                    self.lp_log_frame.shape,
                    dtype=torch.float64,
                    device=self.device,
                    generator=self.rng,
                )
                shot_on = any_shot & (
                    polarity_draw < self.calibrated_noise_on_probability
                )
                shot_off = any_shot & ~shot_on

            # A Bernoulli approximation emits at most one noise event per pixel
            # per interval. Place it uniformly inside the interval instead of
            # stamping every noise event at the frame endpoint.
            shot_off = shot_off & ~shot_on
            noise_ts = self.t_previous + delta_time * torch.rand(
                self.lp_log_frame.shape,
                dtype=torch.float64,
                device=self.device,
                generator=self.rng,
            )
            if self.cfg.refractory_period_s > 0:
                allowed = (
                    noise_ts - self.timestamp_mem
                    >= self.cfg.refractory_period_s
                )
                shot_on = shot_on & allowed
                shot_off = shot_off & allowed

            accepted_shot_on = shot_on
            accepted_shot_off = shot_off
            self.timestamp_mem = torch.where(
                accepted_shot_on | accepted_shot_off,
                noise_ts,
                self.timestamp_mem,
            )

            if accepted_shot_on.any():
                ys, xs = accepted_shot_on.nonzero(as_tuple=True)
                noise_events = torch.stack([
                    noise_ts[accepted_shot_on],
                    xs.double(), ys.double(),
                    torch.ones_like(xs, dtype=torch.float64),
                ], dim=1)
                events_list.append(noise_events)
            if accepted_shot_off.any():
                ys, xs = accepted_shot_off.nonzero(as_tuple=True)
                noise_events = torch.stack([
                    noise_ts[accepted_shot_off],
                    xs.double(), ys.double(),
                    -torch.ones_like(xs, dtype=torch.float64),
                ], dim=1)
                events_list.append(noise_events)

        # ── 合并事件 ──
        if events_list:
            events = torch.cat(events_list, dim=0).cpu().numpy()
            # 按时间戳排序 (事件处理算法的标准期望)
            events = events[events[:, 0].argsort()]
        else:
            events = np.empty((0, 4), dtype=np.float64)
        events = self.readout_model.process(events, self.t_previous, t)

        # ── Step 9: 更新参考值 ──
        # Advance the comparator reference only by events that survived the
        # refractory model. Suppressed events must remain as residual contrast.
        self.base_log_frame += final_pos_evts_frame.float() * self.pos_thres
        self.base_log_frame -= final_neg_evts_frame.float() * self.neg_thres
        # A background-activity event resets the local comparator state, as in
        # the v2e reference implementation.
        shot_mask = accepted_shot_on | accepted_shot_off
        self.base_log_frame = torch.where(
            shot_mask, signal_end, self.base_log_frame
        )

        # ── 统计 ──
        n_on = int((events[:, 3] > 0).sum()) if len(events) > 0 else 0
        n_off = len(events) - n_on
        self.num_events_on += n_on
        self.num_events_off += n_off
        self.num_events_total += len(events)

        self.t_previous = t
        self.frame_counter += 1

        return events

    def get_stats(self) -> dict:
        """返回累计统计信息。"""
        total_time = (
            self.t_previous - self.t_start
            if self.t_previous is not None and self.t_start is not None
            else 0.0
        )
        return {
            'total_events': self.num_events_total,
            'on_events': self.num_events_on,
            'off_events': self.num_events_off,
            'total_time_s': total_time,
            'rate_hz': self.num_events_total / total_time if total_time > 0 else 0,
            'frames_processed': self.frame_counter,
        }


class DvsBatchEmulator:
    """Experimental tensor-native ideal DVS state for ``[B,H,W]`` batches.

    This class is deliberately narrower than :class:`DvsEmulator`: it keeps
    the event quantizer and radiometric transfer on-device, but refuses
    features that still require a per-environment stochastic/readout kernel
    (photoreceptor noise, shot/background maps, leakage, refractory timing and
    non-ideal readout).  It returns packed ``[batch,t,x,y,p]`` events plus a
    device-resident ``B+1`` offset vector, which is suitable for downstream
    training without a CPU round-trip.  The reference single-environment
    emulator remains the source of truth for full sensor ablations.
    """

    def __init__(
        self,
        res: tuple,
        config: DvsNoiseConfig = None,
        preset: str = "clean",
        device: str = None,
        max_events_per_pixel: int = 64,
    ):
        self.device = torch.device(
            device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        self.height, self.width = int(res[0]), int(res[1])
        self.cfg = replace(config) if config is not None else get_preset(preset)
        self.cfg.validate()
        unsupported = {
            "cutoff_hz": self.cfg.cutoff_hz,
            "leak_rate_hz": self.cfg.leak_rate_hz,
            "shot_noise_rate_hz": self.cfg.shot_noise_rate_hz,
            "refractory_period_s": self.cfg.refractory_period_s,
            "pixel_noise_map_path": self.cfg.pixel_noise_map_path,
            "photoreceptor_noise": self.cfg.photoreceptor_noise,
            "readout_model": self.cfg.readout_model,
        }
        if any(value not in (0, 0.0, False, None, "ideal_readout", "none")
               for value in unsupported.values()):
            raise NotImplementedError(
                "DvsBatchEmulator currently supports the noiseless, no-IIR, "
                "no-refractory ideal-readout subset; use DvsEmulator for full noise"
            )
        if max_events_per_pixel < 1:
            raise ValueError("max_events_per_pixel must be positive")
        self.max_events_per_pixel = int(max_events_per_pixel)
        self.rng = torch.Generator(device=self.device).manual_seed(int(self.cfg.seed))
        self.base_log_frame = None
        self.t_previous = None
        self.pos_thres = None
        self.neg_thres = None
        self.batch_size = None
        self.transfer_input = None
        self.transfer_log_response = None
        if self.cfg.response_mode == "calibrated_transfer":
            x, y = load_calibrated_transfer(self.cfg.calibrated_transfer_path)
            self.transfer_input = torch.as_tensor(x, dtype=torch.float64, device=self.device)
            self.transfer_log_response = torch.as_tensor(y, dtype=torch.float64, device=self.device)

    def _prepare(self, frames: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(frames):
            raise TypeError("frames must be a torch.Tensor")
        if frames.ndim != 3 or tuple(frames.shape[1:]) != (self.height, self.width):
            raise ValueError(f"frames must have shape [B,{self.height},{self.width}]")
        input_was_float = frames.dtype.is_floating_point
        values = frames.to(device=self.device, dtype=torch.float64)
        if not torch.isfinite(values).all() or bool((values < 0).any()):
            raise ValueError("batch intensity must be finite and non-negative")
        if not self.cfg.radiance_input and self.cfg.response_mode == "video_linlog":
            if input_was_float and bool(values.max() <= 1.0 + 1e-6):
                values = values * 255.0
            if bool(values.max() > 255.0 + 1e-6):
                raise ValueError("video_linlog batch input must be in [0,255]")
        if self.cfg.radiance_input:
            values = values * float(self.cfg.radiance_scale)
        return values

    def _to_log(self, values: torch.Tensor) -> torch.Tensor:
        if self.cfg.log_input:
            return values
        if self.cfg.response_mode == "physical_log":
            response = float(self.cfg.radiometric_gain) * values + float(self.cfg.radiometric_offset)
            return torch.log(torch.clamp(response, min=float(self.cfg.radiometric_floor)))
        if self.cfg.response_mode == "calibrated_transfer":
            if bool((values < self.transfer_input[0]).any()) or bool((values > self.transfer_input[-1]).any()):
                raise RuntimeError("batch input is outside calibrated transfer range")
            idx = torch.searchsorted(self.transfer_input, values, right=True).clamp(
                1, self.transfer_input.numel() - 1
            )
            x0, x1 = self.transfer_input[idx - 1], self.transfer_input[idx]
            y0, y1 = self.transfer_log_response[idx - 1], self.transfer_log_response[idx]
            weight = (values - x0) / torch.clamp(x1 - x0, min=1e-12)
            return y0 + weight * (y1 - y0)
        return lin_log(values, threshold=float(self.cfg.radiance_log_threshold))

    def initialize_batch(self, frames: torch.Tensor, t: float = 0.0):
        values = self._prepare(frames)
        self.batch_size = values.shape[0]
        self.base_log_frame = self._to_log(values)
        shape = self.base_log_frame.shape
        if self.cfg.sigma_thres > 0:
            self.pos_thres = torch.normal(
                self.cfg.pos_thres, self.cfg.sigma_thres, size=shape,
                generator=self.rng, device=self.device, dtype=torch.float64,
            ).clamp_min(0.01)
            self.neg_thres = torch.normal(
                self.cfg.neg_thres, self.cfg.sigma_thres, size=shape,
                generator=self.rng, device=self.device, dtype=torch.float64,
            ).clamp_min(0.01)
        else:
            self.pos_thres = torch.full(shape, self.cfg.pos_thres, dtype=torch.float64, device=self.device)
            self.neg_thres = torch.full(shape, self.cfg.neg_thres, dtype=torch.float64, device=self.device)
        self.t_previous = torch.full((self.batch_size,), float(t), dtype=torch.float64, device=self.device)

    def generate_events_batch(self, frames: torch.Tensor, t) -> tuple:
        """Advance all environments and return ``(packed_events, offsets)``.

        ``t`` is a scalar or a ``[B]`` tensor in seconds.  Both returned
        tensors stay on ``self.device``.  ``offsets[b]:offsets[b+1]`` indexes
        events for environment ``b``; packed columns are ``batch,t,x,y,p``.
        """
        if self.base_log_frame is None:
            raise RuntimeError("call initialize_batch() first")
        values = self._prepare(frames)
        if values.shape[0] != self.batch_size:
            raise ValueError("batch size cannot change after initialization")
        times = torch.as_tensor(t, dtype=torch.float64, device=self.device)
        if times.ndim == 0:
            times = times.expand(self.batch_size)
        if times.shape != (self.batch_size,):
            raise ValueError("t must be scalar or shape [B]")
        delta = times - self.t_previous
        if bool((delta <= 0).any()):
            raise ValueError("all batch timestamps must increase strictly")
        log_frame = self._to_log(values)
        diff = log_frame - self.base_log_frame
        pos_count = torch.floor(torch.relu(diff) / self.pos_thres).to(torch.int64)
        neg_count = torch.floor(torch.relu(-diff) / self.neg_thres).to(torch.int64)
        if bool(torch.maximum(pos_count, neg_count).max() > self.max_events_per_pixel):
            raise RuntimeError(
                "max_events_per_pixel exceeded; increase the explicit bound or "
                "reduce the interval instead of silently truncating events"
            )
        relative = diff
        safe_relative = torch.where(relative.abs() > 1e-12, relative, torch.ones_like(relative))
        chunks = []
        batch_grid = torch.arange(self.batch_size, device=self.device, dtype=torch.float64)[:, None, None]
        batch_grid = batch_grid.expand(self.batch_size, self.height, self.width)
        yy, xx = torch.meshgrid(
            torch.arange(self.height, device=self.device, dtype=torch.float64),
            torch.arange(self.width, device=self.device, dtype=torch.float64), indexing="ij"
        )
        xx = xx.expand(self.batch_size, -1, -1)
        yy = yy.expand(self.batch_size, -1, -1)
        for i in range(self.max_events_per_pixel):
            pos_mask = pos_count >= (i + 1)
            neg_mask = neg_count >= (i + 1)
            pos_alpha = torch.clamp(((i + 1) * self.pos_thres) / safe_relative, 0.0, 1.0)
            neg_alpha = torch.clamp((-(i + 1) * self.neg_thres) / safe_relative, 0.0, 1.0)
            pos_alpha = torch.where(relative.abs() > 1e-12, pos_alpha, torch.full_like(pos_alpha, (i + 1) / self.max_events_per_pixel))
            neg_alpha = torch.where(relative.abs() > 1e-12, neg_alpha, torch.full_like(neg_alpha, (i + 1) / self.max_events_per_pixel))
            if bool(pos_mask.any()):
                idx = pos_mask.nonzero(as_tuple=False)
                chunks.append(torch.stack((
                    batch_grid[pos_mask], self.t_previous[idx[:, 0]] + delta[idx[:, 0]] * pos_alpha[pos_mask],
                    xx[pos_mask], yy[pos_mask], torch.ones(idx.shape[0], dtype=torch.float64, device=self.device)
                ), dim=1))
            if bool(neg_mask.any()):
                idx = neg_mask.nonzero(as_tuple=False)
                chunks.append(torch.stack((
                    batch_grid[neg_mask], self.t_previous[idx[:, 0]] + delta[idx[:, 0]] * neg_alpha[neg_mask],
                    xx[neg_mask], yy[neg_mask], -torch.ones(idx.shape[0], dtype=torch.float64, device=self.device)
                ), dim=1))
        if chunks:
            events = torch.cat(chunks, dim=0)
            # Group by environment first so offsets remain a valid packed
            # representation, then sort timestamps within each environment.
            order = torch.argsort(events[:, 0] * (times.max() + 1.0) + events[:, 1])
            events = events[order]
            counts = torch.bincount(events[:, 0].to(torch.int64), minlength=self.batch_size)
        else:
            events = torch.empty((0, 5), dtype=torch.float64, device=self.device)
            counts = torch.zeros((self.batch_size,), dtype=torch.int64, device=self.device)
        offsets = torch.cat((torch.zeros((1,), dtype=torch.int64, device=self.device), torch.cumsum(counts, dim=0)))
        self.base_log_frame = self.base_log_frame + pos_count.to(torch.float64) * self.pos_thres
        self.base_log_frame = self.base_log_frame - neg_count.to(torch.float64) * self.neg_thres
        self.t_previous = times
        return events, offsets
