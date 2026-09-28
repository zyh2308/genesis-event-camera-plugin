"""
V2E 事件仿真器核心工具函数集（纯函数，无状态），共 7 个。

这些函数构成了 DVS 像素模型的数学核心：
  1. rescale_intensity_frame()  → 强度缩放 [0-255] → [~0.07, ~1.0]，防零值
  2. lin_log()                  → 线性→对数映射（模拟光感受器响应）
  3. low_pass_filter()          → 强度依赖一阶 IIR 低通（模拟光电带宽）
  4. compute_event_map()        → 差分量化：变化量÷阈值 = ON/OFF 事件数
  5. subtract_leak_current()    → 泄漏电流：参考值缓慢漂回基线
  6. compute_photoreceptor_noise_voltage() → 噪声校准：速率→等效电压 RMS
  7. generate_shot_noise()      → 散粒噪声：泊松随机误触发

管线调用顺序：rescale → lin_log → low_pass → [diff] → event_map → leak → shot_noise

Author: Yuhuang Hu, Tobi Delbruck
Email : yuhuang.hu@ini.uzh.ch, tobi@ini.uzh.ch
"""
import logging
import math
import sys

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)



def lin_log(x, threshold=20):
    """
    【线性→对数映射】模拟光感受器的非线性响应特性。

    真实光感受器：
      - 低光照 → 线性响应（防止 log(0)→-∞ 的发散问题）
      - 高光照 → 对数响应（压缩动态范围，人眼/芯片都这样做）

    分段实现：
      - x ≤ 20: y = x × (ln(20)/20)  ← 线性段，保证原点连续
      - x > 20:  y = ln(x)            ← 对数段，压缩亮区

    最后做浮点舍入 (round × 1e8 / 1e8)，防止加减阈值时浮点精度
    丢失导致 OFF 事件异常——这是从硬件实现中学到的教训。

    :param x: 输入线性强度值，范围 0-255（假设 8-bit 输入）
    :param threshold: 线性→对数转换阈值，默认 20
    :return: 对数域强度值，float32
    """
    # 强制转为 float64 以保证舍入精度
    if x.dtype is not torch.float64:  # note float64 to get rounding to work
        x = x.double()

    # 线性段斜率 = ln(threshold) / threshold，保证分段函数在 threshold 处连续
    f = (1./threshold) * math.log(threshold)

    y = torch.where(x <= threshold, x*f, torch.log(x))

    # 浮点舍入：防止加/减阈值时丢失最低位，确保 ON 之后能观察到正确的 OFF
    # 例如：如果一次加法把某些 bit 推到了"不复存在的精度范围"，
    # 后续减法就无法回到原始值，导致 OFF 事件异常
    rounding = 1e8
    y = torch.round(y*rounding)/rounding

    return y.float()


def rescale_intensity_frame(new_frame):
    """
    【强度缩放】将 8-bit 灰度值 [0-255] 映射到 [~0.07, ~1.0]。

    为什么要缩放？
      1. 避免零值：零值在后续 lin_log（对数）和除法中会炸
      2. 归一化到 [0,1] 区间，使低通滤波器的时间常数计算有意义

    +20 保证最小值 ~0.073（避免零），/275 把 255+20=275 映射到 1.0。
    如果 Genesis 亮度是物理单位（辐射度），这里的常数值需要重调。
    """
    return (new_frame+20)/275.


def low_pass_filter(
        log_new_frame,
        lp_log_frame,
        inten01,
        delta_time,
        cutoff_hz=0):
    """
    【强度依赖一阶 IIR 低通滤波器】模拟 DVS 光感受器的有限带宽。

    核心物理：真实光电转换不是瞬间完成的——光子→电子→电压需要时间。
    亮度越高，光电效应越快（时间常数越短），亮度越低越慢。

    实现为一阶 IIR（指数移动平均）：
      tau = 1 / (2π × cutoff_hz)     ← 滤波器时间常数（秒）
      eps = inten01 × (dt / tau)     ← 更新系数：越亮 eps 越大，响应越快
      eps = clamp(eps, max=1)        ← 防止数值不稳定
      new = (1-eps) × old + eps × new ← 一阶 IIR 递推

    cutoff_hz ≤ 0 时跳过滤波，直接返回输入（无限带宽模式）。

    注释提到最初是二阶 IIR，但实验发现一阶极点几乎总是主导的，
    所以第二级退化成了恒等映射。

    :param log_new_frame: 当前帧的对数域强度 (H,W)
    :param lp_log_frame: 上一帧的低通输出 (H,W)
    :param inten01: 像素强度缩放因子 (H,W)，用于强度依赖的带宽调制
    :param delta_time: 两帧之间的时间间隔（秒）
    :param cutoff_hz: 低通截止频率（Hz），≤0 表示不过滤
    :return: 滤波后的对数强度 (H,W)
    """
    if cutoff_hz <= 0:
        # 不过滤，直接透传
        return log_new_frame

    # 计算时间常数
    tau = 1/(math.pi*2*cutoff_hz)

    # 强度依赖的更新系数：inten01 越大（像素越亮）→ eps 越大 → 响应越快
    if inten01 is not None:
        eps = inten01*(delta_time/tau)
        max_eps = torch.max(eps)
        if max_eps >0.3:
            IIR_MAX_WARNINGS = 10
            if low_pass_filter.iir_warning_count<IIR_MAX_WARNINGS:
                logger.warning(f'IIR lowpass filter update has large maximum update eps={max_eps:.2f} from delta_time/tau={delta_time:.3g}/{tau:.3g}')
                low_pass_filter.iir_warning_count+=1
                if low_pass_filter.iir_warning_count==IIR_MAX_WARNINGS:
                    logger.warning(f'Supressing further warnings about inaccurate IIR lowpass filtering; check timestamp resolution and DVS photoreceptor cutoff frequency')

        eps = torch.clamp(eps, max=1)  # 防止 eps>1 导致滤波器不稳定
    else:
        eps=delta_time/tau

    # 一阶 IIR 递推：指数移动平均
    new_lp_log_frame = (1-eps)*lp_log_frame+eps*log_new_frame

    # 二阶被注释掉了，因为实验发现一阶极点占主导
    # (1-eps)*self.lpLogFrame1+eps*self.lpLogFrame0 原本是二阶的第二级

    return new_lp_log_frame

low_pass_filter.iir_warning_count=0


def subtract_leak_current(base_log_frame,
                          leak_rate_hz,
                          delta_time,
                          pos_thres,
                          leak_jitter_fraction,
                          noise_rate_array,
                          generator=None):
    """
    【泄漏电流】模拟 DVS 像素的参考电压缓慢漂回基线。

    物理背景：真实 DVS 像素中，存储参考电压的电容器会缓慢放电。
    这意味着即使场景不变，参考值也会逐渐下降，当它与当前值的差
    超过阈值时就会触发 OFF 事件——这就是 DVS 的"自发电活动"。

    公式：delta_leak = dt × leak_rate_hz × noise_array × (1 - jitter×rand) × pos_thres

    noise_rate_array 引入像素间 mismatch（制造工艺差异导致不同像素
    泄漏速率不同），leak_jitter_fraction 加入时间方向上的随机抖动。

    :param base_log_frame: 参考对数强度 (H,W)
    :param leak_rate_hz: 泄漏速率（Hz），对应每秒泄漏多少个阈值单位
    :param delta_time: 时间步长（秒）
    :param pos_thres: ON 阈值，用于换算泄漏量
    :param leak_jitter_fraction: 泄漏抖动的相对幅度 [0,1]
    :param noise_rate_array: 像素级噪声率数组 (H,W)，引入 mismatch
    :return: 减去泄漏后的参考对数强度 (H,W)
    """

    rand = torch.randn(
        noise_rate_array.shape, dtype=torch.float32,
        device=noise_rate_array.device, generator=generator)

    # 每个像素的泄漏速率 = 基础速率 × 像素噪声因子 × 随机抖动
    curr_leak_rate = \
        leak_rate_hz*noise_rate_array*(1-leak_jitter_fraction*rand)

    # 泄漏量 = 时间 × 速率 × 阈值（单位：对数域单位）
    delta_leak = delta_time*curr_leak_rate*pos_thres  # this is a matrix

    # 理想模型（无噪声、无 mismatch）：
    #  delta_leak = delta_time*leak_rate_hz*pos_thres  # this is a matrix

    return base_log_frame-delta_leak


def compute_event_map(diff_frame, pos_thres, neg_thres):
    """
    【事件检测与量化】这是 DVS 事件生成的最核心函数。

    输入：diff_frame = 当前对数强度 - 参考对数强度（即亮度的对数变化量）

    算法：
      1. ReLU(diff) → 正变化部分 → ÷ pos_thres → ON 事件数
      2. ReLU(-diff) → 负变化部分 → ÷ neg_thres → OFF 事件数
      3. floor 量化：如果变化量是阈值的 3.2 倍 → 3 个事件

    含义：DVS 不是"变化超过阈值就发 1 个事件"，
          而是"变化超过阈值 N 倍就发 N 个事件"。
          这确保了高亮度变化区域的事件密度更高，
          保留了场景纹理的灰度分辨率。

    返回的是每个像素的整数事件计数图（后续会展开为 [t,x,y,p] 稀疏表）。

    注释掉的旧代码是直接展开为坐标数组的版本，现在改为返回计数图，
    由上层按需展开——更高效。

    :param diff_frame: 差分对数强度 (H,W)
    :param pos_thres: ON 阈值数组 (H,W)，支持像素级 mismatch
    :param neg_thres: OFF 阈值数组 (H,W)，支持像素级 mismatch
    :return: (pos_evts_frame, neg_evts_frame) 各为 (H,W) int32 事件计数
    """
    # 分离正/负变化
    pos_frame = F.relu(diff_frame)
    neg_frame = F.relu(-diff_frame)

    # 量化：变化量 ÷ 阈值 → 事件数（floor 取整，不足一个阈值的被丢弃）
    pos_evts_frame = torch.div(
        pos_frame, pos_thres, rounding_mode="floor").type(torch.int32)
    neg_evts_frame = torch.div(
        neg_frame, neg_thres, rounding_mode="floor").type(torch.int32)

    # 注释掉的旧代码：直接展开为坐标数组
    #  pos_evts_cord = torch.arange(1, max_events+1, ...) ...

    return pos_evts_frame, neg_evts_frame


def compute_photoreceptor_noise_voltage(shot_noise_rate_hz, f3db, sample_rate_hz, pos_thr, neg_thr, sigma_thr) -> float:
    """
    【光感受器噪声电压校准】将指定的散粒噪声速率换算为等效的高斯噪声 RMS。

    物理背景：DVS 像素的散粒噪声（shot noise）本来是泊松过程，
    但注入高斯噪声到光感受器、再经 RC 低通滤波后也能产生等效的
    噪声事件率。这个函数做的就是"反向求解"：
      给定目标噪声事件率 → 求解需要注入多大的高斯噪声。

    算法步骤：
      1. 用 Graca & Delbruck (2021) Fig.3 曲线拟合公式
         从 Rn/f3db 反推 thr/Vn
      2. 对 ON/OFF 阈值做 Monte Carlo 采样 (N=300)，
         取 min(thr_on, thr_off) 作为有效阈值
      3. 对每个采样求 Vn，取均值
      4. 仿真一阶 IIR 低通滤波，求 white→filtered 的 RMS 衰减比
         放大 Vn 以补偿低通衰减

    这是"一次性"的——初始化时算一次，后续每帧复用缓存值。
    如果 sample_rate 变化 <10%，直接返回缓存。

    参考：Graca & Delbruck, "Unraveling the Paradox of
    Intensity-Dependent DVS Pixel Noise", arXiv 2021.
    数据拟合见 media/noise_event_rate_simulation.xlsx

    Parameters
    -----------
     shot_noise_rate_hz: float 目标散粒噪声速率 (Hz)
     f3db: float 光感受器 -3dB 带宽 (Hz)
     sample_rate_hz: float 上采样帧率 (Hz)
     pos_thr: float ON 阈值 (ln 单位)
     neg_thr: float OFF 阈值 (ln 单位)
     sigma_thr: float 阈值 mismatch 标准差

    Returns
    -----------
    float 注入的高斯噪声 RMS 值 (对数域单位)
    """

    def compute_vn_from_log_rate_per_hz(thr, x):
        # y = log10(thr/Vn)
        # x = log10(Rn/f3db)
        # see the plot Fig. 3 from Graca, Rui, and Tobi Delbruck. 2021. “Unraveling the Paradox of Intensity-Dependent DVS Pixel Noise.” arXiv [eess.SY]. arXiv. http://arxiv.org/abs/2109.08640.
        # the fit is computed in media/noise_event_rate_simulation.xlsx spreadsheet
        y = -0.0026 * x ** 3 - 0.036 * x ** 2 - 0.1949 * x + 0.321
        thr_per_vn = 10 ** y  # to get thr/vn
        vn = thr / thr_per_vn  # compute necessary vn to give us this noise rate per pixel at this pixel bandwidth
        return vn

    # Cache every physical input. The legacy sample-rate-only cache could
    # silently return a voltage calibrated for another threshold/noise setup.
    cache_key = tuple(float(v) for v in (
        shot_noise_rate_hz, f3db, sample_rate_hz,
        pos_thr, neg_thr, sigma_thr,
    ))
    cached = compute_photoreceptor_noise_voltage.cache.get(cache_key)
    if cached is not None:
        return cached

    rate_per_bw= (shot_noise_rate_hz / f3db) / 2  # 除以2：仿真数据用的是ON事件率，这里要总速率
    if rate_per_bw>0.5:
        logger.warning(f'shot noise rate per hz of bandwidth is larger than 0.1 (rate_hz={shot_noise_rate_hz} Hz, 3dB bandwidth={f3db} Hz)')
    x=math.log10(rate_per_bw)
    if x<-5.0:
        logger.warning(f'desired noise rate of {shot_noise_rate_hz}Hz is too low to accurately compute a threshold value')
    elif x>0.0:
        logger.warning(f'desired noise rate of {shot_noise_rate_hz}Hz is too large to accurately compute a threshold value')

    # Monte Carlo 数值估计：噪声率对阈值极度敏感（指数关系），
    # 需要对阈值分布采样来获得准确的 Vn
    N=300  # Monte Carlo 采样数
    calibration_rng = np.random.default_rng(0)
    pos_samps=pos_thr+sigma_thr*calibration_rng.standard_normal(N)
    neg_samps=neg_thr+sigma_thr*calibration_rng.standard_normal(N)
    thrs=np.vstack((pos_samps,neg_samps))
    mins=np.min(thrs,axis=0)  # 取 min(ON阈值, OFF阈值)：哪个小哪个先触发
    vns=np.zeros_like(mins)
    for i in range(N):
        thr=mins[i]

        vn = compute_vn_from_log_rate_per_hz(thr, x)
        vns[i]=vn

    vn=np.mean(vns)
    # RC低通仿真：白噪声经过 IIR 后 RMS 衰减，需要反推缩放因子
    # # to get this NEB factor, we generate white samples here, lowpass filter them the same exact way
    # as we do in the emulator (i.e. with same IIR time constant and sample rate)
    # compute the variance, and scale the amplitude to give us vn
    tau=1/(f3db*2*math.pi)
    dt=1/sample_rate_hz
    t=np.arange(0,1000*tau,dt)
    rin = vn*calibration_rng.standard_normal(t.shape)  # 生成 RMS=vn 的白噪声序列
    rms_in=np.std(rin)  # 验证输入 RMS 应为 vn
    rout=np.zeros_like(rin)
    # RC lowpass the noise
    eps=dt/tau
    eps_limit=.1
    if eps>eps_limit:
        logger.warning(f'\neps={eps:.3f} for IIR lowpass is >{eps_limit}, either reduce timestep (currently {dt:.3f}s) (using higher frame rate) or decrease cutuff_hz (currently {f3db:.3f} Hz)'
                       f'\n\tExpect the generated shot noise rate to be significantly lower than the desired rate.'
                       f'\n\tConsider not using --photoreceptor_noise option if you only want simple Poisson shot noise without temporal correlation of lowpass filtering and ON/OFF events.')
    rout[0]=0  # 初始值为均值 0
    # 用与 emulator 完全相同的 IIR 低通滤波
    for i in range(1,len(rin)):
        rout[i]=rout[i-1]*(1-eps)+rin[i]*eps
    rms_out=np.std(rout)  # 输出 RMS (经过低通后衰减)
    scale=rms_in/rms_out  # 衰减比 = 输入RMS/输出RMS
    vnscaled=scale*vn  # 放大 Vn 以补偿低通衰减
    new_rms_out=np.std(scale*rin)  # 验证缩放后的 RMS

    compute_photoreceptor_noise_voltage.cache[cache_key]=vnscaled
    # rout*=vnscaled
    # stdout=np.std(rout)
    # import matplotlib.pyplot as plt
    # plt.plot(t,rin,t,rout)
    # plt.xlabel('time (s)')
    # plt.ylabel('filtered noise')
    # plt.show()
    if not compute_photoreceptor_noise_voltage.vrms_computation_printed:
        logger.info(
        f'For desired shot_noise_rate_hz={shot_noise_rate_hz} Hz, computed photoreceptor_noise_rms={vn:.3f} in ln units,'
        f' scaled by factor {scale:.3f} to {vnscaled:.3f} before 1st-order lowpass with sample rate {sample_rate_hz:.3} Hz, '
        f'sample interval dt={dt*1000:.3f} ms,'
        f', cutoff_hz={f3db} Hz, tau={tau*1000:.3f} ms,  Rn/f3dB={rate_per_bw:.3g} Hz, '
        f' and nominal on/off threshold={pos_thr}/{neg_thr} +/- {sigma_thr:.3f} ln units.'
        # f' The sample lowpass filtered has RMS amplitude {stdout:.3f}.'
        )
        compute_photoreceptor_noise_voltage.vrms_computation_printed=True
    return vnscaled

compute_photoreceptor_noise_voltage.vrms_computation_printed=False
compute_photoreceptor_noise_voltage.cache={}

def generate_shot_noise(
        shot_noise_rate_hz,
        delta_time,
        shot_noise_inten_factor,
        inten01,
        pos_thres_pre_prob,
        neg_thres_pre_prob,
        generator=None):
    """
    【散粒噪声生成】模拟光子到达的泊松随机性产生的误触发。

    物理背景：光的量子性意味着每个像素的光子到达是泊松过程，
    即使在完全均匀的光照下也有涨落。当涨落超过阈值就产生噪声事件。

    算法：
      1. 计算本帧的噪声概率 shot_factor = (rate/2) × dt × intensity_factor
      2. 每个像素生成 [0,1] 均匀随机数
      3. ON 噪声：rand > (1 - shot_factor × pos_thres_ratio)
      4. OFF 噪声：rand < shot_factor × neg_thres_ratio

    关键设计：
      - 阈值越低的像素噪声率越高 (thres_ratio = nominal/actual)
      - intensity_factor 引入微弱的强度依赖
      - 返回布尔掩码 (H,W)，由上层 emulator 展开为事件坐标

    :param shot_noise_rate_hz: 每像素噪声率 (Hz)
    :param delta_time: 本帧时间步长 (秒)
    :param shot_noise_inten_factor: factor to model the slight increase
        of shot noise with intensity when shot noise dominates at low intensity
    :param inten01: the pixel light intensities in this frame; shape is used to generate output
    :param pos_thres_pre_prob: per pixel factor to generate more
        noise from pixels with lower ON threshold: self.pos_thres_nominal/self.pos_thres
    :param neg_thres_pre_prob: same for OFF

    :returns: shot_on_coord, shot_off_coord, each are (h,w) arrays of on and off boolean True for noise events per pixel
    """
    # new shot noise generator, generate for the entire batch of iterations over this frame

    # 安全检查：噪声概率不应 >1
    max_rate_dt = float(torch.as_tensor(shot_noise_rate_hz * delta_time).max())
    if max_rate_dt > 1:
        logger.warning(
            "max(shot_noise_rate_hz*delta_time)=%.3f is too large; "
            "decrease timestamp interval or noise rate",
            max_rate_dt,
        )

    # 散粒噪声因子 = 本帧产生 OFF 噪声事件的概率
    # = (总速率/2) × dt × 强度因子  （÷2 是因为 ON/OFF 各一半）
    shot_noise_factor = (
        (shot_noise_rate_hz/2)*delta_time) * \
        ((shot_noise_inten_factor-1)*inten01+1)  # inten=0→=1, inten=1→=SHOT_NOISE_INTEN_FACTOR

    # 像素级概率：dt × rate × (nom_thres/actual_thres)
    # 阈值越小 → 噪声率越高（mismatch 效应）
    one_minus_shot_ON_prob_this_sample = \
        1 - shot_noise_factor*pos_thres_pre_prob  # ON噪声不发生的概率
    shot_OFF_prob_this_sample = \
        shot_noise_factor*neg_thres_pre_prob  # OFF噪声发生的概率

    # 每个像素生成 [0,1] 均匀随机数
    rand01 = torch.rand(
        size=inten01.shape,
        dtype=torch.float32,
        device=inten01.device,
        generator=generator)  # 所有像素共享同一帧的随机数

    # 生成布尔掩码：True = 该像素产生噪声事件
    shot_on_cord = torch.gt(
        rand01, one_minus_shot_ON_prob_this_sample)
    shot_off_cord = torch.lt(
        rand01, shot_OFF_prob_this_sample)

    # 返回 ON/OFF 噪声事件的布尔掩码，各为 (H,W)
    # 由上层 emulator 将 True 位置展开为 [t,x,y,p] 事件
    return shot_on_cord, shot_off_cord

    # === 旧版散粒噪声（已废弃）=== 直接在每帧生成事件坐标，效率低
    # the right device
    #  device = base_log_frame.device

    # array with True where ON noise event
    #  shot_ON_cord = rand01 > (1-shot_ON_prob_this_sample)
    #
    #  shot_OFF_cord = rand01 < shot_OFF_prob_this_sample

    # get shot noise event ON and OFF cordinates
    #  shot_ON_xy = shot_ON_cord.nonzero(as_tuple=True)
    #  shot_ON_count = shot_ON_xy[0].shape[0]
    #
    #  shot_OFF_xy = shot_OFF_cord.nonzero(as_tuple=True)
    #  shot_OFF_count = shot_OFF_xy[0].shape[0]

    #  self.num_events_on += shotOnCount
    #  self.num_events_off += shotOffCount
    #  self.num_events_total += shotOnCount+shotOffCount

    # update log_frame
    #  base_log_frame += shot_ON_cord*pos_thres
    #  base_log_frame -= shot_OFF_cord*neg_thres

    #  if shot_ON_count > 0:
    #      shot_ON_events = torch.ones(
    #          (shot_ON_count, 4), dtype=torch.float32, device=device)
    #      shot_ON_events[:, 0] *= ts
    #      shot_ON_events[:, 1] = shot_ON_xy[1]
    #      shot_ON_events[:, 2] = shot_ON_xy[0]
    #
    #      base_log_frame += shot_ON_cord*pos_thres
    #  else:
    #      shot_ON_events = torch.zeros(
    #          (0, 4), dtype=torch.float32, device=device)
    #
    #  if shot_OFF_count > 0:
    #      shot_OFF_events = torch.ones(
    #          (shot_OFF_count, 4), dtype=torch.float32, device=device)
    #      shot_OFF_events[:, 0] *= ts
    #      shot_OFF_events[:, 1] = shot_OFF_xy[1]
    #      shot_OFF_events[:, 2] = shot_OFF_xy[0]
    #      shot_OFF_events[:, 3] *= -1
    #
    #      base_log_frame -= shot_OFF_cord*neg_thres
    #  else:
    #      shot_OFF_events = torch.zeros(
    #          (0, 4), dtype=torch.float32, device=device)
    # end temporal noise

    #  return shot_ON_events, shot_OFF_events, base_log_frame
    #  return shot_ON_cord, shot_OFF_cord, base_log_frame
    #  return shot_ON_cord, shot_OFF_cord


if __name__ == "__main__":

    temp_input = torch.randint(0, 256, (1280, 720), dtype=torch.float32).cuda()

    for i in range(1000):
        temp_out = lin_log(temp_input, threshold=20)

    pass
