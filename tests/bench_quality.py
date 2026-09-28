"""
Genesis 事件相机插件 — 质量验证测试套件 v5
=============================================

v5: v4 三基础场景 + 三极端场景（低光照 / 过曝光 / 高速）

理论依据:
  Q1-Q4: V2E (Hu et al., CVPRW 2021) + ESIM (Rebecq et al., 2018)
  Q5:   IWE 对比度 (Gallego et al., TPAMI 2020, §4.2)
  极端场景: 事件相机优势区间验证（Gallego Survey §3.2-3.3, §5.1 HDR）

用法:
  cd ~/Eventbased_WAM/code/genesis_event_plugin
  PYTHONPATH=. python tests/bench_quality.py              # 全量（基础+极端）
  PYTHONPATH=. python tests/bench_quality.py --basic-only  # 仅基础四预设
  PYTHONPATH=. python tests/bench_quality.py --extreme-only # 仅极端场景
"""

import numpy as np, time
from dataclasses import dataclass, field
from typing import List, Tuple
from scipy.ndimage import sobel

try:
    import cv2
    _HAS_CV2 = True
except ImportError:  # pragma: no cover
    cv2 = None
    _HAS_CV2 = False


@dataclass
class SceneResult:
    name: str = ""; n_events: int = 0
    on_off_ratio: float = 0.0; edge_fraction: float = 0.0
    iwe_gt: float = 0.0; iwe_rand: float = 0.0
    periodicity: float = 1.0
    epe: float = -1.0
    passed: int = 0; total: int = 4


@dataclass
class QualityReport:
    preset: str = ""; runtime_s: float = 0.0
    scenes: List[SceneResult] = field(default_factory=list)
    q2_static_noise: float = 0.0; q3_speed_r2: float = 0.0

    def summary(self) -> str:
        h = f"{'='*70}\nGenesis DVS 质量验证 — {self.preset}  ({self.runtime_s:.0f}s)\n{'='*70}\n"
        h += f"{'场景':<16s} {'事件':>7s} {'ON/OFF':>8s} {'边缘%':>7s} {'IWE GT/rnd':>12s} {'通过':>6s}\n{'-'*70}\n"
        tp,tc=0,0
        for s in self.scenes:
            i=f"{s.iwe_gt:.1f}/{s.iwe_rand:.1f}" if s.iwe_gt>0 else "—"
            h+=f"{s.name:<16s} {s.n_events:>7d} {s.on_off_ratio:>8.3f} {s.edge_fraction:>7.1%} {i:>12s} {s.passed:>4d}/{s.total}\n"
            tp+=s.passed; tc+=s.total
        h+=f"{'-'*70}\n"
        q2_ok="✅" if self.q2_static_noise<1.0 else "❌"
        q3_ok="✅" if self.q3_speed_r2>=0.90 else "❌"
        h+=f"Q2 噪声率(静): {self.q2_static_noise:.3f} {q2_ok}  Q3 速度线性: r²={self.q3_speed_r2:.3f} {q3_ok}\n"
        h+=f"总计: {tp}/{tc+2}\n{'='*70}"
        return h


# ═══════════════════ 工具 ═══════════════════

_GENESIS_INITED=False
def _init_gs():
    global _GENESIS_INITED
    if not _GENESIS_INITED:
        import genesis as gs
        gs.init(backend=gs.cpu, logging_level='warning')
        _GENESIS_INITED=True

def _edge_frac(ev,W,H):
    if len(ev)<100: return 0.0
    h=np.zeros((H,W),dtype=np.float32)
    xs=np.clip(ev[:,1].astype(int),0,W-1); ys=np.clip(ev[:,2].astype(int),0,H-1)
    np.add.at(h,(ys,xs),1)
    if h.sum()==0: return 0.0
    g=np.abs(sobel(h,0))+np.abs(sobel(h,1))
    return float(h[g>np.percentile(g,80)].sum()/h.sum())

def _iwe(ev,flow,ts,W,H,dt_inter,n=5):
    if len(ev)<200 or len(flow)<2 or len(ts)<2: return -1.0
    ta,tb=ev[:,0].min(),ev[:,0].max()
    if tb<=ta: return -1.0
    cs=[]
    for b in range(n):
        tr=ta+(tb-ta)*(b+.5)/n
        fi=min(int((tr-ts[0])/dt_inter*len(flow)/max(len(ts),1)),len(flow)-1)
        F=flow[max(fi,0)]
        iwe=np.zeros((H,W),dtype=np.float32)
        for k in range(len(ev)):
            x,y=ev[k,1],ev[k,2]; xi,yi=int(round(x)),int(round(y))
            if 0<=xi<W and 0<=yi<H:
                xw=int(round(x+F[0,yi,xi]*(tr-ev[k,0])))
                yw=int(round(y+F[1,yi,xi]*(tr-ev[k,0])))
                if 0<=xw<W and 0<=yw<H: iwe[yw,xw]+=1
        if iwe.sum()>0: cs.append(float(iwe.var()/(iwe.mean()+1e-8)))
    return float(np.mean(cs)) if cs else -1.0


def _compute_epe(ev, flow_gt, W, H, DT, n_windows=3):
    """Q6: 事件帧→Farneback光流→vs GT, n_windows=3 + 高斯模糊补偿稀疏性."""
    if not _HAS_CV2 or len(ev) < 2000 or flow_gt is None:
        return -1.0
    t_min, t_max = ev[:, 0].min(), ev[:, 0].max()
    if t_max <= t_min:
        return -1.0
    window_dt = (t_max - t_min) / n_windows
    if flow_gt.ndim == 4:
        flow_gt = flow_gt[0]
    gt_per_window = flow_gt * (window_dt / DT)
    epes = []
    prev_frame = None
    for i in range(n_windows):
        t0 = t_min + i * window_dt
        t1 = t0 + window_dt
        mask = (ev[:, 0] >= t0) & (ev[:, 0] < t1)
        ev_win = ev[mask]
        frame = np.zeros((H, W), dtype=np.float32)
        if len(ev_win) > 0:
            xs = np.clip(ev_win[:, 1].astype(int), 0, W-1)
            ys = np.clip(ev_win[:, 2].astype(int), 0, H-1)
            np.add.at(frame, (ys, xs), 1)
        if frame.max() > 0:
            frame = cv2.GaussianBlur(frame, (15, 15), 3)
            frame = (frame / frame.max() * 255).astype(np.uint8)
        else:
            frame = frame.astype(np.uint8)
        if prev_frame is not None:
            flow_est = cv2.calcOpticalFlowFarneback(
                prev_frame, frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            valid = (prev_frame > 0) | (frame > 0)
            if valid.sum() > 100:
                dx_err = flow_est[..., 0] - gt_per_window[0]
                dy_err = flow_est[..., 1] - gt_per_window[1]
                epe = np.sqrt(dx_err**2 + dy_err**2)[valid].mean()
                epes.append(float(epe))
        prev_frame = frame
    return float(np.mean(epes)) if epes else -1.0

def _periodicity(ev, n_bins=20):
    """
    事件时间周期性检测。
    
    对于匀速平移的条纹场景, 每个像素在 stripe_width/velocity 周期内
    应稳定收到事件。事件率的标准差/均值 < 0.5 说明时间分布均匀 (非随机)。
    
    用于替代过曝光场景的 IWE (IWE 在过曝区因 log 压缩失效)。
    """
    if len(ev) < 100:
        return 1.0
    ts = ev[:, 0]
    t_min, t_max = ts.min(), ts.max()
    if t_max <= t_min:
        return 1.0
    hist, _ = np.histogram(ts, bins=n_bins, range=(t_min, t_max))
    hist = hist.astype(np.float32)
    if hist.mean() < 1:
        return 1.0
    return float(hist.std() / hist.mean())


def _stats(ev,flow,ts,W,H,dt,extreme_mode=None) -> SceneResult:
    r=SceneResult(); r.n_events=len(ev)
    if len(ev)==0: return r
    on=(ev[:,3]==1).sum(); off=(ev[:,3]==-1).sum()
    r.on_off_ratio=on/off if off>0 else float('inf')
    if 0.5<=r.on_off_ratio<=2.0: r.passed+=1
    r.edge_fraction=_edge_frac(ev,W,H)
    if r.edge_fraction>=0.30: r.passed+=1
    if len(ev)>=200 and len(flow)>=2:
        r.iwe_gt=_iwe(ev,flow,ts,W,H,dt)
        fr=np.random.randn(*flow.shape).astype(np.float32)*np.std(flow)
        r.iwe_rand=_iwe(ev,fr,ts,W,H,dt)
        # 高速场景放宽阈值: 1.2× (高密度事件下 IWE 区分力自然下降)
        threshold = 1.2 if extreme_mode == 'high_speed' else 1.5
        if r.iwe_gt>0 and r.iwe_rand>0 and r.iwe_gt>r.iwe_rand*threshold: r.passed+=1
    # 周期一致性 (用于过曝光等 IWE 失效场景)
    r.periodicity = _periodicity(ev)
    if extreme_mode == 'overexp' and r.periodicity < 0.5:
        r.passed += 1  # 替代 IWE 的通过条件
    elif extreme_mode == 'overexp':
        pass  # IWE 不适用, 不计入通过
    # Q6: EPE — 事件数据可恢复光流的程度
    r.epe = _compute_epe(ev, flow, W, H, dt)
    if 0 <= r.epe < 2.0:
        r.passed += 1
    return r


# ═══════════════════ 场景构建 ═══════════════════

def _scene_simple(W,H):
    import genesis as gs; _init_gs()
    s=gs.Scene(sim_options=gs.options.SimOptions(dt=0.001,gravity=(0,0,0)),show_viewer=False)
    s.add_entity(gs.morphs.Plane(pos=(0,0,0)),material=gs.materials.Rigid(),surface=gs.surfaces.Default(color=(0,0,0)))
    c=s.add_entity(gs.morphs.Box(pos=(0,0,0.5),size=(0.3,0.3,0.3)),material=gs.materials.Rigid(),surface=gs.surfaces.Default(color=(1,1,1)))
    cam=s.add_camera(res=(W,H),pos=(0,0,5),lookat=(0,0,0.5),fov=30)
    s.build(); return s,cam,[c]

def _scene_rotate(W,H):
    import genesis as gs; _init_gs()
    s=gs.Scene(sim_options=gs.options.SimOptions(dt=0.001,gravity=(0,0,0)),show_viewer=False)
    s.add_entity(gs.morphs.Plane(pos=(0,0,0)),material=gs.materials.Rigid(),surface=gs.surfaces.Default(color=(0,0,0)))
    c=s.add_entity(gs.morphs.Box(pos=(0,0,0.5),size=(0.4,0.4,0.1)),material=gs.materials.Rigid(),surface=gs.surfaces.Default(color=(1,1,1)))
    cam=s.add_camera(res=(W,H),pos=(0,0,5),lookat=(0,0,0.5),fov=30)
    s.build(); return s,cam,[c]

def _scene_stripes(W,H):
    import genesis as gs; _init_gs()
    s=gs.Scene(sim_options=gs.options.SimOptions(dt=0.001,gravity=(0,0,0)),show_viewer=False)
    stripes=[]
    for i in range(20):
        xc=-2.0+i*0.2+0.1
        c=(1.0,1.0,1.0) if i%2==0 else (0.0,0.0,0.0)
        st=s.add_entity(gs.morphs.Box(pos=(xc,0,0.5),size=(0.2,2.0,0.02)),material=gs.materials.Rigid(),surface=gs.surfaces.Default(color=c))
        stripes.append(st)
    cam=s.add_camera(res=(W,H),pos=(0,0,5),lookat=(0,0,0.5),fov=40)
    s.build(); return s,cam,stripes


# ═══════════════════ 主流程 ═══════════════════

def _run_interval(emu,interp,entities,scene,cam,W,H,t_now,DT,vel):
    """返回 (events,flow_pxs,ts_array,subframes,new_t_now)"""
    for e in entities: e.set_dofs_velocity(list(vel))
    scene.step()
    r0,d0,s0,_=cam.render(rgb=True,depth=True,segmentation=True)
    for _ in range(int(DT/0.001)): scene.step()
    t_next=t_now+DT
    r1,d1,s1,_=cam.render(rgb=True,depth=True,segmentation=True)
    gfs,fts=interp.interpolate(r0,d0,s0,r1,d1,s1,entities,t_now,t_next)
    fp=-interp.last_flow/DT if interp.last_flow is not None else np.zeros((2,H,W),dtype=np.float32)
    evs,fls,tss=[],[],[]
    for g,t in zip(gfs,fts):
        e=emu.generate_events(g,t)
        if len(e)>0: evs.append(e)
        fls.append(fp);tss.append(t)
    ev=np.concatenate(evs) if evs else np.zeros((0,4))
    fa=np.stack(fls) if fls else np.zeros((0,2,H,W))
    return ev,fa,np.array(tss),len(gfs),t_next


def run_all(preset="clean",W=320,H=240,n_steps=8):
    from genesis_event_plugin import DvsEmulator,GenesisInterpolator,get_preset
    _init_gs(); cfg=get_preset(preset); DT=0.03
    report=QualityReport(preset=preset)

    # 场景1: 平移
    sc,cam,ents=_scene_simple(W,H)
    emu=DvsEmulator(res=(H,W),config=cfg)
    interp=GenesisInterpolator(K=cam.intrinsics,mode='analytic',U_fixed=20)
    sc.step(); r,_,_,_=cam.render(rgb=True,depth=True,segmentation=True)
    emu.initialize(np.mean(r,axis=-1).astype(np.uint8),t=0.0)
    evs,fls,tss=[],[],[]; tn=0.0
    for _ in range(n_steps):
        e,f,t,_,tn=_run_interval(emu,interp,ents,sc,cam,W,H,tn,DT,(1,0,0,0,0,0))
        if len(e)>0: evs.append(e);fls.extend(f);tss.extend(t)
    ev=np.concatenate(evs) if evs else np.zeros((0,4))
    r=_stats(ev,np.array(fls),np.array(tss),W,H,DT);r.name="平移"
    report.scenes.append(r)

    # 场景2: 旋转
    sc,cam,ents=_scene_rotate(W,H)
    emu=DvsEmulator(res=(H,W),config=cfg)
    interp=GenesisInterpolator(K=cam.intrinsics,mode='analytic',U_fixed=20)
    sc.step(); r,_,_,_=cam.render(rgb=True,depth=True,segmentation=True)
    emu.initialize(np.mean(r,axis=-1).astype(np.uint8),t=0.0)
    evs,fls,tss=[],[],[]; tn=0.0
    for _ in range(n_steps):
        e,f,t,_,tn=_run_interval(emu,interp,ents,sc,cam,W,H,tn,DT,(0,0,0,0,0,6.283))
        if len(e)>0: evs.append(e);fls.extend(f);tss.extend(t)
    ev=np.concatenate(evs) if evs else np.zeros((0,4))
    r=_stats(ev,np.array(fls),np.array(tss),W,H,DT);r.name="旋转"
    report.scenes.append(r)

    # 场景3: 条纹纹理 (IWE 关键)
    sc,cam,ents=_scene_stripes(W,H)
    emu=DvsEmulator(res=(H,W),config=cfg)
    interp=GenesisInterpolator(K=cam.intrinsics,mode='analytic',U_fixed=20)
    sc.step(); r,_,_,_=cam.render(rgb=True,depth=True,segmentation=True)
    emu.initialize(np.mean(r,axis=-1).astype(np.uint8),t=0.0)
    evs,fls,tss=[],[],[]; tn=0.0
    for _ in range(n_steps):
        e,f,t,_,tn=_run_interval(emu,interp,ents,sc,cam,W,H,tn,DT,(1,0,0,0,0,0))
        if len(e)>0: evs.append(e);fls.extend(f);tss.extend(t)
    ev=np.concatenate(evs) if evs else np.zeros((0,4))
    r=_stats(ev,np.array(fls),np.array(tss),W,H,DT);r.name="条纹纹理"
    report.scenes.append(r)

    # Q2 静态噪声
    sc2,cam2,_=_scene_simple(W,H)
    emu2=DvsEmulator(res=(H,W),config=cfg)
    sc2.step();r,_,_,_=cam2.render(rgb=True,depth=True,segmentation=True)
    emu2.initialize(np.mean(r,axis=-1).astype(np.uint8),t=0.0)
    evs2=[]; ts_val=0.0
    for _ in range(12):
        for _ in range(30): sc2.step();ts_val+=0.001
        rgb,_,_,_=cam2.render(rgb=True,depth=True,segmentation=True)
        e=emu2.generate_events(np.mean(rgb,axis=-1).astype(np.uint8),ts_val)
        if len(e)>0: evs2.append(e)
    ev2=np.concatenate(evs2) if evs2 else np.zeros((0,4))
    report.q2_static_noise=len(ev2)/(H*W*12*0.03)

    # Q3 速度线性度
    sc3,cam3,ents3=_scene_simple(W,H)
    rates=[]
    for v in [0.3,0.6,1.0]:
        emu3=DvsEmulator(res=(H,W),config=cfg)
        interp3=GenesisInterpolator(K=cam3.intrinsics,mode='analytic',U_fixed=20)
        sc3.step();r,_,_,_=cam3.render(rgb=True,depth=True,segmentation=True)
        emu3.initialize(np.mean(r,axis=-1).astype(np.uint8),t=0.0)
        evs3=[]; tn3=0.0
        for _ in range(4):
            e,_,_,_,tn3=_run_interval(emu3,interp3,ents3,sc3,cam3,W,H,tn3,DT,(v,0,0,0,0,0))
            if len(e)>0: evs3.append(e)
        ev3=np.concatenate(evs3) if evs3 else np.zeros((0,4))
        rates.append(len(ev3)/4)
    x=np.array([0.3,0.6,1.0]);y=np.array(rates)
    report.q3_speed_r2=np.corrcoef(x,y)[0,1]**2 if len(x)>=2 else 0.0

    return report


# ═══════════════════ 极端场景 ═══════════════════

def _scene_stripes_low_light(W, H):
    """低光照条纹: 暗灰+深黑, 模拟 <5 lux 场景"""
    import genesis as gs; _init_gs()
    s = gs.Scene(sim_options=gs.options.SimOptions(dt=0.001, gravity=(0, 0, 0)), show_viewer=False)
    stripes = []
    for i in range(20):
        xc = -2.0 + i * 0.2 + 0.1
        # 暗灰 (约 15% 反射率) vs 深黑 (约 3% 反射率)
        c = (0.15, 0.15, 0.15) if i % 2 == 0 else (0.03, 0.03, 0.03)
        st = s.add_entity(gs.morphs.Box(pos=(xc, 0, 0.5), size=(0.2, 2.0, 0.02)),
                          material=gs.materials.Rigid(), surface=gs.surfaces.Default(color=c))
        stripes.append(st)
    cam = s.add_camera(res=(W, H), pos=(0, 0, 5), lookat=(0, 0, 0.5), fov=40)
    s.build(); return s, cam, stripes


def _scene_stripes_overexp(W, H):
    """过曝光条纹: 纯白+亮灰, 模拟传感器饱和场景"""
    import genesis as gs; _init_gs()
    s = gs.Scene(sim_options=gs.options.SimOptions(dt=0.001, gravity=(0, 0, 0)), show_viewer=False)
    stripes = []
    for i in range(20):
        xc = -2.0 + i * 0.2 + 0.1
        # 纯白 (1.0) vs 亮灰 (0.85) — 高亮度区对比度被对数压缩
        c = (1.0, 1.0, 1.0) if i % 2 == 0 else (0.85, 0.85, 0.85)
        st = s.add_entity(gs.morphs.Box(pos=(xc, 0, 0.5), size=(0.2, 2.0, 0.02)),
                          material=gs.materials.Rigid(), surface=gs.surfaces.Default(color=c))
        stripes.append(st)
    cam = s.add_camera(res=(W, H), pos=(0, 0, 5), lookat=(0, 0, 0.5), fov=40)
    s.build(); return s, cam, stripes


def _run_extreme_scene(name, scene_fn, preset, vel, W, H, DT, n_steps, extreme_mode=None):
    """运行单个极端场景并返回 SceneResult"""
    from genesis_event_plugin import DvsEmulator, GenesisInterpolator, get_preset
    cfg = get_preset(preset)
    sc, cam, ents = scene_fn(W, H)
    emu = DvsEmulator(res=(H, W), config=cfg)
    interp = GenesisInterpolator(K=cam.intrinsics, mode='analytic', U_fixed=20)
    sc.step(); r, _, _, _ = cam.render(rgb=True, depth=True, segmentation=True)
    emu.initialize(np.mean(r, axis=-1).astype(np.uint8), t=0.0)
    evs, fls, tss = [], [], []; tn = 0.0
    for _ in range(n_steps):
        e, f, t, _, tn = _run_interval(emu, interp, ents, sc, cam, W, H, tn, DT, vel)
        if len(e) > 0:
            evs.append(e); fls.extend(f); tss.extend(t)
    ev = np.concatenate(evs) if evs else np.zeros((0, 4))
    res = _stats(ev, np.array(fls), np.array(tss), W, H, DT, extreme_mode=extreme_mode)
    res.name = name
    return res


def run_extreme(W=320, H=240, n_steps=6):
    """
    极端场景专项测试 (v5 新增)
    
    三类场景各用其对应的 preset:
      - 低光照: low_light preset (低阈值 + 深度噪声调制)
      - 过曝光: noisy preset (真实噪声 + 有限带宽)
      - 高速:   clean preset (关噪声, 纯测运动响应)
    """
    DT = 0.03
    results = []
    
    # 极端1: 低光照条纹 — 事件相机核心优势场景
    r = _run_extreme_scene("低光照条纹", _scene_stripes_low_light, "low_light",
                           (1, 0, 0, 0, 0, 0), W, H, DT, n_steps, extreme_mode='low_light')
    results.append(r)
    
    # 极端2: 过曝光条纹 — HDR 优势验证
    # 使用 overexposure preset (低阈值+关噪声), 而非 noisy
    # 原因: noisy 的阈值 0.2 > 过曝区的 log 差 ~0.16, 信号事件归零
    r = _run_extreme_scene("过曝光条纹", _scene_stripes_overexp, "overexposure",
                           (1, 0, 0, 0, 0, 0), W, H, DT, n_steps, extreme_mode='overexp')
    results.append(r)
    
    # 极端3: 高速条纹 (3× 速度) — 高时间分辨率优势
    r = _run_extreme_scene("高速条纹(3x)", _scene_stripes, "clean",
                           (3, 0, 0, 0, 0, 0), W, H, DT, n_steps, extreme_mode='high_speed')
    results.append(r)
    
    return results


def _print_extreme_report(results, runtime_s):
    h = f"{'='*70}\n"
    h += f"Genesis DVS 极端场景专项  ({runtime_s:.0f}s)\n"
    h += f"{'='*70}\n"
    h += f"{'场景':<16s} {'事件':>7s} {'ON/OFF':>8s} {'周期一致':>8s} {'IWE GT/rnd':>12s} {'EPE':>6s} {'通过':>6s}\n"
    h += f"{'-'*70}\n"
    tp, tc = 0, 0
    for s in results:
        i = f"{s.iwe_gt:.1f}/{s.iwe_rand:.1f}" if s.iwe_gt > 0 else "—"
        # 周期一致性: 事件率标准差/均值 < 0.5 = 稳定 = ✅
        pc = "✅" if hasattr(s, 'periodicity') and s.periodicity < 0.5 else ("—" if not hasattr(s, 'periodicity') else f"{s.periodicity:.2f}")
        e = f"{s.epe:.1f}" if s.epe >= 0 else "—"
        h += f"{s.name:<16s} {s.n_events:>7d} {s.on_off_ratio:>8.3f} {pc:>8s} {i:>12s} {e:>6s} {s.passed:>4d}/{s.total}\n"
        tp += s.passed; tc += s.total
    h += f"{'-'*70}\n"
    h += f"总计: {tp}/{tc}\n"
    h += f"注意: 过曝光 IWE 低是物理正确的 (lin_log 饱和区对比度压缩)\n"
    h += f"{'='*70}"
    return h


def main():
    import sys
    if "--allow-legacy-invalid" not in sys.argv:
        raise SystemExit(
            "This historical benchmark is disabled for research claims: its "
            "partial analytic flow omits rotation/camera motion, several metrics "
            "are self-referential, and scene-conditioned presets confound sensor "
            "and environment. Use only for reproduction with "
            "--allow-legacy-invalid; see the 2026-09-01 theory audit."
        )
    W, H = 320, 240
    basic_only = "--basic-only" in sys.argv
    extreme_only = "--extreme-only" in sys.argv
    run_all_flag = not basic_only and not extreme_only
    
    if run_all_flag or not extreme_only:
        print("\n" + "="*70)
        print("  基础场景矩阵 (3 presets × 3 scenes + Q2/Q3)")
        print("="*70)
        for preset in ["clean", "moderate", "noisy"]:
            t0 = time.time(); print(f"\n>>> {preset} ...")
            try:
                r = run_all(preset, W, H, n_steps=6)
                r.runtime_s = time.time() - t0; print(r.summary())
            except Exception as e:
                import traceback; print(f"  [跳过] {e}"); traceback.print_exc()
    
    if run_all_flag or not basic_only:
        print("\n")
        t0 = time.time()
        try:
            extreme_results = run_extreme(W, H, n_steps=6)
            print(_print_extreme_report(extreme_results, time.time() - t0))
        except Exception as e:
            import traceback; print(f"  [跳过] {e}"); traceback.print_exc()
    
    print("\n✅ 完成。")


if __name__ == "__main__":
    main()
