#!/usr/bin/env python3
"""
Franka 官方 demo 事件动图生成器（完整过程 + fixed/wrist 双视角）
================================================================

对 7 个 Franka 官方 demo 重新生成事件动图：
  01 franka_cube    — IK 抓取提升 (400 步)
  02 gravity_comp   — 3×Franka 重力补偿 (1000 步)
  03 suction_cup    — 吸盘抓取 (800 步)
  06 single_franka  — 单臂 idle (1000 步)
  07 ik_franka      — IK 圆形轨迹跟踪 (2000 步)
  08 ik_grasp       — IK 抓取 (700 步)
  09 control_robot  — PD/速度/力控制 (1400 步)

每个 demo 输出:
  fixed_events.gif  — 全局视角事件动图
  wrist_events.gif  — 手腕视角事件动图

事件可视化风格（与既有 demos_events 一致）:
  ON 事件 → 红色, OFF 事件 → 蓝色, 背景黑色, 滑动窗口累积

用法:
  cd ~/Eventbased_WAM/code/genesis_event_plugin
  PYTHONPATH=. python scripts/demo_runner.py --demo franka_cube
  PYTHONPATH=. python scripts/demo_runner.py --all
"""

import os
import sys
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as ScipyR
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import genesis as gs
from genesis_event_plugin import GenesisEventPlugin

# ════════════════════════════════════════════════════════
# 全局配置
# ════════════════════════════════════════════════════════
CAM_RES = (320, 240)        # (W, H)
CAM_FOV = 40
PRESET = 'real_v2'
SLIDING_WINDOW = 5000       # 事件累积窗口（最近 N 个事件）
TARGET_GIF_FRAMES = 64      # GIF 帧数（覆盖完整过程）
EVENT_BRIGHT = 30           # 每个事件贡献的亮度（与既有风格一致）

# wrist 相机相对 hand 的位姿偏移（参考 grasp_task.py，加大偏移防穿模）
WRIST_OFF = np.array([0.0, 0.03, -0.08])
WRIST_LOOK = np.array([0.0, -0.03, 0.15])

# fixed 相机位姿（每个 demo 单独配置，见 DEMOS）
FRANKA_MJCF = 'xml/franka_emika_panda/panda.xml'


def qapply(q, v):
    return ScipyR.from_quat(q).apply(v)


def wrist_T(hp, hq):
    """从 hand link 位姿计算 wrist 相机 4×4 变换矩阵。"""
    cp = hp + qapply(hq, WRIST_OFF)
    cl = hp + qapply(hq, WRIST_LOOK)
    fwd = cl - cp
    fwd /= np.linalg.norm(fwd)
    wu = np.array([0.0, 0.0, 1.0])
    ri = np.cross(fwd, wu)
    if np.linalg.norm(ri) < 1e-6:
        ri = np.array([1.0, 0.0, 0.0])
    ri /= np.linalg.norm(ri)
    up = np.cross(ri, fwd)
    T = np.eye(4)
    T[:3, 0] = ri
    T[:3, 1] = up
    T[:3, 2] = -fwd
    T[:3, 3] = cp
    return T


# ════════════════════════════════════════════════════════
# 双相机事件动图记录器
# ════════════════════════════════════════════════════════
class DualEventRecorder:
    def __init__(self, scene, fixed_cam, wrist_cam, hand_link, output_dir):
        self.scene = scene
        self.hand_link = hand_link
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.pf = GenesisEventPlugin(
            output_dir=os.path.join(output_dir, '_fixed'),
            preset=PRESET, interpolation_mode='analytic',
            output_event_frames=False, output_rgb_frames=False,
        )
        self.pw = GenesisEventPlugin(
            output_dir=os.path.join(output_dir, '_wrist'),
            preset=PRESET, interpolation_mode='analytic',
            output_event_frames=False, output_rgb_frames=False,
        )
        # moving_entities 由调用方在 attach 前注入
        self.pf.attach(scene, fixed_cam, [])
        self.pw.attach(scene, wrist_cam, [])

        self.fixed_events = []   # [t, x, y, p]
        self.wrist_events = []
        self.fixed_frames = []
        self.wrist_frames = []
        self._step = 0
        self._frame_interval = 1
        self._last = False

    def start(self, total_steps):
        self.pf.start_episode()
        self.pw.start_episode()
        self._step = 0
        self._frame_interval = max(1, total_steps // TARGET_GIF_FRAMES)

    def update_wrist(self):
        hp = self.hand_link.get_pos().cpu().numpy()
        hq = self.hand_link.get_quat().cpu().numpy()
        self.pw.cam.set_pose(wrist_T(hp, hq))

    def capture(self):
        """推进一帧：更新 wrist 相机、渲染并生成事件、周期性合成 GIF 帧。"""
        self.update_wrist()
        fe = self.pf.capture()
        we = self.pw.capture()
        if len(fe) > 0:
            self.fixed_events.append(fe)
            self._trim(self.fixed_events)
        if len(we) > 0:
            self.wrist_events.append(we)
            self._trim(self.wrist_events)
        self._step += 1
        if self._step % self._frame_interval == 0 or self._last:
            self.fixed_frames.append(self._render_frame(self.fixed_events))
            self.wrist_frames.append(self._render_frame(self.wrist_events))
        self._last = False

    @staticmethod
    def _trim(buf):
        total = sum(len(e) for e in buf)
        while buf and total > SLIDING_WINDOW:
            removed = len(buf[0])
            buf.pop(0)
            total -= removed

    def _render_frame(self, event_buf):
        """把滑动窗口事件渲染成 ON红/OFF蓝 图。"""
        H, W = CAM_RES[1], CAM_RES[0]
        on = np.zeros((H, W), dtype=np.float32)
        off = np.zeros((H, W), dtype=np.float32)
        for ev in event_buf:
            xs = np.clip(ev[:, 1].astype(int), 0, W - 1)
            ys = np.clip(ev[:, 2].astype(int), 0, H - 1)
            pos = ev[:, 3] > 0
            np.add.at(on, (ys[pos], xs[pos]), 1)
            np.add.at(off, (ys[~pos], xs[~pos]), 1)
        img = np.zeros((H, W, 3), dtype=np.uint8)
        img[:, :, 0] = np.clip(on * EVENT_BRIGHT, 0, 255).astype(np.uint8)
        img[:, :, 2] = np.clip(off * EVENT_BRIGHT, 0, 255).astype(np.uint8)
        return img

    def finish(self):
        self._last = True
        self.capture()
        self.pf.end_episode()
        self.pw.end_episode()

    def save_gifs(self):
        import imageio.v2 as iio
        for name, frames in [('fixed_events.gif', self.fixed_frames),
                             ('wrist_events.gif', self.wrist_frames)]:
            if not frames:
                print(f"  skip {name}: no frames")
                continue
            path = os.path.join(self.output_dir, name)
            iio.mimsave(path, frames, duration=0.06, loop=0)
        print(f"  saved: fixed {len(self.fixed_frames)}f, wrist {len(self.wrist_frames)}f -> {self.output_dir}")

    def close(self):
        self.pf.close()
        self.pw.close()


# ════════════════════════════════════════════════════════
# 工具：建场景 + 双相机
# ════════════════════════════════════════════════════════
def build_scene(fixed_pos, fixed_look, dt=0.01, gravity=True):
    gs.init(backend=gs.gpu, logging_level='warning')
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, gravity=(0, 0, -9.81) if gravity else (0, 0, 0)),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    fixed_cam = scene.add_camera(res=CAM_RES, pos=fixed_pos, lookat=fixed_look, fov=CAM_FOV)
    wrist_cam = scene.add_camera(res=CAM_RES, pos=fixed_pos, lookat=fixed_look, fov=CAM_FOV)
    return scene, fixed_cam, wrist_cam


def make_recorder(scene, fc, wc, hand_link, outdir, franka):
    rec = DualEventRecorder(scene, fc, wc, hand_link, outdir)
    # 注入运动实体：franka 所有 links
    ents = list(franka.links[1:])
    rec.pf.moving_entities = ents
    rec.pw.moving_entities = ents
    return rec


# ════════════════════════════════════════════════════════
# Demo 任务实现（忠实复现官方逻辑 + 双相机 + 完整过程）
# ════════════════════════════════════════════════════════

def demo_franka_cube(outdir):
    scene, fc, wc = build_scene((3, -1, 1.5), (0, 0, 0.5))
    franka = scene.add_entity(gs.morphs.MJCF(file=FRANKA_MJCF))
    cube = scene.add_entity(gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.65, 0.0, 0.02)))
    scene.build()

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    franka.set_dofs_kp([100.0, 100.0], fingers_dof)
    franka.set_dofs_kv([10.0, 10.0], fingers_dof)
    qpos = np.array([-1.0124, 1.5559, 1.3662, -1.6878, -1.5799, 1.7757, 1.4602, 0.04, 0.04])
    franka.set_qpos(qpos)
    scene.step()

    hand = franka.get_link("hand")
    rec = make_recorder(scene, fc, wc, hand, outdir, franka)
    total = 400
    rec.start(total)

    qpos = franka.inverse_kinematics(link=hand, pos=np.array([0.65, 0.0, 0.135]), quat=np.array([0, 1, 0, 0]))
    franka.control_dofs_position(qpos[:-2], motors_dof)
    for i in range(100):
        scene.step(); rec.capture()

    for i in range(100):
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_position(np.array([0.0, 0.0]), fingers_dof)
        scene.step(); rec.capture()

    qpos = franka.inverse_kinematics(link=hand, pos=np.array([0.65, 0.0, 0.3]), quat=np.array([0, 1, 0, 0]))
    for i in range(200):
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_position(np.array([0.0, 0.0]), fingers_dof)
        scene.step(); rec.capture()

    rec.finish(); rec.save_gifs(); rec.close()
    gs.destroy()


def demo_gravity_comp(outdir):
    scene, fc, wc = build_scene((3.5, 1.0, 2.5), (0, 1.0, 0.5))
    franka = scene.add_entity(gs.morphs.MJCF(file=FRANKA_MJCF, pos=(0, 0, 0)))
    scene.add_entity(gs.morphs.MJCF(file=FRANKA_MJCF, pos=(0, 1.0, 0)),
                     material=gs.materials.Rigid(gravity_compensation=0.5))
    scene.add_entity(gs.morphs.MJCF(file=FRANKA_MJCF, pos=(0, 2.0, 0)),
                     material=gs.materials.Rigid(gravity_compensation=1.0))
    scene.build()

    hand = franka.get_link("hand")
    rec = make_recorder(scene, fc, wc, hand, outdir, franka)
    total = 1000
    rec.start(total)
    for i in range(total):
        scene.step(); rec.capture()
    rec.finish(); rec.save_gifs(); rec.close()
    gs.destroy()


def demo_suction_cup(outdir):
    scene, fc, wc = build_scene((3, -1, 1.5), (0, 0, 0.5))
    cube = scene.add_entity(gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.65, 0.0, 0.02)),
                            surface=gs.surfaces.Plastic(color=(1, 0, 0)))
    cube_2 = scene.add_entity(gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.4, 0.2, 0.02)),
                              surface=gs.surfaces.Plastic(color=(0, 1, 0)))
    franka = scene.add_entity(gs.morphs.MJCF(file=FRANKA_MJCF), vis_mode="collision")
    scene.build()

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    hand = franka.get_link("hand")
    franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
    franka.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
    franka.set_dofs_force_range(
        np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
        np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]))

    total = 100 + 100 + 50 + 50 + 100 + 400
    rec = make_recorder(scene, fc, wc, hand, outdir, franka)
    rec.start(total)

    qpos = franka.inverse_kinematics(link=hand, pos=np.array([0.65, 0.0, 0.25]), quat=np.array([0, 1, 0, 0]))
    qpos[-2:] = 0.04
    path = franka.plan_path(qpos_goal=qpos, num_waypoints=100)
    for wp in path:
        franka.control_dofs_position(wp)
        franka.control_dofs_force(np.array([0.5, 0.5]), fingers_dof)
        scene.step(); rec.capture()

    for i in range(100):
        scene.step(); rec.capture()

    qpos = franka.inverse_kinematics(link=hand, pos=np.array([0.65, 0.0, 0.130]), quat=np.array([0, 1, 0, 0]))
    franka.control_dofs_position(qpos[:-2], motors_dof)
    for i in range(50):
        scene.step(); rec.capture()

    rigid = scene.sim.rigid_solver
    link_cube = cube.get_link("box_baselink").idx
    link_franka = franka.get_link("hand").idx
    rigid.add_weld_constraint(link_cube, link_franka)

    qpos = franka.inverse_kinematics(link=hand, pos=np.array([0.65, 0.0, 0.28]), quat=np.array([0, 1, 0, 0]))
    franka.control_dofs_position(qpos[:-2], motors_dof)
    for i in range(50):
        scene.step(); rec.capture()

    qpos = franka.inverse_kinematics(link=hand, pos=np.array([0.4, 0.2, 0.18]), quat=np.array([0, 1, 0, 0]))
    franka.control_dofs_position(qpos[:-2], motors_dof)
    for i in range(100):
        scene.step(); rec.capture()

    rigid.delete_weld_constraint(link_cube, link_franka)
    for i in range(400):
        scene.step(); rec.capture()

    rec.finish(); rec.save_gifs(); rec.close()
    gs.destroy()


def demo_single_franka(outdir):
    scene, fc, wc = build_scene((3.5, 0.0, 2.5), (0, 0, 0.5))
    franka = scene.add_entity(gs.morphs.MJCF(file=FRANKA_MJCF))
    scene.build()
    hand = franka.get_link("hand")
    rec = make_recorder(scene, fc, wc, hand, outdir, franka)
    total = 1000
    rec.start(total)
    for i in range(total):
        scene.step(); rec.capture()
    rec.finish(); rec.save_gifs(); rec.close()
    gs.destroy()


def demo_ik_franka(outdir):
    # 官方用 gravity=0 + 无碰撞 + visualizer.update；这里改用 scene.step 以便渲染
    gs.init(backend=gs.gpu, logging_level='warning')
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(enable_joint_limit=False, enable_collision=False, gravity=(0, 0, 0)),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(gs.morphs.MJCF(file=FRANKA_MJCF))
    target_entity = scene.add_entity(gs.morphs.Mesh(file='meshes/axis.obj', scale=0.15),
                                     surface=gs.surfaces.Default(color=(1, 0.5, 0.5, 1)))
    fc = scene.add_camera(res=CAM_RES, pos=(0, -2, 1.5), lookat=(0, 0, 0.5), fov=CAM_FOV)
    wc = scene.add_camera(res=CAM_RES, pos=(0, -2, 1.5), lookat=(0, 0, 0.5), fov=CAM_FOV)
    scene.build()

    hand = robot.get_link("hand")
    rec = make_recorder(scene, fc, wc, hand, outdir, robot)
    target_quat = np.array([0, 1, 0, 0])
    center = np.array([0.4, -0.2, 0.25])
    r = 0.1
    total = 2000
    rec.start(total)
    for i in range(total):
        target_pos = center + np.array([np.cos(i / 360 * np.pi), np.sin(i / 360 * np.pi), 0]) * r
        target_entity.set_qpos(np.concatenate([target_pos, target_quat]))
        q, err = robot.inverse_kinematics(link=hand, pos=target_pos, quat=target_quat,
                                          return_error=True, rot_mask=[False, False, True])
        robot.set_qpos(q)
        scene.step(); rec.capture()
    rec.finish(); rec.save_gifs(); rec.close()
    gs.destroy()


def demo_ik_grasp(outdir):
    scene, fc, wc = build_scene((3, -1, 1.5), (0, 0, 0.5))
    cube = scene.add_entity(gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.65, 0.0, 0.02)))
    franka = scene.add_entity(gs.morphs.MJCF(file=FRANKA_MJCF))
    scene.build()

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
    franka.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
    franka.set_dofs_force_range(
        np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
        np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]))
    hand = franka.get_link("hand")

    total = 200 + 100 + 100 + 100 + 200
    rec = make_recorder(scene, fc, wc, hand, outdir, franka)
    rec.start(total)

    qpos = franka.inverse_kinematics(link=hand, pos=np.array([0.65, 0.0, 0.25]), quat=np.array([0, 1, 0, 0]))
    qpos[-2:] = 0.04
    path = franka.plan_path(qpos_goal=qpos, num_waypoints=200)
    for wp in path:
        franka.control_dofs_position(wp)
        scene.step(); rec.capture()

    for i in range(100):
        scene.step(); rec.capture()

    qpos = franka.inverse_kinematics(link=hand, pos=np.array([0.65, 0.0, 0.130]), quat=np.array([0, 1, 0, 0]))
    franka.control_dofs_position(qpos[:-2], motors_dof)
    for i in range(100):
        scene.step(); rec.capture()

    franka.control_dofs_position(qpos[:-2], motors_dof)
    franka.control_dofs_force(np.array([-0.5, -0.5]), fingers_dof)
    for i in range(100):
        scene.step(); rec.capture()

    qpos = franka.inverse_kinematics(link=hand, pos=np.array([0.65, 0.0, 0.28]), quat=np.array([0, 1, 0, 0]))
    franka.control_dofs_position(qpos[:-2], motors_dof)
    for i in range(200):
        scene.step(); rec.capture()

    rec.finish(); rec.save_gifs(); rec.close()
    gs.destroy()


def demo_control_robot(outdir):
    scene, fc, wc = build_scene((0, -3.5, 2.5), (0, 0, 0.5))
    franka = scene.add_entity(gs.morphs.MJCF(file=FRANKA_MJCF))
    scene.build()

    joints_name = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7",
                   "finger_joint1", "finger_joint2")
    motors_dof_idx = [franka.get_joint(name).dofs_idx_local[0] for name in joints_name]
    franka.set_dofs_kp(kp=np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
                       dofs_idx_local=motors_dof_idx)
    franka.set_dofs_kv(kv=np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
                       dofs_idx_local=motors_dof_idx)
    franka.set_dofs_force_range(
        lower=np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
        upper=np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
        dofs_idx_local=motors_dof_idx)

    hand = franka.get_link("hand")
    total = 150 + 1250
    rec = make_recorder(scene, fc, wc, hand, outdir, franka)
    rec.start(total)

    for i in range(150):
        if i < 50:
            franka.set_dofs_position(np.array([1, 1, 0, 0, 0, 0, 0, 0.04, 0.04]), motors_dof_idx)
        elif i < 100:
            franka.set_dofs_position(np.array([-1, 0.8, 1, -2, 1, 0.5, -0.5, 0.04, 0.04]), motors_dof_idx)
        else:
            franka.set_dofs_position(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]), motors_dof_idx)
        scene.step(); rec.capture()

    for i in range(1250):
        if i == 0:
            franka.control_dofs_position(np.array([1, 1, 0, 0, 0, 0, 0, 0.04, 0.04]), motors_dof_idx)
        elif i == 250:
            franka.control_dofs_position(np.array([-1, 0.8, 1, -2, 1, 0.5, -0.5, 0.04, 0.04]), motors_dof_idx)
        elif i == 500:
            franka.control_dofs_position(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]), motors_dof_idx)
        elif i == 750:
            franka.control_dofs_position(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])[1:], motors_dof_idx[1:])
            franka.control_dofs_velocity(np.array([1.0, 0, 0, 0, 0, 0, 0, 0, 0])[:1], motors_dof_idx[:1])
        elif i == 1000:
            franka.control_dofs_force(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]), motors_dof_idx)
        scene.step(); rec.capture()

    rec.finish(); rec.save_gifs(); rec.close()
    gs.destroy()


# ════════════════════════════════════════════════════════
# 注册表 & main
# ════════════════════════════════════════════════════════
DEMOS = {
    'franka_cube': demo_franka_cube,
    'gravity_comp': demo_gravity_comp,
    'suction_cup': demo_suction_cup,
    'single_franka': demo_single_franka,
    'ik_franka': demo_ik_franka,
    'ik_grasp': demo_ik_grasp,
    'control_robot': demo_control_robot,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--demo', default=None, help='单个 demo 名')
    ap.add_argument('--all', action='store_true', help='运行全部 7 个 demo')
    ap.add_argument('--outdir', default=None, help='输出根目录')
    a = ap.parse_args()

    root = a.outdir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'output', 'franka_demos')
    os.makedirs(root, exist_ok=True)

    names = list(DEMOS.keys()) if a.all else ([a.demo] if a.demo else None)
    if names is None:
        print("请指定 --demo <name> 或 --all")
        print("可用 demo:", ", ".join(DEMOS.keys()))
        sys.exit(1)

    for name in names:
        print(f"\n{'='*60}\n  [{name}]\n{'='*60}")
        outdir = os.path.join(root, name)
        try:
            DEMOS[name](outdir)
        except Exception as e:
            import traceback
            print(f"  ❌ {name} 失败: {e}")
            traceback.print_exc()


if __name__ == '__main__':
    main()
