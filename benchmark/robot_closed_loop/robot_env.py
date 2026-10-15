"""Genesis Panda task scene. This package is downstream of the frozen core."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np


def as_numpy(value: Any, dtype=np.float64) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def pose_position(body: Any) -> np.ndarray:
    return as_numpy(body.get_pos(relative=False)).reshape(-1)[:3]


def object_state(target: Any, t_s: float) -> Dict[str, Any]:
    dof_v = as_numpy(target.get_dofs_velocity()).reshape(-1)
    linear = np.zeros(3, dtype=np.float64)
    angular = np.zeros(3, dtype=np.float64)
    linear[: min(3, len(dof_v))] = dof_v[:3]
    if len(dof_v) >= 6:
        angular[:] = dof_v[3:6]
    return {
        "time_s": float(t_s),
        "position_m": pose_position(target),
        "velocity_m_s": linear,
        "angular_velocity_rad_s": angular,
    }


def quaternion_matrix_wxyz(quaternion: Sequence[float]) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64).reshape(4)
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


@dataclass
class RobotScene:
    gs: Any
    scene: Any
    plane: Any
    target: Any
    robot: Any
    camera: Any
    hand: Any
    left_finger: Any
    right_finger: Any
    home_qpos: np.ndarray
    hand_quat_wxyz: np.ndarray
    arm_dofs: np.ndarray
    finger_dofs: np.ndarray
    dt_s: float
    config: Dict[str, Any]

    def reset_episode(self, position_m: Sequence[float], heading_rad: float = 0.0) -> None:
        """Episode-boundary reset; no target state is overwritten during motion."""
        self.scene.reset()
        self.robot.set_qpos(self.home_qpos, zero_velocity=True)
        x, y, z = (float(v) for v in position_m)
        quat = np.asarray([np.cos(heading_rad / 2), 0.0, 0.0, np.sin(heading_rad / 2)])
        self.target.set_qpos(np.concatenate(([x, y, z], quat)), zero_velocity=True)
        self.target.set_dofs_velocity(np.zeros(6, dtype=np.float32))

    def set_initial_target_velocity(self, velocity_m_s: Sequence[float]) -> None:
        v = np.asarray(velocity_m_s, dtype=np.float32).reshape(3)
        self.target.set_dofs_velocity(np.concatenate((v, np.zeros(3, dtype=np.float32))))

    def hold_robot_home(self) -> None:
        self.robot.control_dofs_position(self.home_qpos[:7], self.arm_dofs)
        self.robot.control_dofs_position(self.home_qpos[7:9], self.finger_dofs)

    def hand_position(self) -> np.ndarray:
        return pose_position(self.hand)

    def tcp_position(self) -> np.ndarray:
        pos = self.hand_position()
        quat = as_numpy(self.hand.get_quat(relative=False)).reshape(-1)[:4]
        offset = np.asarray([0.0, 0.0, float(self.config["robot"]["hand_link_z_offset_to_pad_center_m"])])
        return pos + quaternion_matrix_wxyz(quat) @ offset

    def jaw_opening_m(self) -> float:
        left = pose_position(self.left_finger)
        right = pose_position(self.right_finger)
        return float(abs(left[1] - right[1]))

    def finger_joint_positions(self) -> np.ndarray:
        return as_numpy(self.robot.get_dofs_position()).reshape(-1)[7:9]

    def finger_joint_velocities(self) -> np.ndarray:
        return as_numpy(self.robot.get_dofs_velocity()).reshape(-1)[7:9]

    def command_hand(self, xyz_m: Sequence[float]) -> np.ndarray:
        current_q = as_numpy(self.robot.get_dofs_position()).reshape(-1)
        q_goal = self.robot.inverse_kinematics(
            link=self.hand,
            pos=np.asarray(xyz_m, dtype=np.float32),
            quat=self.hand_quat_wxyz.astype(np.float32),
            init_qpos=current_q.astype(np.float32),
            max_solver_iters=100,
        )
        q_goal = as_numpy(q_goal).reshape(-1)
        self.robot.control_dofs_position(q_goal[:7], self.arm_dofs)
        return q_goal

    def command_fingers_open(self) -> None:
        opening_each = float(self.config["robot"]["initial_finger_joint_position_m_each"])
        self.robot.control_dofs_position(np.full(2, opening_each, dtype=np.float32), self.finger_dofs)

    def command_fingers_close(self) -> None:
        force = float(self.config["robot"]["finger_close_joint_force_command_each"])
        self.robot.control_dofs_force(np.full(2, force, dtype=np.float32), self.finger_dofs)

    def contacts_with_target(self) -> Dict[str, Any]:
        raw = self.target.get_contacts(with_entity=self.robot)
        out = {}
        for key, value in raw.items():
            out[key] = as_numpy(value)
        links_a = out.get("link_a", np.empty((0,), dtype=np.int64)).reshape(-1)
        links_b = out.get("link_b", np.empty((0,), dtype=np.int64)).reshape(-1)
        robot_links = np.concatenate((links_a, links_b)).astype(np.int64)
        left_id, right_id = int(self.left_finger.idx), int(self.right_finger.idx)
        names = set(int(v) for v in robot_links)
        return {
            "raw": out,
            "left": left_id in names,
            "right": right_id in names,
            "bilateral": left_id in names and right_id in names,
            "robot_link_ids": sorted(names),
        }

    def support_contact(self) -> bool:
        contacts = self.target.get_contacts(with_entity=self.plane)
        if "geom_a" in contacts:
            value = contacts["geom_a"]
            if hasattr(value, "numel"):
                return int(value.numel()) > 0
            return np.asarray(value).size > 0
        return False

    def render(self, *, hdr: bool = False, segmentation: bool = False):
        return self.camera.render(
            rgb=True,
            depth=bool(hdr or segmentation),
            segmentation=bool(segmentation),
            radiance=bool(hdr),
        )


def build_robot_scene(config: Dict[str, Any], *, dt_s: Optional[float] = None) -> RobotScene:
    import genesis as gs

    if getattr(gs, "_initialized", False) is False:
        gs.init(backend=gs.cpu, logging_level="warning")
    physics = config["physics"]
    dt = float(dt_s if dt_s is not None else physics["dt_s"])
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, gravity=tuple(physics["gravity_m_s2"])),
        vis_options=gs.options.VisOptions(ambient_light=(0.25, 0.25, 0.25)),
        show_viewer=False,
    )
    plane = scene.add_entity(
        gs.morphs.Plane(),
        material=gs.materials.Rigid(friction=float(physics["support_friction"])),
        surface=gs.surfaces.Default(color=(0.40, 0.42, 0.44, 1.0)),
    )
    target_cfg = config["target"]
    size = tuple(float(v) for v in target_cfg["size_m"])
    target = scene.add_entity(
        gs.morphs.Box(size=size, pos=(0.80, 0.0, size[2] / 2)),
        material=gs.materials.Rigid(
            friction=float(target_cfg["friction"]),
            rho=float(target_cfg["density_kg_m3"]),
        ),
        surface=gs.surfaces.Default(color=(0.14, 0.54, 0.26, 1.0)),
    )
    package_root = Path(gs.__file__).resolve().parent
    robot_xml = package_root / "assets" / "xml" / "franka_emika_panda" / "panda.xml"
    if not robot_xml.is_file():
        raise FileNotFoundError(f"Genesis Panda asset missing: {robot_xml}")
    robot = scene.add_entity(gs.morphs.MJCF(file=str(robot_xml)))

    camera_cfg = config["camera"]
    w, h = (int(v) for v in camera_cfg["resolution_wh"])
    camera = scene.add_camera(
        res=(w, h),
        pos=tuple(camera_cfg["position_m"]),
        lookat=tuple(camera_cfg["lookat_m"]),
        up=tuple(camera_cfg["up_m"]),
        fov=float(camera_cfg["vertical_fov_deg"]),
        GUI=False,
    )
    scene.build()

    dof = config["robot"]
    robot.set_dofs_kp(np.asarray(dof["arm_kp"] + [100, 100], dtype=np.float32))
    robot.set_dofs_kv(np.asarray(dof["arm_kv"] + [10, 10], dtype=np.float32))
    robot.set_dofs_force_range(
        np.asarray(dof["arm_force_range"][0] + [dof["finger_force_range_each"][0]] * 2, dtype=np.float32),
        np.asarray(dof["arm_force_range"][1] + [dof["finger_force_range_each"][1]] * 2, dtype=np.float32),
    )
    # Only Panda's finger pads touch the target in the intended task. A uniform
    # robot value is used because Genesis exposes the MJCF geometry friction as
    # per-link values but the public material setter is entity-wide.
    robot.set_friction(float(dof["robot_friction_coefficient"]))

    hand = robot.get_link("hand")
    left_finger = robot.get_link("left_finger")
    right_finger = robot.get_link("right_finger")
    hand_quat = np.asarray(dof["hand_orientation_wxyz"], dtype=np.float64)
    tcp = np.asarray(dof["controller_tcp_reference_m"], dtype=np.float64)
    hand_pos = tcp + np.asarray([0.0, 0.0, float(dof["hand_link_z_offset_to_pad_center_m"])])
    q_home = robot.inverse_kinematics(
        link=hand,
        pos=hand_pos.astype(np.float32),
        quat=hand_quat.astype(np.float32),
        init_qpos=as_numpy(robot.get_dofs_position()).reshape(-1).astype(np.float32),
        max_solver_iters=100,
    )
    q_home = as_numpy(q_home).reshape(-1)
    q_home[7:9] = float(dof["initial_finger_joint_position_m_each"])
    arm_dofs = np.arange(7, dtype=np.int32)
    finger_dofs = np.arange(7, 9, dtype=np.int32)
    measured_mass = float(as_numpy(target.get_mass()).reshape(-1)[0])
    expected_mass = float(target_cfg["mass_kg"])
    if abs(measured_mass - expected_mass) > 0.005:
        raise RuntimeError(f"target mass {measured_mass:.6f} kg != configured {expected_mass:.6f} kg")

    return RobotScene(
        gs=gs,
        scene=scene,
        plane=plane,
        target=target,
        robot=robot,
        camera=camera,
        hand=hand,
        left_finger=left_finger,
        right_finger=right_finger,
        home_qpos=q_home,
        hand_quat_wxyz=hand_quat,
        arm_dofs=arm_dofs,
        finger_dofs=finger_dofs,
        dt_s=dt,
        config=config,
    )
