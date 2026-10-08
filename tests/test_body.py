"""The public Tiangong 2 Pro loads correctly: fixed base, locked legs, motors, no self-contact."""

import mujoco
import numpy as np
import pytest

from humanoid_perception.robot import contract as c
from humanoid_perception.robot.loader import COLLISION_GROUP, ROBOT_PREFIX, TIANGONG_LEG_JOINTS
from humanoid_perception.sim.stand import lowest_robot_point, robot_geom_ids


def test_contract_resolves_to_seventeen_motors_plus_waist(sim):
    assert sim.layout.core == c.UPPER_BODY_JOINTS
    assert sim.layout.waist == ("body_yaw_joint",)
    assert sim.layout.hands == ()           # the no-hands URDF
    assert sim.layout.size == 18
    assert len(sim.info.actuated) == 18


def test_legs_are_locked_and_base_is_welded(sim):
    for j in TIANGONG_LEG_JOINTS:
        assert mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, ROBOT_PREFIX + j) < 0
    assert sim.model.body_jntnum[sim.root_body] == 0      # no free joint: a fixed base
    assert sim.info.root_link == "pelvis"


def test_static_links_are_not_fused_so_the_mass_and_frames_survive(sim):
    assert sim.model.body_subtreemass[0] == pytest.approx(sim.info.urdf_mass, rel=0.02)
    for link in ("camera_head_link", "imu_link", "left_tcp_link", "right_tcp_link"):
        assert mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, ROBOT_PREFIX + link) >= 0


def test_robot_stands_just_above_the_floor(sim):
    mujoco.mj_forward(sim.model, sim.data)
    lowest = lowest_robot_point(sim.model, sim.data)
    assert lowest == pytest.approx(sim.cfg.body.floor_clearance, abs=2e-3)
    assert sim.body_position("pelvis")[2] > 0.8          # a standing humanoid, not a crouch


def test_robot_never_collides_with_itself(sim):
    mujoco.mj_forward(sim.model, sim.data)
    robot = set(robot_geom_ids(sim.model))
    for k in range(sim.data.ncon):
        con = sim.data.contact[k]
        assert not (con.geom1 in robot and con.geom2 in robot), "robot-robot contact"


def test_collision_meshes_are_in_the_hidden_render_group(sim):
    hulls = [g for g in robot_geom_ids(sim.model) if sim.model.geom_contype[g]]
    assert hulls and all(sim.model.geom_group[g] == COLLISION_GROUP for g in hulls)


def test_motor_torque_limits_are_the_urdf_efforts(sim):
    for joint, effort in (("shoulder_roll_r_joint", 95.0), ("elbow_pitch_l_joint", 35.0),
                          ("head_pitch_joint", 6.3), ("body_yaw_joint", 91.0)):
        aid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{ROBOT_PREFIX}motor_{joint}")
        assert tuple(sim.model.actuator_forcerange[aid]) == (-effort, effort)


def test_hand_frames_sit_on_the_hands(sim):
    """A tool-centre-point metres from its wrist would make every reach metric meaningless.
    The URDF puts each TCP 8.5 cm below its wrist-roll link; check the compiled model agrees."""
    mujoco.mj_forward(sim.model, sim.data)
    for side in ("right", "left"):
        tcp = sim.body_position(f"{side}_tcp_link")
        wrist = sim.body_position(f"wrist_roll_{side[0]}_link")
        assert np.linalg.norm(tcp - wrist) == pytest.approx(0.0848, abs=1e-3), (side, tcp, wrist)
