"""The same M0 gate on a private body, when one is present.

The repository ships only the public Tiangong 2 Pro, but the VLA is validated live on our
own robot in the lab. That body is described in the gitignored ``private/`` folder and must
pass the same gate as the public one: load, hold HOME, render from its head camera. On a
machine without ``private/bodies/*.yaml`` every test here is skipped.
"""

import os
from pathlib import Path

import mujoco
import numpy as np
import pytest

from humanoid_perception.robot import contract as c
from humanoid_perception.robot.loader import BodyConfig
from humanoid_perception.sim.stand import StandingSim, WorldConfig

# <repo>/private by default; HUMANOID_PERCEPTION_PRIVATE points elsewhere (e.g. from a worktree).
PRIVATE = Path(os.environ.get("HUMANOID_PERCEPTION_PRIVATE", Path(__file__).resolve().parents[1] / "private")) / "bodies"
BODIES = sorted(PRIVATE.glob("*.yaml")) if PRIVATE.is_dir() else []

pytestmark = pytest.mark.skipif(not BODIES, reason="no private body on this machine")


@pytest.fixture(scope="module", params=BODIES, ids=lambda p: p.stem)
def private_sim(request):
    s = StandingSim(WorldConfig(body=BodyConfig.from_yaml(request.param)))
    yield s
    s.close()


def test_private_body_takes_the_contract(private_sim):
    assert private_sim.layout.core == c.UPPER_BODY_JOINTS
    assert private_sim.layout.size >= 17
    assert private_sim.model.body_subtreemass[0] == pytest.approx(private_sim.info.urdf_mass, rel=0.02)


def test_private_body_mimics_become_constraints(private_sim):
    names = [mujoco.mj_id2name(private_sim.model, mujoco.mjtObj.mjOBJ_EQUALITY, i)
             for i in range(private_sim.model.neq)]
    assert sum(n.startswith("robot/mimic_") for n in names) == len(private_sim.info.mimic)
    assert sum(n.startswith("robot/hold_") for n in names) == len(private_sim.info.holds)


def test_private_body_holds_home(private_sim):
    private_sim.set_targets(np.zeros(private_sim.layout.size))
    worst = 0.0
    for _ in range(50 * 10):
        private_sim.control_step()
        worst = max(worst, float(np.abs(private_sim.joint_offsets()).max()))
    assert np.degrees(worst) < 1.0


def test_private_body_tracks_a_head_target(private_sim):
    target = np.zeros(private_sim.layout.size)
    i = private_sim.layout.index("head_pitch_joint")
    target[i] = 0.2
    private_sim.set_targets(target)
    for _ in range(50 * 3):
        private_sim.control_step()
    assert private_sim.joint_offsets()[i] == pytest.approx(0.2, abs=np.radians(1.0))
    private_sim.set_targets(np.zeros(private_sim.layout.size))
    for _ in range(50 * 3):
        private_sim.control_step()


def test_private_body_renders_from_its_head_camera(private_sim):
    frame = private_sim.render()
    assert frame.rgb.shape[2] == 3 and frame.depth.shape == frame.rgb.shape[:2]
    assert frame.valid.mean() > 0.2
    forward = frame.T_world_cam[:3, 2]
    assert forward[0] > 0.5                               # looks ahead of the robot
    assert forward[2] < 0                                 # and downward, at the table
