"""The VLA loop's own pieces: the safety filter and the table scene the policy looks at."""

import os
from pathlib import Path

import numpy as np
import pytest

from humanoid_perception.robot.loader import BodyConfig
from humanoid_perception.sim.stand import StandingSim, TableConfig, WorldConfig
from humanoid_perception.vla.safety import SafetyFilter

PRIVATE = Path(os.environ.get("HUMANOID_PERCEPTION_PRIVATE", Path(__file__).resolve().parents[1] / "private"))


def test_safety_filter_clamps_to_envelope_and_joint_range():
    rng = np.array([[-1.0, 1.0], [-0.1, 2.0], [-3.0, 3.0]])
    f = SafetyFilter(rng, max_rate=1000.0, max_offset=np.array([0.5, 0.5, 0.5]))
    out = f(np.array([0.9, -0.9, 0.2]), previous=np.zeros(3))
    assert np.allclose(out, [0.5, -0.1, 0.2])       # envelope, then the joint's own lower limit
    assert f.clamped == 2 and f.rate_limited == 0


def test_safety_filter_rate_limits_each_step():
    f = SafetyFilter(np.array([[-3.0, 3.0]]), max_rate=1.0, control_hz=50.0)
    cmd = np.zeros(1)
    for _ in range(10):
        cmd = f(np.array([1.0]), cmd)
    assert np.isclose(cmd[0], 10 * 1.0 / 50.0)       # 0.02 rad per step, never a jump
    assert f.rate_limited == 10


def _red_pixels(body: BodyConfig) -> int:
    sim = StandingSim(WorldConfig(body=body, table=TableConfig()))
    try:
        rgb = sim.render().rgb.astype(int)
    finally:
        sim.close()
    return int(np.count_nonzero((rgb[..., 0] > 150) & (rgb[..., 1] < 60) & (rgb[..., 2] < 60)))


def test_table_objects_rest_on_the_table():
    from humanoid_perception.sim.stand import SceneObject
    import mujoco
    objs = (SceneObject("box", "box", (0.04, 0.06, 0.05), (0.1, 0.2, 0.9, 1.0), (0.6, -0.2)),
            SceneObject("bottle", "cylinder", (0.035, 0.11, 0.0), (0.1, 0.7, 0.2, 1.0), (0.65, 0.2)))
    sim = StandingSim(WorldConfig(table=TableConfig(objects=objs)))
    try:
        for o, half_h in zip(objs, (0.05, 0.11)):
            g = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, f"obj_{o.name}")
            assert g >= 0
            assert np.allclose(sim.data.geom_xpos[g], [o.xy[0], o.xy[1], 0.75 + half_h])
    finally:
        sim.close()
    with pytest.raises(ValueError):
        StandingSim(WorldConfig(table=TableConfig(objects=(SceneObject("x", "sphere", (0.1, 0, 0), (1, 1, 1, 1), (0.5, 0)),))))


def test_head_camera_sees_the_cup_on_the_public_body():
    assert _red_pixels(BodyConfig()) > 200


@pytest.mark.skipif(not (PRIVATE / "bodies").is_dir(), reason="no private bodies on this machine")
@pytest.mark.parametrize("yaml", sorted((PRIVATE / "bodies").glob("*.yaml")) if (PRIVATE / "bodies").is_dir() else [])
def test_head_camera_sees_the_cup_on_private_bodies(yaml):
    body = BodyConfig.from_yaml(yaml)
    assert _red_pixels(body) > 200
