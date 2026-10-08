"""The head camera: a generic RGB-D on camera_head_link, looking out of the lens, slightly down."""

import numpy as np
import pytest


def test_intrinsics_follow_the_vertical_fov(sim):
    k = sim.camera.intrinsics()
    assert (k.width, k.height) == (640, 480)
    assert k.fx == pytest.approx(240 / np.tan(np.radians(29.0)), abs=0.05)   # fovy 58 deg
    assert k.fy == k.fx
    assert (k.cx, k.cy) == (319.5, 239.5)


def test_camera_looks_along_the_optical_z_axis(sim):
    import mujoco

    mujoco.mj_forward(sim.model, sim.data)
    forward = sim.camera.pose(sim.data)[:3, 2]           # OpenCV z = viewing direction, in world
    link_z = sim.body_rotation("camera_head_link")[:, 2]
    assert np.dot(forward, link_z) == pytest.approx(1.0, abs=1e-6)


def test_camera_at_home_looks_ahead_and_30_degrees_down(sim):
    """At the URDF zero pose the lens points straight ahead in yaw and 30 deg below the
    horizontal (measured 2026-10-08: forward = (0.866, 0, -0.5), camera 1.647 m up)."""
    import mujoco

    sim.set_targets(np.zeros(sim.layout.size))
    mujoco.mj_forward(sim.model, sim.data)
    forward = sim.camera.pose(sim.data)[:3, 2]
    down = np.degrees(np.arcsin(-forward[2]))
    yaw = np.degrees(np.arctan2(forward[1], forward[0]))
    assert down == pytest.approx(30.0, abs=1.0)
    assert yaw == pytest.approx(0.0, abs=1.0)
    assert sim.camera.pose(sim.data)[2, 3] == pytest.approx(1.647, abs=0.02)


def test_render_gives_rgb_and_metric_depth(sim):
    frame = sim.render()
    assert frame.rgb.shape == (480, 640, 3) and frame.rgb.dtype == np.uint8
    assert frame.depth.shape == (480, 640) and frame.depth.dtype == np.float32
    valid = frame.valid
    assert valid.mean() > 0.3                             # floor and wall ahead are in range
    assert frame.depth[valid].min() >= 0.2 and frame.depth[valid].max() <= 4.0
    assert frame.rgb.std() > 10                           # not a blank image


def test_floor_backprojects_to_z_zero(sim):
    """Pixels on the floor, lifted through the intrinsics and the camera pose, land at z = 0."""
    frame = sim.render()
    pts, _ = frame.point_cloud("world")
    floor = pts[np.abs(pts[:, 2]) < 0.05]
    assert len(floor) > 0.1 * len(pts)
    assert np.abs(floor[:, 2]).mean() < 0.01
