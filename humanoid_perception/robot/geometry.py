"""Rigid-body geometry: rotations, 4x4 transforms, camera frame conventions.

CONVENTIONS (stated once, used everywhere)
  Quaternions are (w, x, y, z), MuJoCo's order. Reorder at the boundary if one is ever
  handed to scipy or ROS, never inside this package.
  A transform T_a_b maps points written in frame b into frame a: p_a = T_a_b @ p_b.
  Camera frame = OpenCV: x right, y DOWN, z FORWARD. MuJoCo's own camera frame is the
  OpenGL one (x right, y up, looking down -z); the two differ by a half turn about x,
  ``MJ_TO_CV``, applied in one place (camera.py).
  Metres and radians everywhere, except config files, which take degrees.
"""

from __future__ import annotations

import mujoco
import numpy as np

# OpenCV camera frame <-> MuJoCo camera frame. Its own inverse.
MJ_TO_CV = np.diag([1.0, -1.0, -1.0])

# A ROS camera_link frame is x forward, y left, z up. A MuJoCo camera looks down its -z
# axis with +y up in the image. Written in the camera_link frame, the MuJoCo camera's
# axes are x (image right) = -y_link, y (image up) = +z_link, z = -x_link: it looks along
# +x_link, out of the lens.
ROS_LINK_TO_MJ_CAMERA = np.array([[0.0, 0.0, -1.0],
                                  [-1.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0]])


def quat_to_mat(quat_wxyz) -> np.ndarray:
    q = np.asarray(quat_wxyz, dtype=np.float64)
    q = q / np.linalg.norm(q)
    out = np.zeros(9)
    mujoco.mju_quat2Mat(out, q)
    return out.reshape(3, 3)


def mat_to_quat(R) -> np.ndarray:
    """3x3 rotation -> (w, x, y, z) with w >= 0, so equal rotations print the same numbers."""
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R, dtype=np.float64).reshape(9))
    return -q if q[0] < 0 else q


def rpy_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ROS / URDF rpy in radians, R = Rz(yaw) @ Ry(pitch) @ Rx(roll) -> (w, x, y, z)."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def make_transform(R, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def invert_transform(T) -> np.ndarray:
    """Exact inverse of a rigid transform (R^T, not a general matrix inverse)."""
    R, t = T[:3, :3], T[:3, 3]
    return make_transform(R.T, -R.T @ t)


def transform_points(T, points) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    return p @ T[:3, :3].T + T[:3, 3]
