"""The RGB-D camera: what a real depth camera hands the perception code, from the sim.

Pinhole model: a camera-frame point (X, Y, Z) (OpenCV: x right, y down, z forward) lands
on pixel u = fx X / Z + cx, v = fy Y / Z + cy. MuJoCo describes a camera by one vertical
field of view, so fy = (H / 2) / tan(fovy / 2) and fx = fy (square pixels). The principal
point is ((W - 1) / 2, (H - 1) / 2): a pixel's integer index names its centre, so the
middle of a 640-wide image is 319.5. Half a pixel is ~1 mm at 1 m, too small to see and
big enough to bias every estimate.

Depth is Z along the optical axis, not the ray length, as RealSense, Orbbec and ROS report
it. Pixels where nothing was hit or outside [min_depth, max_depth] read 0 = no measurement.

Everything in ``RGBDFrame`` is what the real robot also has (its camera pose comes from
the joint encoders through the kinematic chain). Ground truth lives elsewhere, so no
perception code can peek at it by accident.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .geometry import MJ_TO_CV, invert_transform, make_transform, transform_points
from .loader import CameraMount


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_fovy(cls, width: int, height: int, fovy_deg: float) -> "CameraIntrinsics":
        f = (height / 2) / np.tan(np.radians(fovy_deg) / 2)
        return cls(width, height, f, f, (width - 1) / 2, (height - 1) / 2)

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    def project(self, points_cam) -> tuple[np.ndarray, np.ndarray]:
        """Camera-frame points (N, 3) -> pixels (N, 2) as (u, v) and depth Z (N,). Check Z > 0."""
        p = np.atleast_2d(np.asarray(points_cam, dtype=np.float64))
        z = p[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.fx * p[:, 0] / z + self.cx
            v = self.fy * p[:, 1] / z + self.cy
        return np.column_stack([u, v]), z

    def backproject(self, depth: np.ndarray) -> np.ndarray:
        """Depth (H, W) -> camera-frame point per pixel (H, W, 3); depth 0 gives (0, 0, 0)."""
        v, u = np.indices(depth.shape, dtype=np.float64)
        z = depth.astype(np.float64)
        return np.stack([(u - self.cx) * z / self.fx, (v - self.cy) * z / self.fy, z], axis=-1)


@dataclass
class RGBDFrame:
    camera_name: str
    rgb: np.ndarray            # (H, W, 3) uint8
    depth: np.ndarray          # (H, W) float32 metres, Z-depth, 0 = no measurement
    intrinsics: CameraIntrinsics
    T_world_cam: np.ndarray    # (4, 4): OpenCV camera frame -> world
    time: float

    @property
    def valid(self) -> np.ndarray:
        return self.depth > 0

    def point_cloud(self, frame: str = "camera", mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        keep = self.valid if mask is None else (self.valid & mask)
        pts = self.intrinsics.backproject(self.depth)[keep]
        if frame == "world":
            pts = transform_points(self.T_world_cam, pts)
        elif frame != "camera":
            raise ValueError("frame must be 'camera' or 'world'")
        return pts, self.rgb[keep]


class RGBDCamera:
    """Renders one MuJoCo camera into RGB and depth. One renderer, created once, reused."""

    def __init__(self, model: mujoco.MjModel, cfg: CameraMount):
        self.cfg = cfg
        self.model = model
        self.cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cfg.name)
        if self.cam_id < 0:
            raise KeyError(f"camera {cfg.name!r} is not in the model")
        self._renderer = mujoco.Renderer(model, height=cfg.height, width=cfg.width)
        # Draw groups 0-2 only: group 3 holds the robot's collision hulls.
        self._scene_option = mujoco.MjvOption()
        self._scene_option.geomgroup[:] = [1, 1, 1, 0, 0, 0]

    @property
    def name(self) -> str:
        return self.cfg.name

    def intrinsics(self) -> CameraIntrinsics:
        return CameraIntrinsics.from_fovy(self.cfg.width, self.cfg.height, float(self.model.cam_fovy[self.cam_id]))

    def pose(self, data: mujoco.MjData) -> np.ndarray:
        """T_world_cam in the OpenCV camera frame. Kinematics must be current (mj_forward)."""
        R_world_mj = data.cam_xmat[self.cam_id].reshape(3, 3)
        return make_transform(R_world_mj @ MJ_TO_CV, data.cam_xpos[self.cam_id])

    def T_cam_world(self, data: mujoco.MjData) -> np.ndarray:
        return invert_transform(self.pose(data))

    def render_rgb(self, data: mujoco.MjData) -> np.ndarray:
        r = self._renderer
        r.update_scene(data, camera=self.cam_id, scene_option=self._scene_option)
        return r.render().copy()

    def render(self, data: mujoco.MjData) -> RGBDFrame:
        r = self._renderer
        rgb = self.render_rgb(data)
        r.enable_depth_rendering()
        try:
            r.update_scene(data, camera=self.cam_id, scene_option=self._scene_option)
            depth = r.render().astype(np.float32)
        finally:
            r.disable_depth_rendering()
        far = self.model.vis.map.zfar * self.model.stat.extent
        no_hit = depth >= far * 0.999
        out_of_range = (depth < self.cfg.min_depth) | (depth > self.cfg.max_depth)
        depth[no_hit | out_of_range] = 0.0
        return RGBDFrame(self.name, rgb, depth, self.intrinsics(), self.pose(data), float(data.time))

    def close(self) -> None:
        self._renderer.close()
