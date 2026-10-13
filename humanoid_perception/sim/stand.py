"""The standing robot: fixed base, motors holding a posture, a camera on its head.

This is the phase-1 world of docs/PLAN.md. There is no locomotion policy in it: the body
is bolted to the world with its legs locked, and the upper-body motors hold whatever the
action stream asks. "Standing" here means exactly that; the real robot's equivalent is a
gantry.

The action interface is the contract: ``set_targets(offsets)`` takes radians as offsets
from HOME in ``layout.all`` order, and ``joint_offsets()`` reads them back the same way.
HOME is the URDF zero posture.

Physics: 2 ms steps, implicitfast integration (stable with motor damping), elliptic
friction cones. The control rate is a separate number, 50 Hz, the usual policy step of
humanoid locomotion controllers, so the same stream can feed one later.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from ..robot.camera import RGBDCamera, RGBDFrame
from ..robot.contract import JointLayout, resolve_layout
from ..robot.loader import ROBOT_PREFIX, BodyConfig, BodyInfo, add_mounted_camera, attach_body

CONTROL_HZ = 50.0


@dataclass(frozen=True)
class SceneObject:
    """A plain object resting on the table. ``size`` follows MuJoCo: a cylinder is
    (radius, half-height, 0), a box is its three half-extents. Its geom is named
    ``obj_<name>``; move it at run time through ``model.geom_pos``."""

    name: str
    shape: str                                   # "cylinder" or "box"
    size: tuple[float, float, float]
    rgba: tuple[float, float, float, float]
    xy: tuple[float, float]


@dataclass
class TableConfig:
    """A plain table in front of the robot with a red cup and optional other objects on it
    (positions in the world frame, x forward from the robot's root, metres). Static
    geometry: nothing on it moves under physics."""

    center: tuple[float, float] = (0.60, 0.0)
    size: tuple[float, float] = (0.6, 1.0)       # depth (x), width (y)
    height: float = 0.75                         # top surface above the floor
    # 0.70 m out keeps the cup well inside the head camera's view on both bodies (a taller
    # robot with a 30 deg-down camera saw a cup at 0.55 m only at the bottom edge).
    cup: tuple[float, float] | None = (0.70, 0.05)
    cup_radius: float = 0.04
    cup_height: float = 0.10
    objects: tuple[SceneObject, ...] = ()


@dataclass
class WorldConfig:
    timestep: float = 0.002
    room: bool = True            # floor and three walls, so the background has real depth
    room_size: float = 3.0       # half-extent of the floor, metres
    body: BodyConfig | None = None
    table: TableConfig | None = None   # None = empty room (the M0 world)

    def __post_init__(self):
        if self.body is None:
            self.body = BodyConfig()


def add_table(spec: mujoco.MjSpec, t: TableConfig) -> None:
    top = 0.02
    (cx, cy), (dx, dy) = t.center, t.size
    spec.worldbody.add_geom(name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX,
                            pos=[cx, cy, t.height - top], size=[dx / 2, dy / 2, top],
                            rgba=[0.62, 0.47, 0.32, 1.0], contype=1, conaffinity=1)
    leg_h = (t.height - 2 * top) / 2
    spec.worldbody.add_geom(name="table_pedestal", type=mujoco.mjtGeom.mjGEOM_BOX,
                            pos=[cx, cy, leg_h], size=[0.06, 0.06, leg_h],
                            rgba=[0.35, 0.30, 0.25, 1.0], contype=1, conaffinity=1)
    if t.cup is not None:
        spec.worldbody.add_geom(name="cup", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                                pos=[t.cup[0], t.cup[1], t.height + t.cup_height / 2],
                                size=[t.cup_radius, t.cup_height / 2, 0],
                                rgba=[0.85, 0.08, 0.08, 1.0], contype=1, conaffinity=1)
    shapes = {"cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER, "box": mujoco.mjtGeom.mjGEOM_BOX}
    for o in t.objects:
        if o.shape not in shapes:
            raise ValueError(f"object {o.name!r}: shape must be one of {sorted(shapes)}")
        half_h = o.size[1] if o.shape == "cylinder" else o.size[2]
        spec.worldbody.add_geom(name=f"obj_{o.name}", type=shapes[o.shape],
                                pos=[o.xy[0], o.xy[1], t.height + half_h], size=list(o.size),
                                rgba=list(o.rgba), contype=1, conaffinity=1)


def build_world(cfg: WorldConfig) -> tuple[mujoco.MjSpec, BodyInfo]:
    spec = mujoco.MjSpec()
    spec.modelname = "standing_" + cfg.body.name
    spec.option.timestep = cfg.timestep
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10.0
    spec.visual.map.znear = 0.01

    spec.worldbody.add_light(pos=[0.5, -1.0, 2.5], dir=[-0.2, 0.4, -1.0])
    spec.worldbody.add_light(pos=[-1.0, 1.5, 2.5], dir=[0.4, -0.5, -1.0])

    tex = spec.add_texture(name="floor_tex", type=mujoco.mjtTexture.mjTEXTURE_2D,
                           builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, width=256, height=256,
                           rgb1=[0.55, 0.55, 0.55], rgb2=[0.42, 0.42, 0.42])
    mat = spec.add_material(name="floor_mat", texrepeat=[6, 6], reflectance=0.05)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
    s = cfg.room_size
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[s, s, 0.05],
                            material="floor_mat", contype=1, conaffinity=1)
    if cfg.room:
        h = 1.5
        for name, pos, size in (
            ("wall_front", [s, 0, h], [0.02, s, h]),
            ("wall_left", [0, s, h], [s, 0.02, h]),
            ("wall_right", [0, -s, h], [s, 0.02, h]),
        ):
            spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=size,
                                    rgba=[0.80, 0.80, 0.78, 1.0], contype=1, conaffinity=1)

    if cfg.table is not None:
        add_table(spec, cfg.table)

    info = attach_body(spec, cfg.body)
    add_mounted_camera(spec, cfg.body.camera)
    return spec, info


def robot_geom_ids(model: mujoco.MjModel) -> list[int]:
    """URDF geoms carry no names; a geom is the robot's if its body is."""
    return [g for g in range(model.ngeom)
            if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]).startswith(ROBOT_PREFIX)]


def lowest_robot_point(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """World z of the lowest point of any robot geom in the current pose.

    Mesh geoms are measured from their vertices (exact); any other shape from its
    bounding sphere (conservative). Kinematics must be current (mj_forward).
    """
    z = np.inf
    for g in robot_geom_ids(model):
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            m = model.geom_dataid[g]
            adr, n = model.mesh_vertadr[m], model.mesh_vertnum[m]
            verts = model.mesh_vert[adr:adr + n]
            world = verts @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
            z = min(z, float(world[:, 2].min()))
        else:
            z = min(z, float(data.geom_xpos[g][2] - model.geom_rbound[g]))
    return z


class StandingSim:
    """The standing robot with motors, stepped at the physics rate, commanded at 50 Hz."""

    def __init__(self, cfg: WorldConfig | None = None):
        self.cfg = cfg or WorldConfig()
        self.spec, self.info = build_world(self.cfg)
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        self.layout: JointLayout = resolve_layout(self.info.name, self.info.actuated, self.cfg.body.waist_joint)

        m = self.model
        self._act_ids = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{ROBOT_PREFIX}motor_{j}")
                                  for j in self.layout.all])
        jnt_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, ROBOT_PREFIX + j) for j in self.layout.all]
        if min(self._act_ids) < 0 or min(jnt_ids) < 0:
            raise RuntimeError("a contract joint has no motor or is missing from the compiled model")
        self._qpos_adr = np.array([m.jnt_qposadr[j] for j in jnt_ids])
        self._qvel_adr = np.array([m.jnt_dofadr[j] for j in jnt_ids])
        self.joint_range = np.array([m.jnt_range[j] for j in jnt_ids])
        self.home = np.zeros(self.layout.size)   # URDF zero posture
        self.steps_per_control = int(round(1.0 / (CONTROL_HZ * self.cfg.timestep)))
        self._camera: RGBDCamera | None = None
        self.root_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, ROBOT_PREFIX + self.info.root_link)

        self.set_targets(np.zeros(self.layout.size))
        mujoco.mj_forward(self.model, self.data)
        self._settle_height()

    def _settle_height(self) -> None:
        """With position z = 0 ("auto"), raise the bolted root so the robot's lowest point
        clears the floor by ``floor_clearance``. A fixed body's position is a model field,
        so this needs no recompile."""
        body = self.cfg.body
        if body.position[2] != 0.0 or body.floor_clearance is None:
            return
        lowest = lowest_robot_point(self.model, self.data)
        self.model.body_pos[self.root_body][2] += body.floor_clearance - lowest
        mujoco.mj_forward(self.model, self.data)

    # ----------------------------------------------------------------- the action stream

    def set_targets(self, offsets: np.ndarray) -> None:
        """Command the motors, radians as offsets from HOME in ``layout.all`` order."""
        offsets = np.asarray(offsets, dtype=np.float64)
        if offsets.shape != (self.layout.size,):
            raise ValueError(f"expected {self.layout.size} targets, got {offsets.shape}")
        self.data.ctrl[self._act_ids] = self.home + offsets

    def joint_offsets(self) -> np.ndarray:
        """Measured joint angles as offsets from HOME, ``layout.all`` order."""
        return self.data.qpos[self._qpos_adr] - self.home

    def joint_velocities(self) -> np.ndarray:
        return self.data.qvel[self._qvel_adr].copy()

    # -------------------------------------------------------------------------- the clock

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)

    def control_step(self) -> None:
        """Advance one control period (1 / CONTROL_HZ) of physics."""
        self.step(self.steps_per_control)

    def run(self, seconds: float) -> None:
        self.step(int(round(seconds / self.cfg.timestep)))

    @property
    def time(self) -> float:
        return float(self.data.time)

    # ------------------------------------------------------------------------- the camera

    @property
    def camera(self) -> RGBDCamera:
        if self._camera is None:
            self._camera = RGBDCamera(self.model, self.cfg.body.camera)
        return self._camera

    def render(self) -> RGBDFrame:
        mujoco.mj_forward(self.model, self.data)
        return self.camera.render(self.data)

    # ------------------------------------------------------------------------ kinematics

    def body_position(self, name: str) -> np.ndarray:
        """World position of a robot body by its UNprefixed URDF link name."""
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, ROBOT_PREFIX + name)
        if bid < 0:
            raise KeyError(f"no robot body {name!r}")
        return self.data.xpos[bid].copy()

    def body_rotation(self, name: str) -> np.ndarray:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, ROBOT_PREFIX + name)
        if bid < 0:
            raise KeyError(f"no robot body {name!r}")
        return self.data.xmat[bid].reshape(3, 3).copy()

    def close(self) -> None:
        if self._camera is not None:
            self._camera.close()
            self._camera = None
