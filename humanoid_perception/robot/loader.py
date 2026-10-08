"""Load a body's URDF into MuJoCo correctly, with motors, and bolt it to the world.

A ROS URDF loaded naively into MuJoCo compiles, renders, and can be wrong in four silent
ways. This module applies the fixes, each of which is checked by a test:

  1. MIMIC JOINTS ARE DROPPED. ``<mimic>`` couples a follower to a leader (finger knuckles,
     crank shafts). MuJoCo's importer ignores the tag, so every follower becomes a free
     joint. Fix: read the tags and add a joint-equality constraint per coupling.
  2. FIXED LINKS ARE MERGED AWAY. MuJoCo fuses fixed-joint children into their parent by
     default, which deletes camera and IMU frames (nowhere to mount a camera) and fuses a
     bolted-down base into the world (its mass vanishes). Fix: fusestatic off.
  3. THE ROBOT COLLIDES WITH ITSELF. Neighbouring collision meshes overlap in the zero
     pose; ROS skips those pairs via the SRDF. Fix: robot geoms never collide with each
     other, only with the world and the objects (contype / conaffinity bits).
  4. COLLISION MESHES ARE VISIBLE. They import into render group 0, which cameras draw.
     Fix: group 3, which the camera never draws, while physics keeps them.

Plus: continuous joints carrying limits that MuJoCo would wrongly enforce (stripped, per
the URDF rule), and visual and collision meshes sharing a file stem that MuJoCo silently
merges (each mesh gets a unique placeholder name; the real file goes back on the spec).

Motors are position servos, torque = kp (target - q) - kv q_dot, clipped to the joint's
torque limit (the URDF effort, or a rated-torque file when the body ships one). Every
robot body has gravity compensation on, as real arm controllers do, so kp only has to
fight the task, not the arm's own weight.

The default ``BodyConfig`` is the PUBLIC body this repository ships: the Tiangong 2 Pro
from the Open X-Humanoid URDF release (assets/robots/tiangong2pro, OpenAtom Open Hardware
License 1.0). A private robot plugs in through ``BodyConfig.from_yaml`` without touching
this file.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import yaml

from .geometry import MJ_TO_CV, ROS_LINK_TO_MJ_CAMERA, mat_to_quat, quat_to_mat, rpy_to_quat

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ASSETS = PACKAGE_ROOT / "assets"

ROBOT_PREFIX = "robot/"      # every robot element's name in the scene starts with this
COLLISION_GROUP = 3          # render group for collision meshes: physics yes, cameras no
# Two geoms touch if (a.contype & b.conaffinity) or (b.contype & a.conaffinity).
# Scene geoms are 1/1. Robot 2/1: robot-scene -> 1 & 1 = yes; robot-robot -> 2 & 1 = no.
ROBOT_CONTYPE, ROBOT_CONAFFINITY = 2, 1

#: The Tiangong 2 Pro's twelve leg joints. Locked (made fixed) for the standing phase.
TIANGONG_LEG_JOINTS: tuple[str, ...] = tuple(
    f"{j}_{s}_joint" for s in ("l", "r")
    for j in ("hip_roll", "hip_pitch", "hip_yaw", "knee_pitch", "ankle_pitch", "ankle_roll")
)


@dataclass
class CameraMount:
    """A camera fixed to a robot body.

    ``frame`` says how to read the mount body's axes: ``"optical"`` (OpenCV / ROS optical
    frame: z out of the lens, x right, y down) or ``"ros_link"`` (ROS camera_link: x out
    of the lens, y left, z up). The Tiangong's ``camera_head_link`` is an optical frame.

    Defaults are the Orbbec Gemini E depth stream from its datasheet (v1.1): 640x480,
    FOV 79 x 62 deg, range 0.2-2.5 m. MuJoCo takes one vertical FOV with square pixels, so
    horizontal comes out 77.4 deg. The same camera is on every body this project runs on,
    in simulation and in the lab, so sim images match the real ones; a unit's factory
    calibration replaces the datasheet numbers when it is read out.
    """

    name: str = "head_cam"
    mount: str = ROBOT_PREFIX + "camera_head_link"
    frame: str = "optical"
    width: int = 640
    height: int = 480
    fovy_deg: float = 62.0
    min_depth: float = 0.2
    max_depth: float = 2.5
    mount_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    mount_rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class BodyConfig:
    """One robot body: where its URDF is and how it is prepared for MuJoCo."""

    name: str = "tiangong2pro"
    urdf: Path = ASSETS / "robots/tiangong2pro/urdf/tiangong2.0_pro_urdf.urdf"
    # ROS "package://<name>/..." mesh paths -> the folder that package lives in.
    packages: dict[str, Path] = field(default_factory=lambda: {
        "tiangong2pro_urdf": ASSETS / "robots/tiangong2pro",
    })
    # Where the URDF root is bolted. A z of 0 means "auto": raise the root so the lowest
    # point of the robot clears the floor by ``floor_clearance`` (see sim/stand.py).
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw_deg: float = 0.0
    floor_clearance: float = 0.01
    # Joints turned into fixed joints. For the standing phase the legs are a statue.
    lock_joints: tuple[str, ...] = TIANGONG_LEG_JOINTS
    # Position motors: (joint-name regex, kp, kv), first match wins. Tiers follow the
    # URDF torque classes (91-95, 35, 24, 6.3 N.m); the gains themselves are placeholders.
    actuators: tuple[tuple[str, float, float], ...] = (
        (r"shoulder_(pitch|roll)_[lr]_joint", 100.0, 2.0),
        (r"body_yaw_joint", 100.0, 2.0),
        (r"(shoulder_yaw|elbow_pitch)_[lr]_joint", 40.0, 1.0),
        (r"elbow_yaw_[lr]_joint", 30.0, 0.8),
        (r"wrist_(pitch|roll)_[lr]_joint", 15.0, 0.4),
        (r"head_(yaw|pitch|roll)_joint", 15.0, 0.4),
    )
    # Optional ros2_control xacro with "rated_torque" per joint; joints it does not list
    # fall back to the URDF <limit effort>.
    torque_limits: Path | None = None
    # The body's own name for its waist joint (None = no waist channel).
    waist_joint: str | None = "body_yaw_joint"
    # Passive damping on joints with no motor of their own, e.g. a heavy mimic follower.
    joint_damping: dict[str, float] = field(default_factory=dict)
    # Joints pinned at an angle by a constraint; their mimic followers follow.
    hold_joints: dict[str, float] = field(default_factory=dict)
    # Sliding friction of the robot's collision shapes. None = MuJoCo's default.
    hand_friction: float | None = None
    camera: CameraMount = field(default_factory=CameraMount)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BodyConfig":
        """A private or alternative body described in YAML; paths resolve next to the file.

        Keys are the field names above. ``camera`` is a mapping with ``CameraMount`` keys;
        ``actuators`` is a list of ``{joints: regex, kp: n, kv: n}``.
        """
        path = Path(path).resolve()
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        base = path.parent

        def rel(p):
            p = Path(p)
            return p if p.is_absolute() else base / p

        kw = {}
        for key, value in raw.items():
            if key == "urdf":
                kw[key] = rel(value)
            elif key == "torque_limits":
                kw[key] = rel(value) if value else None
            elif key == "packages":
                kw[key] = {k: rel(v) for k, v in value.items()}
            elif key == "actuators":
                kw[key] = tuple((str(a["joints"]), float(a["kp"]), float(a.get("kv", 0.0))) for a in value)
            elif key == "lock_joints":
                kw[key] = tuple(value)
            elif key in ("position", "mount_pos"):
                kw[key] = tuple(float(x) for x in value)
            elif key == "camera":
                cam = dict(value)
                for k in ("mount_pos", "mount_rpy_deg"):
                    if k in cam:
                        cam[k] = tuple(float(x) for x in cam[k])
                kw[key] = CameraMount(**cam)
            else:
                kw[key] = value
        return cls(**kw)

    def resolve_mesh(self, filename: str, urdf_dir: Path) -> Path:
        if filename.startswith("package://"):
            pkg, _, rel = filename[len("package://"):].partition("/")
            if pkg not in self.packages:
                raise ValueError(f"URDF mesh {filename!r}: package {pkg!r} is not in BodyConfig.packages")
            return Path(self.packages[pkg]) / rel
        if filename.startswith("file://"):
            return Path(filename[len("file://"):])
        p = Path(filename)
        return p if p.is_absolute() else urdf_dir / p


@dataclass
class MimicJoint:
    follower: str
    leader: str
    multiplier: float
    offset: float


@dataclass
class BodyInfo:
    """What the rest of the package needs to know about the loaded body (names UNprefixed)."""

    name: str
    root_link: str
    joints: list[str]                     # every movable joint after locking
    mimic: list[MimicJoint]
    locked: list[str]
    urdf_mass: float                      # sum of <inertial><mass> in the URDF, kg
    mesh_files: dict[str, Path] = field(default_factory=dict)
    effort: dict[str, float] = field(default_factory=dict)
    holds: dict[str, float] = field(default_factory=dict)
    actuated: list[str] = field(default_factory=list)      # joints with a motor, actuator order
    independent: list[str] = field(default_factory=list)   # non-mimic movable joints

    def __post_init__(self):
        followers = {m.follower for m in self.mimic}
        self.independent = [j for j in self.joints if j not in followers]


# ------------------------------------------------------------------------------ URDF prep


def actuator_gains(cfg: BodyConfig, joint: str) -> tuple[float, float] | None:
    for pattern, kp, kv in cfg.actuators:
        if re.fullmatch(pattern, joint):
            return kp, kv
    return None


def read_rated_torques(path: Path) -> dict[str, float]:
    """rated_torque per joint from a ros2_control xacro, N.m."""
    out: dict[str, float] = {}
    for j in ET.parse(path).getroot().iter("joint"):
        for p in j.findall("param"):
            if p.get("name") == "rated_torque" and j.get("name"):
                out[j.get("name")] = float(p.text)
    return out


def prepare_urdf(cfg: BodyConfig) -> tuple[str, BodyInfo]:
    """Read the URDF and fix it up for MuJoCo. Returns (URDF text, BodyInfo).

    Done on the XML tree, never with text replacement, so attribute order or a comment can
    never fool it.
    """
    path = Path(cfg.urdf)
    root = ET.parse(path).getroot()

    for tag in ("ros2_control", "gazebo", "transmission"):
        for el in list(root.findall(tag)):
            root.remove(el)

    # Unique placeholder name per (role, file): MuJoCo names a mesh asset after its file
    # stem and silently merges equal names; visual meshes and collision meshes often share
    # stems.
    mesh_files: dict[str, Path] = {}
    placeholder_of: dict[tuple[str, Path], str] = {}
    for role in ("visual", "collision"):
        for el in root.iter(role):
            for mesh in el.iter("mesh"):
                full = cfg.resolve_mesh(mesh.get("filename", ""), path.parent)
                if not full.is_file():
                    raise FileNotFoundError(f"URDF mesh file not found: {full}")
                key = (role, full.resolve())
                if key not in placeholder_of:
                    placeholder_of[key] = f"{full.stem}__{role}_{len(placeholder_of)}{full.suffix}"
                    mesh_files[placeholder_of[key]] = full.resolve()
                mesh.set("filename", (full.parent / placeholder_of[key]).as_posix())

    joints = {j.get("name"): j for j in root.findall("joint")}

    mimicked = {j.find("mimic").get("joint"): n for n, j in joints.items() if j.find("mimic") is not None}
    for name in cfg.lock_joints:
        if name not in joints:
            raise ValueError(f"lock_joints: no joint {name!r} in the URDF")
        if joints[name].find("mimic") is not None or name in mimicked:
            raise ValueError(f"lock_joints: {name!r} takes part in a mimic coupling and cannot be locked")
        j = joints[name]
        j.set("type", "fixed")
        for child in list(j):
            if child.tag in ("limit", "axis", "dynamics", "mimic", "calibration", "safety_controller"):
                j.remove(child)

    # URDF rule: a continuous joint has no position limits. MuJoCo's importer would enforce
    # them anyway. Strip them.
    for j in joints.values():
        if j.get("type") == "continuous" and j.find("limit") is not None:
            for attr in ("lower", "upper"):
                j.find("limit").attrib.pop(attr, None)

    movable = [n for n, j in joints.items() if j.get("type") in ("revolute", "continuous", "prismatic")]

    mimic: list[MimicJoint] = []
    for name in movable:
        m = joints[name].find("mimic")
        if m is None:
            continue
        leader = m.get("joint")
        if leader not in movable:
            raise ValueError(f"joint {name!r} mimics {leader!r}, which is not a movable joint")
        mimic.append(MimicJoint(name, leader, float(m.get("multiplier", 1.0)), float(m.get("offset", 0.0))))
        joints[name].remove(m)

    links = [l.get("name") for l in root.findall("link")]
    children = {j.find("child").get("link") for j in joints.values()}
    roots = [l for l in links if l not in children]
    if len(roots) != 1:
        raise ValueError(f"URDF must have exactly one root link, found {roots}")

    mass = sum(float(m.get("value")) for m in root.iter("mass"))
    effort = {n: float(joints[n].find("limit").get("effort", 0) or 0)
              for n in movable if joints[n].find("limit") is not None}

    mj = root.find("mujoco")
    if mj is None:
        mj = ET.Element("mujoco")
        root.insert(0, mj)
    compiler = mj.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(mj, "compiler")
    compiler.attrib.update(fusestatic="false", discardvisual="false", strippath="false")

    info = BodyInfo(cfg.name, roots[0], movable, mimic, list(cfg.lock_joints), mass,
                    mesh_files=mesh_files, effort=effort)

    for name in cfg.hold_joints:
        if name not in info.independent:
            raise ValueError(f"hold_joints: {name!r} is not an independent movable joint")
    info.holds = dict(cfg.hold_joints)
    for pattern, _, _ in cfg.actuators:
        if not any(re.fullmatch(pattern, j) for j in info.independent):
            raise ValueError(f"actuators: pattern {pattern!r} matches no joint")
    info.actuated = [j for j in info.independent if actuator_gains(cfg, j) is not None and j not in info.holds]
    return ET.tostring(root, encoding="unicode"), info


# ----------------------------------------------------------------------------------- spec


def build_body_spec(cfg: BodyConfig) -> tuple[mujoco.MjSpec, BodyInfo]:
    """The body as an MjSpec with all four fixes, motors, holds and damping applied."""
    text, info = prepare_urdf(cfg)
    spec = mujoco.MjSpec.from_string(text)

    for mesh in spec.meshes:
        real = info.mesh_files.get(Path(mesh.file).name)
        if real is None:
            raise RuntimeError(f"mesh {mesh.name!r}: no real file recorded for {mesh.file!r}")
        mesh.file = real.as_posix()

    # Fix 3 + 4. The importer marks visual geoms contype = conaffinity = 0.
    for g in spec.geoms:
        if g.contype or g.conaffinity:
            g.group = COLLISION_GROUP
            g.contype, g.conaffinity = ROBOT_CONTYPE, ROBOT_CONAFFINITY
            if cfg.hand_friction is not None:
                g.friction = [float(cfg.hand_friction), 0.005, 0.0001]
            g.condim = 4

    for b in spec.bodies:
        if b.name != "world":
            b.gravcomp = 1.0

    # Fix 1. MuJoCo: q_f - ref_f = a0 + a1 (q_l - ref_l). URDF: q_f = offset + k q_l.
    for m in info.mimic:
        ref_f, ref_l = spec.joint(m.follower).ref, spec.joint(m.leader).ref
        eq = spec.add_equality(name=f"mimic_{m.follower}", type=mujoco.mjtEq.mjEQ_JOINT)
        eq.name1, eq.name2 = m.follower, m.leader
        data = np.zeros(len(eq.data))
        data[0], data[1] = m.offset + m.multiplier * ref_l - ref_f, m.multiplier
        eq.data = data

    for j, c in cfg.joint_damping.items():
        jt = spec.joint(j)
        if jt is None:
            raise ValueError(f"joint_damping: no joint {j!r} in the URDF")
        d = np.zeros(len(jt.damping))
        d[0] = c
        jt.damping = d

    for j, q in info.holds.items():
        eq = spec.add_equality(name=f"hold_{j}", type=mujoco.mjtEq.mjEQ_JOINT)
        eq.name1 = j
        data = np.zeros(len(eq.data))
        data[0] = q - spec.joint(j).ref
        eq.data = data

    rated = read_rated_torques(Path(cfg.torque_limits)) if cfg.torque_limits else {}
    for j in info.actuated:
        kp, kv = actuator_gains(cfg, j)
        limit = rated.get(j) or info.effort.get(j)
        act = spec.add_actuator(name=f"motor_{j}", target=j, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        act.set_to_position(kp=kp, kv=kv)
        if limit:
            act.forcelimited = True
            act.forcerange = [-limit, limit]
        if spec.joint(j).limited:
            lo, hi = spec.joint(j).range
            act.ctrllimited = True
            act.ctrlrange = [lo, hi]
    return spec, info


def attach_body(scene: mujoco.MjSpec, cfg: BodyConfig) -> BodyInfo:
    """Bolt the body into a scene at cfg.position / cfg.yaw_deg, names prefixed 'robot/'.

    The root is attached through a frame with no joint between it and the world: a robot
    bolted down. That IS the fixed base of the standing phase.
    """
    body, info = build_body_spec(cfg)
    frame = scene.worldbody.add_frame(pos=list(cfg.position),
                                      quat=list(rpy_to_quat(0.0, 0.0, np.radians(cfg.yaw_deg))))
    scene.attach(body, prefix=ROBOT_PREFIX, frame=frame)
    scene.compiler.fusestatic = False
    return info


def add_mounted_camera(scene: mujoco.MjSpec, cam: CameraMount) -> None:
    """Put a MuJoCo camera on ``cam.mount``, looking out of the lens of that body's frame."""
    body = scene.body(cam.mount)
    if body is None:
        raise KeyError(f"camera {cam.name!r}: mount body {cam.mount!r} not found in the model")
    if cam.frame == "optical":
        R_link_to_mj = MJ_TO_CV             # MuJoCo camera looks down -z; optical z is forward
    elif cam.frame == "ros_link":
        R_link_to_mj = ROS_LINK_TO_MJ_CAMERA
    else:
        raise ValueError(f"camera {cam.name!r}: frame must be 'optical' or 'ros_link', got {cam.frame!r}")
    R_offset = quat_to_mat(rpy_to_quat(*np.radians(cam.mount_rpy_deg)))
    body.add_camera(name=cam.name, pos=list(cam.mount_pos),
                    quat=list(mat_to_quat(R_offset @ R_link_to_mj)), fovy=cam.fovy_deg)
