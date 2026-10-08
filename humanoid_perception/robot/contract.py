"""The action contract: which joints the perception layer commands, in what order.

WHY A CONTRACT AND NOT "WHATEVER THE URDF LISTS"
------------------------------------------------
More than one body can carry the same upper body: the public Tiangong 2 Pro this
repository ships, and any private robot that plugs in through a ``BodyConfig``. Two
URDFs can use the same joint names and still list them in a different order (the left
wrist pair is the classic case). If the action vector's layout were taken from the file, a
model trained on one body would move the wrong wrist on the other and nobody would see
an error.

So the layout is fixed HERE, once, symmetric left/right, and every body is resolved
against it BY NAME. A body that lacks a core joint fails loudly at load time.

The 17 core joints are head (3) + right arm (7) + left arm (7). Two extension channels
ride beside them when the body has them AND they are motorised: the waist (1, whatever the
body calls it) and the hands (12: thumb 2 + index/middle/ring/little 1 per hand; every
other finger joint is a mimic follower and is never commanded).

Angles in the stream are OFFSETS FROM HOME in radians, so "all zeros" always means "the
home posture", whatever body is loaded. HOME is the body's URDF zero unless the loader
says otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

HEAD_JOINTS: tuple[str, ...] = (
    "head_yaw_joint",
    "head_pitch_joint",
    "head_roll_joint",
)

# One arm, proximal to distal. Roll before pitch at the wrist on BOTH sides, whatever
# order a particular URDF happens to list them in.
_ARM_TEMPLATE: tuple[str, ...] = (
    "shoulder_pitch_{s}_joint",
    "shoulder_roll_{s}_joint",
    "shoulder_yaw_{s}_joint",
    "elbow_pitch_{s}_joint",
    "elbow_yaw_{s}_joint",
    "wrist_roll_{s}_joint",
    "wrist_pitch_{s}_joint",
)
RIGHT_ARM_JOINTS: tuple[str, ...] = tuple(j.format(s="r") for j in _ARM_TEMPLATE)
LEFT_ARM_JOINTS: tuple[str, ...] = tuple(j.format(s="l") for j in _ARM_TEMPLATE)

#: The 17-number core stream. This order is the contract.
UPPER_BODY_JOINTS: tuple[str, ...] = HEAD_JOINTS + RIGHT_ARM_JOINTS + LEFT_ARM_JOINTS
assert len(UPPER_BODY_JOINTS) == 17

#: Extension channel: the hands, 6 independent joints per hand, right then left.
_HAND_TEMPLATE: tuple[str, ...] = (
    "{side}_thumb_1_joint",
    "{side}_thumb_2_joint",
    "{side}_index_1_joint",
    "{side}_middle_1_joint",
    "{side}_ring_1_joint",
    "{side}_little_1_joint",
)
RIGHT_HAND_JOINTS: tuple[str, ...] = tuple(j.format(side="right") for j in _HAND_TEMPLATE)
LEFT_HAND_JOINTS: tuple[str, ...] = tuple(j.format(side="left") for j in _HAND_TEMPLATE)
HAND_JOINTS: tuple[str, ...] = RIGHT_HAND_JOINTS + LEFT_HAND_JOINTS
assert len(HAND_JOINTS) == 12


@dataclass(frozen=True)
class JointLayout:
    """The joints a loaded body exposes to the action stream, in contract order.

    ``core`` is always the 17 upper-body joints. ``waist`` and ``hands`` are present only
    when the body has them motorised (empty tuple otherwise), so ``layout.hands`` is a
    well-defined answer for every body.
    """

    body: str
    core: tuple[str, ...]
    waist: tuple[str, ...]
    hands: tuple[str, ...]

    @property
    def all(self) -> tuple[str, ...]:
        """Every commanded joint: core, then waist, then hands."""
        return self.core + self.waist + self.hands

    @property
    def size(self) -> int:
        return len(self.all)

    def index(self, joint: str) -> int:
        return self.all.index(joint)


def resolve_layout(body: str, commandable_joints: Iterable[str], waist_joint: str | None = None) -> JointLayout:
    """Match the contract against the joints a body can actually be commanded on.

    Fails loudly if any of the 17 core joints is missing: a body without them cannot take
    the stream. ``waist_joint`` is the body's own name for its waist (None = no waist
    channel); the hands appear only when every joint of the channel is commandable.
    """
    have = set(commandable_joints)
    missing = [j for j in UPPER_BODY_JOINTS if j not in have]
    if missing:
        raise ValueError(
            f"body {body!r} cannot be commanded on core contract joints {missing}; "
            f"the 17-joint stream cannot be applied"
        )
    waist = (waist_joint,) if waist_joint and waist_joint in have else ()
    hands = HAND_JOINTS if all(j in have for j in HAND_JOINTS) else ()
    return JointLayout(body=body, core=UPPER_BODY_JOINTS, waist=waist, hands=hands)
