"""The 17-joint contract is fixed, symmetric, and matches a body by name."""

import pytest

from humanoid_perception.robot import contract as c


def test_core_is_seventeen_in_the_documented_order():
    assert len(c.UPPER_BODY_JOINTS) == 17
    assert c.UPPER_BODY_JOINTS[:3] == ("head_yaw_joint", "head_pitch_joint", "head_roll_joint")
    assert c.UPPER_BODY_JOINTS[3] == "shoulder_pitch_r_joint"
    assert c.UPPER_BODY_JOINTS[10] == "shoulder_pitch_l_joint"
    # roll before pitch at BOTH wrists: the order is ours, not the URDF's
    assert c.UPPER_BODY_JOINTS[8:10] == ("wrist_roll_r_joint", "wrist_pitch_r_joint")
    assert c.UPPER_BODY_JOINTS[15:17] == ("wrist_roll_l_joint", "wrist_pitch_l_joint")


def test_left_and_right_arms_mirror_each_other():
    for r, l in zip(c.RIGHT_ARM_JOINTS, c.LEFT_ARM_JOINTS):
        assert r.replace("_r_", "_l_") == l


def test_layout_without_hands_or_waist():
    layout = c.resolve_layout("some_body", c.UPPER_BODY_JOINTS)
    assert layout.all == c.UPPER_BODY_JOINTS
    assert layout.waist == () and layout.hands == () and layout.size == 17


def test_layout_with_waist_and_hands_appends_in_order():
    have = c.UPPER_BODY_JOINTS + ("torso_joint",) + c.HAND_JOINTS
    layout = c.resolve_layout("some_body", have, waist_joint="torso_joint")
    assert layout.all == c.UPPER_BODY_JOINTS + ("torso_joint",) + c.HAND_JOINTS
    assert layout.size == 30
    assert layout.index("left_thumb_1_joint") == 18 + 6


def test_waist_channel_needs_the_named_joint_to_be_commandable():
    layout = c.resolve_layout("some_body", c.UPPER_BODY_JOINTS, waist_joint="body_yaw_joint")
    assert layout.waist == ()


def test_missing_core_joint_fails_loudly():
    have = [j for j in c.UPPER_BODY_JOINTS if j != "elbow_yaw_l_joint"]
    with pytest.raises(ValueError, match="elbow_yaw_l_joint"):
        c.resolve_layout("some_body", have)
