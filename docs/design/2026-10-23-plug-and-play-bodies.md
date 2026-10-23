# 2026-10-23 · Design: plug-and-play humanoid bodies

Status: proposed, not implemented. Reference robot: Tiangong 2 Pro.

## Goal

Someone with a different humanoid should be able to fine-tune SmolVLA (and later TernaVLA) on
their robot by writing **one YAML file** next to their URDF, with no code changes:
generate demonstrations in simulation, export a LeRobot dataset, train, evaluate.

## What already works

| Piece | Where | Body-agnostic because |
|---|---|---|
| Action contract | `robot/contract.py` | 17 core joints (head 3, right arm 7, left arm 7) fixed in one order; every body is resolved against it by joint name and fails loudly if a core joint is missing. Waist and hands are optional extension channels. |
| Body description | `robot/loader.py`, `BodyConfig.from_yaml` | URDF path, mesh packages, locked joints, motor gains by regex, waist joint name, camera mount: all from YAML. |
| Offsets from home | contract | Angles are offsets from the body's home pose, so "all zeros" means home on every body. |
| Dataset export | demonstration pipeline\*, `to_lerobot.py --layout core` | Picks the 17 core joints by name from whatever the batch recorded. |
| Normalisation | `to_lerobot.py --std-floor`\* | A joint that never moves in the demos no longer turns sensor noise into huge z-scores. |
| Dataset check | `check_lerobot.py`\* | Checks the dataset against its own joint names and the 17-joint core, not a fixed width. |
| Training | SmolVLA fine-tune scripts\* | State/action width and camera keys are read from the dataset's `meta/info.json`. |

\* The demonstration pipeline (skills, export, training, evaluation) is not in this repository
yet; it builds on the contract and loader here.

## What blocks a different humanoid today

1. **Joint names must match the contract exactly.** `resolve_layout` looks up names such as
   `shoulder_pitch_r_joint`. The Tiangong 2 Pro uses these names; most other humanoids do not
   (for example `right_shoulder_pitch_joint`). Today the only way in is to rename joints in the
   URDF.
2. **Body facts are hard-coded in the demonstration skills.** Palm and fingertip link names,
   the wrist link used for hand checks, and the preferred arm posture for reaching (including
   its sign convention) live in Python constants, not in the body description.
3. **Velocity limits come only from the URDF.** Demonstrations are timed against URDF limits;
   when those are a template rather than the real motors, the demos are not safe to replay on
   hardware. There is no per-body override.
4. **A checkpoint does not record its body.** Nothing stops a policy trained on one body from
   being run on another with a different home pose or joint signs.
5. **Every script assumes one body path.** Generation, export, training and evaluation should
   all take `--body path/to/body.yaml`.

## Proposed changes

### 1. Joint-name aliases in the body YAML

```yaml
joint_names:                     # contract name -> this body's URDF name; unlisted = same name
  head_yaw_joint: neck_yaw_joint
  shoulder_pitch_r_joint: right_shoulder_pitch_joint
  # ...
```

The loader maps body names to contract names once, before `resolve_layout`. Everything
downstream (datasets, policies, logs) uses contract names only, so datasets from different
bodies share one vocabulary. A typo still fails loudly, naming the missing contract joint.

### 2. End effectors and posture in the body YAML

```yaml
end_effectors:
  right: {palm: right_palm_link, wrist: right_wrist_link, tips: [right_index_tip, right_thumb_tip]}
  left:  {palm: left_palm_link,  wrist: left_wrist_link,  tips: [left_index_tip,  left_thumb_tip]}
reach_posture: [-0.8, 0.0, 0.0, -1.5, 0.0, 0.0, 0.0]   # example; contract arm order, radians from home
```

The skills read these instead of constants. `reach_posture` is in contract arm order (shoulder
pitch, roll, yaw, elbow pitch, yaw, wrist roll, pitch), so its signs are the body's own.

### 3. Velocity caps

```yaml
velocity_caps: {default_fraction_of_urdf: 0.5, joints: {elbow_pitch_r_joint: 1.5}}   # rad/s
```

Demonstrations are timed against these caps, and the evaluator uses the same numbers as its
safety limit. The default stays the URDF limit, so existing bodies are unchanged until someone
sets it.

### 4. Body stamp in datasets and checkpoints

The export writes `meta/body.json`: body name, contract version, joint list in order, home pose
and the alias map. Training copies it into the checkpoint. The SmolVLA driver compares it with
the loaded body and refuses a mismatch, unless the caller explicitly asks for cross-body
transfer.

### 5. One `--body` flag everywhere

`generate`, `to_lerobot`, `check_lerobot`, the training scripts and `evaluate` all take
`--body`. The default stays the reference robot.

## Bringing a new humanoid, once this lands

1. Put the URDF and meshes in a folder and write `my_robot.yaml` (`BodyConfig` keys plus
   `joint_names`, `end_effectors`, `reach_posture`, `velocity_caps`, `camera`).
2. `stand.py --body my_robot.yaml`: the robot loads, stands and resolves the 17-joint contract.
3. `generate.py --body my_robot.yaml`: scripted demonstrations in simulation.
4. `to_lerobot.py --layout core --std-floor 0.02`, then `check_lerobot.py`.
5. Fine-tune SmolVLA on one GPU (about 7 h and ₹200–350 for 20k steps on an L4 spot VM).
6. `evaluate.py --body my_robot.yaml`: closed-loop success per skill.

## Deliberately out of scope

**Bodies without a 3-joint head or two 7-joint arms.** The contract stays strictly 17 core
joints: a body that lacks one fails at load time. Supporting such bodies (for example with
masked channels that the policy ignores) is a separate decision, because it changes the action
space every trained model shares.

## Acceptance test

A copy of the Tiangong 2 Pro URDF with every upper-body joint renamed must work through
steps 2–4 using only a YAML with `joint_names`, and produce a dataset identical (same names,
same values) to the one from the original URDF.
