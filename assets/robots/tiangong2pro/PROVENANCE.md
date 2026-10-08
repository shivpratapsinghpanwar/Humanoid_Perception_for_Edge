# Where these robot files came from

The Tiangong 2.0 Pro humanoid, as published by the Beijing Humanoid Robot Innovation Center
(Open X-Humanoid) in their URDF release. Nothing under this folder is edited by hand; it is
a copy of the upstream package, and `MANIFEST.sha256` records the hash of every file so a
test can prove it (`tests/test_assets_fresh.py`).

## Source

| | |
|---|---|
| Repository | https://github.com/Open-X-Humanoid/TienKung_URDF |
| Package | `tiangong2pro_urdf/` |
| Commit | `4c2c061d72dcdab60d74e45d3efb17246a3b6f62` (2026-09-21) |
| Copied | 2026-10-08 |
| Licence | **OpenAtom Open Hardware License, Version 1.0** — the repository's `licence` file, reproduced here as `LICENCE`. Redistribution with or without modification and commercial use are permitted; the licence and attribution notices must travel with the files and modifications must be marked. |

## What is here, and what is not

| copied | as | note |
|---|---|---|
| `urdf/tiangong2.0_pro_urdf.urdf` | `urdf/tiangong2.0_pro_urdf.urdf` | the body without hands: 30 revolute joints, the one the loader uses |
| `urdf/tiangong2.0_pro_with_hands.urdf` | `urdf/tiangong2.0_pro_with_hands.urdf` | the same body with two dexterous hands (24 finger joints); not loaded yet |
| `meshes/*.STL` (63 files, 116 MB) | `meshes/` | visual and collision geometry |
| `tiangong2pro_torq.xml` | same | the upstream MuJoCo motor-class file: torque ranges, damping and friction loss per class |
| `关键参数/` | `key_parameters/` | the upstream key-parameter table (CSV + PNG); folder renamed to ASCII, contents untouched |
| `package.xml` | same | ROS package manifest |
| not copied | — | `*.xacro`, `launch/`, `config/`, `script/`, `CMakeLists.txt`: ROS tooling this project does not use |

Everything about how the robot behaves in MuJoCo is decided at **load** time in
`humanoid_perception/robot/loader.py`, never in these files: static links kept, collision
meshes hidden from cameras, no self-collision, legs locked for the standing phase, motors,
the head camera.

## Facts about the model worth knowing

- Root link `pelvis`; 12 leg joints, `body_yaw_joint` (waist), 3 head joints, 7 joints per
  arm. Joint names follow the `<part>_<axis>_<l|r>_joint` pattern.
- URDF total mass 67.96 kg.
- `camera_head_link` is an **optical** frame (z out of the lens). At the URDF zero pose the
  lens points straight ahead in yaw and **30° below the horizontal**, 1.65 m above the floor
  when the robot stands on locked straight legs (measured in MuJoCo, 2026-10-08).
  `imu_link` sits on the pelvis; `left_tcp_link` / `right_tcp_link` are 8.48 cm below the
  wrist-roll links.
- No `<mimic>` joints in the no-hands URDF.
- Torque limits per joint are the URDF `<limit effort>` values (6.3 / 24 / 35 / 91 / 95 N·m
  on the upper body); the upstream MuJoCo file groups them into motor classes.

## Re-vendoring

Clone the repository above at a newer commit, copy the same files, regenerate
`MANIFEST.sha256`, update the commit hash here, and run the tests.
