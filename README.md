# Humanoid Perception for Edge

A vision-language-action layer for a humanoid, built to run on a CPU: a camera and a
sentence in, a smooth stream of upper-body joint targets out. First on a standing robot in
MuJoCo, then on a real camera, then on a robot, and last on a Jetson.

The public body is the **Tiangong 2.0 Pro** from the Open X-Humanoid URDF release
(OpenAtom Open Hardware License 1.0, vendored with provenance under `assets/robots/`). Any
other robot with the same upper-body joint names plugs in through a `BodyConfig` — the
action contract resolves joints by name — without changing the code or the repository.

The design, the decisions and the milestones are in [`docs/PLAN.md`](docs/PLAN.md). Read
it first.

## Quick start

```powershell
$env:PYTHONIOENCODING = "utf-8"
uv sync                              # Python 3.12 environment from uv.lock

uv run scripts\stand.py              # the robot holds HOME for 60 s; its head camera renders the room
uv run scripts\stand.py --view       # the same, with the MuJoCo viewer open
uv run pytest                        # the M0 gate as tests
```

## Layout

| Path | What |
|---|---|
| `humanoid_perception/robot/contract.py` | **The one contract**: the 17-joint upper-body action stream, resolved by joint name |
| `humanoid_perception/robot/loader.py` | URDF → MuJoCo, done right (mimics, unfused links, hidden collision meshes, motors); `BodyConfig.from_yaml` for other bodies |
| `humanoid_perception/robot/camera.py` | The RGB-D camera: intrinsics, pose, metric depth |
| `humanoid_perception/sim/stand.py` | The standing robot: legs locked, pelvis bolted just above the floor, 50 Hz control clock |
| `assets/robots/tiangong2pro/` | The Tiangong 2.0 Pro, vendored with `PROVENANCE.md`, `LICENCE` and a hash manifest |
| `scripts/` | `stand.py` today; the census, the tracks and the evaluator follow per the plan |
| `tests/` | Every claim above, as a test |
| `docs/` | `PLAN.md` |

A gitignored `private/` folder can hold bodies and notes that are not public; nothing in
it is ever committed.

## Status

| milestone | state |
|---|---|
| M0 bootstrap | robot loads, holds HOME, renders from the head camera; tests green |
| M1 CPU latency census | next |
