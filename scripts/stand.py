"""M0 gate: the robot holds HOME, and its head camera sees the room.

    uv run scripts/stand.py                 # hold 60 s, report drift, save head_cam.png / head_cam_depth.png
    uv run scripts/stand.py --seconds 10
    uv run scripts/stand.py --view          # also open the MuJoCo viewer (close it to finish)
    uv run scripts/stand.py --body private/bodies/mobile_manipulator.yaml   # a private body instead

Prints the worst joint drift from HOME over the run in degrees, the per-step cost, and
where the images went. The gate in docs/PLAN.md is drift < 1 deg over 60 s.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from humanoid_perception.robot.loader import BodyConfig  # noqa: E402
from humanoid_perception.sim.stand import StandingSim, WorldConfig  # noqa: E402


def save_images(frame, out: Path) -> None:
    from PIL import Image

    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame.rgb).save(out / "head_cam.png")
    d = frame.depth
    shown = np.zeros_like(d, dtype=np.uint8)
    valid = d > 0
    if valid.any():
        lo, hi = d[valid].min(), d[valid].max()
        shown[valid] = (255 * (1 - (d[valid] - lo) / max(hi - lo, 1e-6))).astype(np.uint8)
    Image.fromarray(shown).save(out / "head_cam_depth.png")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--out", type=Path, default=Path("output/stand"))
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--body", type=Path, default=None, help="a BodyConfig YAML; default: the public Tiangong 2 Pro")
    args = ap.parse_args()

    body = BodyConfig.from_yaml(args.body) if args.body else BodyConfig()
    sim = StandingSim(WorldConfig(body=body))
    print(f"body {sim.info.name}: {len(sim.info.joints)} movable joints, {len(sim.info.mimic)} mimic couplings, "
          f"{len(sim.info.actuated)} motors; stream size {sim.layout.size} "
          f"(core 17 + waist {len(sim.layout.waist)} + hands {len(sim.layout.hands)})")
    print(f"URDF mass {sim.info.urdf_mass:.2f} kg, model mass {sim.model.body_subtreemass[0]:.2f} kg")

    worst = 0.0
    n_ctrl = int(round(args.seconds * 50))
    t0 = time.perf_counter()
    for _ in range(n_ctrl):
        sim.control_step()
        worst = max(worst, float(np.abs(sim.joint_offsets()).max()))
    wall = time.perf_counter() - t0
    per_step_ms = 1000 * wall / max(n_ctrl * sim.steps_per_control, 1)
    print(f"held HOME for {sim.time:.1f} s sim: worst drift {np.degrees(worst):.3f} deg "
          f"({'PASS' if np.degrees(worst) < 1.0 else 'FAIL'} at 1 deg); "
          f"{per_step_ms:.3f} ms per physics step, {sim.time / wall:.1f}x real time")

    frame = sim.render()
    k = frame.intrinsics
    valid = frame.valid
    print(f"head_cam {k.width}x{k.height}, fx {k.fx:.2f}, cx {k.cx:.1f}; depth valid {100 * valid.mean():.1f}% "
          f"range {frame.depth[valid].min():.2f}-{frame.depth[valid].max():.2f} m" if valid.any() else
          f"head_cam {k.width}x{k.height}: no valid depth")
    save_images(frame, args.out)
    print(f"images -> {args.out / 'head_cam.png'}, {args.out / 'head_cam_depth.png'}")

    if args.view:
        import mujoco.viewer

        with mujoco.viewer.launch_passive(sim.model, sim.data) as v:
            while v.is_running():
                sim.control_step()
                v.sync()
    sim.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
