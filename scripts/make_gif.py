"""Render a short GIF of the standing robot: an outside view beside its own head camera.

    uv run scripts/make_gif.py                          # docs/media/m0_stand.gif, 6 s at 8 fps, 2 x 288x216
    uv run scripts/make_gif.py --body private/bodies/x.yaml --out output/x.gif

The robot performs a small scripted motion through the action stream (a nod, a glance to
the right, the right arm raised) so the clip shows both the body moving and what its camera
sees while it moves. Frames are rendered off-screen with mujoco.Renderer and written with
Pillow; the output is kept small enough to live in the repository.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from humanoid_perception.robot.loader import BodyConfig  # noqa: E402
from humanoid_perception.sim.stand import StandingSim, WorldConfig  # noqa: E402


def scripted_targets(sim: StandingSim, t: float) -> np.ndarray:
    """Offsets from HOME at time t: nod (0-2 s), look right (2-4 s), raise the right arm (3-6 s)."""
    q = np.zeros(sim.layout.size)
    L = sim.layout
    if t < 2.0:
        q[L.index("head_pitch_joint")] = 0.25 * np.sin(np.pi * t)            # nod, within the +-25 deg range
    elif t < 4.0:
        q[L.index("head_yaw_joint")] = -0.6 * np.sin(0.5 * np.pi * (t - 2.0))  # glance right and back
    if t > 3.0:
        s = min(1.0, (t - 3.0) / 1.5)
        s = 0.5 - 0.5 * np.cos(np.pi * s)                                      # ease in
        q[L.index("shoulder_pitch_r_joint")] = -1.2 * s                        # arm forward and up
        q[L.index("elbow_pitch_r_joint")] = -0.8 * s
    # clamp to the joint ranges, offsets relative to a zero HOME
    return np.clip(q, sim.joint_range[:, 0], sim.joint_range[:, 1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--body", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("docs/media/m0_stand.gif"))
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--size", type=int, nargs=2, default=(288, 216), metavar=("W", "H"))
    ap.add_argument("--colors", type=int, default=96, help="palette size; fewer colours, smaller file")
    args = ap.parse_args()

    from PIL import Image

    body = BodyConfig.from_yaml(args.body) if args.body else BodyConfig()
    sim = StandingSim(WorldConfig(body=body))
    W, H = args.size

    # Outside view: a free camera looking at the upper body from the front-right.
    ext = mujoco.Renderer(sim.model, height=H, width=W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    mount = sim.cfg.body.camera.mount.removeprefix("robot/")      # the head camera's body, any body
    head = sim.body_position(mount)
    cam.lookat[:] = [head[0], head[1], head[2] - 0.35]
    cam.distance, cam.azimuth, cam.elevation = 2.6, 150.0, -12.0
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]

    head_cam = mujoco.Renderer(sim.model, height=H, width=W)
    head_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_CAMERA, sim.cfg.body.camera.name)

    frames = []
    n_frames = int(round(args.seconds * args.fps))
    ctrl_per_frame = max(1, int(round(50 / args.fps)))
    for k in range(n_frames):
        t = k / args.fps
        sim.set_targets(scripted_targets(sim, t))
        for _ in range(ctrl_per_frame):
            sim.control_step()
        mujoco.mj_forward(sim.model, sim.data)
        ext.update_scene(sim.data, camera=cam, scene_option=opt)
        a = ext.render()
        head_cam.update_scene(sim.data, camera=head_id, scene_option=opt)
        b = head_cam.render()
        im = Image.fromarray(np.concatenate([a, b], axis=1))
        # A fixed small palette without dithering: dither noise is what makes a GIF of a
        # shaded scene large, and banding on a floor is a fair price for a file that fits.
        frames.append(im.quantize(colors=args.colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=int(1000 / args.fps),
                   loop=0, optimize=True)
    ext.close(); head_cam.close(); sim.close()
    print(f"{args.out}: {len(frames)} frames, {W * 2}x{H}, {args.out.stat().st_size / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
