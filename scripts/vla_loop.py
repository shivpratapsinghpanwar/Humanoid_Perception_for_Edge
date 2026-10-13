"""Closed loop: a VLA drives the standing robot in simulation, one action chunk at a time.

    uv run scripts/vla_loop.py                                   # public Tiangong 2 Pro
    uv run scripts/vla_loop.py --body private/bodies/x.yaml      # a private body (output stays local)

Each cycle: render the head camera -> the policy computes a chunk (physics is PAUSED while it
thinks, so a slow CPU changes the wall-clock time, never the robot's behaviour) -> the robot
executes the chunk at 50 Hz through the safety filter -> repeat.

With ``lerobot/smolvla_base`` this is a PLUMBING TEST (see humanoid_perception/vla/smolvla.py):
the checkpoint was trained on 6-joint arms, so its outputs drive five right-arm joints by
small, clamped offsets and mean nothing yet. What it proves: frames in, chunks out, safe
execution, and the real per-chunk cost on this CPU.

Writes <out>/run.json (per-chunk timing and filter counts) and <out>/vla_loop.gif
(outside view | head camera, with a frame per chunk saying how long the model thought).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from humanoid_perception.bench.machine import PowerWatch, capture  # noqa: E402
from humanoid_perception.robot.loader import BodyConfig  # noqa: E402
from humanoid_perception.sim.stand import CONTROL_HZ, StandingSim, TableConfig, WorldConfig  # noqa: E402
from humanoid_perception.vla.safety import SafetyFilter  # noqa: E402
from humanoid_perception.vla.smolvla import SmolVLADriver  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--body", type=Path, default=None, help="BodyConfig YAML; default: the public Tiangong 2 Pro")
    ap.add_argument("--instruction", default="look at the red cup")
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--execute", type=int, default=50, help="control steps executed per chunk (<= chunk size)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--amplitude", type=float, default=0.15, help="rad per unit of normalised action")
    ap.add_argument("--max-offset", type=float, default=0.35, help="envelope around HOME, rad")
    ap.add_argument("--max-rate", type=float, default=1.0, help="rad/s")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--size", type=int, nargs=2, default=(320, 240), metavar=("W", "H"))
    ap.add_argument("--colors", type=int, default=96)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    from PIL import Image, ImageDraw

    body = BodyConfig.from_yaml(args.body) if args.body else BodyConfig()
    out = args.out or Path("output/vla_loop") / body.name
    out.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()
    sim = StandingSim(WorldConfig(body=body, table=TableConfig()))
    n = sim.layout.size
    driver = SmolVLADriver(sim.layout.all, args.instruction, amplitude=args.amplitude, threads=args.threads)
    load_s = time.perf_counter() - t_start
    safety = SafetyFilter(sim.joint_range, max_rate=args.max_rate, control_hz=CONTROL_HZ,
                          max_offset=np.full(n, args.max_offset))
    print(f"loaded body {body.name!r} ({n} commanded joints) and SmolVLA in {load_s:.0f} s; "
          f"driving {driver.driven_joints}", flush=True)

    W, H = args.size
    ext = mujoco.Renderer(sim.model, height=H, width=W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    head = sim.body_position(sim.cfg.body.camera.mount.removeprefix("robot/"))
    cam.lookat[:] = [head[0] + 0.25, head[1], head[2] - 0.45]
    cam.distance, cam.azimuth, cam.elevation = 2.4, 145.0, -18.0
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]
    head_r = mujoco.Renderer(sim.model, height=H, width=W)
    head_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_CAMERA, sim.cfg.body.camera.name)

    def views() -> np.ndarray:
        mujoco.mj_forward(sim.model, sim.data)
        ext.update_scene(sim.data, camera=cam, scene_option=opt)
        head_r.update_scene(sim.data, camera=head_id, scene_option=opt)
        return np.concatenate([ext.render(), head_r.render()], axis=1)

    def to_gif(rgb: np.ndarray, text: str | None = None) -> Image.Image:
        im = Image.fromarray(rgb)
        if text:
            d = ImageDraw.Draw(im)
            d.rectangle([0, 0, im.width, 22], fill=(0, 0, 0))
            d.text((6, 5), text, fill=(255, 230, 80))
        return im.quantize(colors=args.colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)

    frames, durations, log = [], [], []
    ctrl_per_frame = max(1, int(round(CONTROL_HZ / args.fps)))
    driven_idx = [sim.layout.index(j) for j in driver.driven_joints]
    cmd = np.zeros(n)
    sim.set_targets(cmd)
    sim.run(0.5)                                     # settle at HOME
    state0 = capture()

    with PowerWatch() as power:
        for c in range(args.chunks):
            frame = sim.render()                     # what the policy sees: 640x480 head camera
            chunk, info = driver.chunk(frame.rgb, sim.joint_offsets())
            frames.append(to_gif(views(), f"chunk {c + 1}: SmolVLA thought {info.inference_s:.1f} s "
                                          f"(physics paused) - \"{args.instruction}\""))
            durations.append(1500)
            before = safety.clamped, safety.rate_limited
            steps = min(args.execute, chunk.shape[0])
            peak = 0.0
            for k in range(steps):
                cmd = safety(chunk[k], cmd)
                sim.set_targets(cmd)
                sim.control_step()
                peak = max(peak, float(np.abs(cmd[driven_idx]).max()))
                if k % ctrl_per_frame == 0:
                    frames.append(to_gif(views(), f"chunk {c + 1}: executing at 50 Hz, t = {sim.time:.2f} s"))
                    durations.append(int(1000 / args.fps))
            measured = sim.joint_offsets()[driven_idx]
            log.append(dict(chunk=c + 1, inference_s=round(info.inference_s, 2), action_shape=list(info.shape),
                            normalised_range=[round(info.normalised_min, 3), round(info.normalised_max, 3)],
                            executed_steps=steps, peak_command_rad=round(peak, 3),
                            measured_offsets_rad=[round(float(x), 3) for x in measured],
                            clamped=safety.clamped - before[0], rate_limited=safety.rate_limited - before[1]))
            print(f"chunk {c + 1}/{args.chunks}: thought {info.inference_s:.1f} s, normalised actions "
                  f"[{info.normalised_min:+.2f}, {info.normalised_max:+.2f}], peak command {peak:.3f} rad, "
                  f"clamped {log[-1]['clamped']}, rate-limited {log[-1]['rate_limited']}", flush=True)

    wall = time.perf_counter() - t_start
    gif = out / "vla_loop.gif"
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
    state1 = capture()
    summary = dict(
        body=body.name, instruction=args.instruction, policy="lerobot/smolvla_base (fp32, plumbing test)",
        driven_joints=driver.driven_joints, amplitude_rad=args.amplitude, max_offset_rad=args.max_offset,
        max_rate_rad_s=args.max_rate, chunks=log, sim_seconds=round(sim.time, 2), wall_seconds=round(wall, 1),
        load_seconds=round(load_s, 1),
        mean_inference_s=round(float(np.mean([r["inference_s"] for r in log])), 2),
        power=dict(start_on_ac=state0.on_ac_power, end_on_ac=state1.on_ac_power,
                   samples=power.samples, battery_samples=power.battery_samples),
        gif=str(gif))
    (out / "run.json").write_text(json.dumps(summary, indent=2))
    ext.close(); head_r.close(); sim.close()
    print(f"done: {args.chunks} chunks, sim {sim.time:.1f} s in {wall:.0f} s wall; "
          f"{gif} ({gif.stat().st_size / 1e6:.2f} MB); log {out / 'run.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
