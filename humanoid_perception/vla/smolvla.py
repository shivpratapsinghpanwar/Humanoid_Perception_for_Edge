"""SmolVLA as a driver of the standing robot.

``lerobot/smolvla_base`` was pretrained on small 6-joint arms (SO-100 style: shoulder pan,
shoulder lift, elbow flex, wrist flex, wrist roll, gripper) seen by three cameras. It knows
nothing about a humanoid. Until a checkpoint is fine-tuned on this robot, driving the robot
with it is a PLUMBING TEST: it proves that camera frames go in, a chunk of actions comes
out, and the robot executes it safely. The motion itself means nothing.

The mapping is therefore explicit and conservative: each model output dimension drives one
contract joint, as an offset from HOME of at most ``amplitude * clip`` radians, taken from
the model's NORMALISED action space (zero mean, unit spread) so the size of the motion does
not depend on another robot's units.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

REPO = "lerobot/smolvla_base"

# SO-100 order of the base checkpoint -> the closest joint of the right arm. The gripper
# dimension has no arm joint and is ignored.
BASE_ARM_MAP: tuple[str | None, ...] = (
    "shoulder_yaw_r_joint",     # shoulder_pan
    "shoulder_pitch_r_joint",   # shoulder_lift
    "elbow_pitch_r_joint",      # elbow_flex
    "wrist_pitch_r_joint",      # wrist_flex
    "wrist_roll_r_joint",       # wrist_roll
    None,                       # gripper
)


def load_smolvla(repo: str = REPO):
    """Policy and its pre/post-processors on the CPU, in fp32.

    The checkpoint loads in bfloat16; a CPU without bf16 arithmetic (AVX2 laptops)
    emulates it at ~8x the cost, so the policy is cast to fp32 and checked."""
    import torch
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    from ..bench.checkpoints import fetch

    path = str(fetch(repo))
    policy = SmolVLAPolicy.from_pretrained(path)
    policy.config.device = "cpu"
    policy.to(device="cpu", dtype=torch.float32).eval()
    dtypes = sorted({str(p.dtype) for p in policy.parameters()})
    if dtypes != ["torch.float32"]:
        raise RuntimeError(f"expected fp32 weights after the cast, found {dtypes}")
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=path,
        preprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    return policy, pre, post


@dataclass
class ChunkInfo:
    inference_s: float
    normalised_min: float
    normalised_max: float
    shape: tuple[int, ...]


@dataclass
class SmolVLADriver:
    """Turns (head-camera frame, joint offsets, instruction) into a chunk of contract targets."""

    layout_joints: tuple[str, ...]
    instruction: str
    joint_map: tuple[str | None, ...] = BASE_ARM_MAP
    amplitude: float = 0.15               # rad per unit of normalised action
    clip: float = 2.0                     # normalised actions are clipped to +-clip
    threads: int = 4
    _policy: object = field(default=None, init=False, repr=False)

    def __post_init__(self):
        import torch

        torch.set_num_threads(self.threads)
        self._policy, self._pre, self._post = load_smolvla()
        cfg = self._policy.config
        self.image_keys: list[str] = []
        self.state_key, self.state_dim = None, 0
        for key, feat in cfg.input_features.items():
            kind = str(getattr(getattr(feat, "type", None), "name", feat)).upper()
            if "VISUAL" in kind:
                self.image_keys.append(key)
            elif "STATE" in kind:
                self.state_key, self.state_dim = key, int(np.prod(feat.shape))
        self.chunk_size = int(cfg.chunk_size)
        self._targets = [None if j is None else self.layout_joints.index(j) for j in self.joint_map]

    @property
    def driven_joints(self) -> list[str]:
        return [j for j in self.joint_map if j is not None]

    def chunk(self, rgb: np.ndarray, offsets: np.ndarray) -> tuple[np.ndarray, ChunkInfo]:
        """(H, W, 3) uint8 frame + current offsets (layout order) -> (chunk_size, n) offsets."""
        import torch

        img = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)
        obs: dict = {k: img.clone() for k in self.image_keys}
        if self.state_key is not None:
            state = np.zeros(self.state_dim, dtype=np.float32)
            for i, idx in enumerate(self._targets[: self.state_dim]):
                if idx is not None:
                    state[i] = np.degrees(offsets[idx])
            obs[self.state_key] = torch.from_numpy(state).unsqueeze(0)
        obs["task"] = self.instruction

        t0 = time.perf_counter()
        with torch.inference_mode():
            batch = self._pre(obs)
            actions = self._policy.predict_action_chunk(batch)      # normalised space
        dt = time.perf_counter() - t0
        a = actions[0].float().cpu().numpy()                        # (chunk, action_dim)

        # Joints the model does not drive hold HOME (zero offset). Re-commanding the MEASURED
        # angle instead would let gravity sag compound chunk after chunk.
        out = np.zeros((a.shape[0], len(self.layout_joints)))
        driven = np.clip(a, -self.clip, self.clip) * self.amplitude
        for i, idx in enumerate(self._targets[: a.shape[1]]):
            if idx is not None:
                out[:, idx] = driven[:, i]
        return out, ChunkInfo(inference_s=dt, normalised_min=float(a.min()), normalised_max=float(a.max()),
                              shape=tuple(a.shape))
