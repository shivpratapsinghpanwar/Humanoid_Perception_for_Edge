"""The last gate between any policy and the motors.

Whatever a model outputs, the robot only ever receives targets that are
  1. inside an envelope of +-``max_offset`` rad around HOME (per joint, optional),
  2. inside the joint's own URDF range, and
  3. at most ``max_rate`` rad/s away from the previous command (a rate limit, so one wild
     chunk cannot fling an arm).
Every clamp is counted, so a log shows how hard the filter had to work.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SafetyFilter:
    joint_range: np.ndarray              # (n, 2) radians, in layout order
    max_rate: float = 1.0                # rad/s
    control_hz: float = 50.0
    max_offset: np.ndarray | None = None  # (n,) radians; None = only the joint range
    clamped: int = field(default=0, init=False)
    rate_limited: int = field(default=0, init=False)

    def __call__(self, target: np.ndarray, previous: np.ndarray) -> np.ndarray:
        t = np.asarray(target, dtype=np.float64)
        lo, hi = self.joint_range[:, 0], self.joint_range[:, 1]
        if self.max_offset is not None:
            lo = np.maximum(lo, -self.max_offset)
            hi = np.minimum(hi, self.max_offset)
        c = np.clip(t, lo, hi)
        self.clamped += int(np.count_nonzero(c != t))
        step = self.max_rate / self.control_hz
        r = np.clip(c, previous - step, previous + step)
        self.rate_limited += int(np.count_nonzero(r != c))
        return r
