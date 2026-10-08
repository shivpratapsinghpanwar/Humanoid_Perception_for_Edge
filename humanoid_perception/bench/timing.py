"""Timing and memory for one model under test.

``measure`` calls ``fn`` ``warmup`` times unrecorded (first calls pay for allocation, kernel
selection and page faults), then ``n`` times recorded, and reports p50 / p95 / mean / min /
max in milliseconds. ``peak_rss_mb`` reads the process's peak working set; each model is run
in its own subprocess by scripts/census.py so peaks do not contaminate each other.
"""

from __future__ import annotations

import os
import statistics
import time
from dataclasses import dataclass
from typing import Callable


@dataclass
class Timing:
    n: int
    warmup: int
    p50_ms: float
    p95_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float
    samples_ms: list[float]

    def to_dict(self) -> dict:
        return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in self.__dict__.items()
                if k != "samples_ms"} | {"samples_ms": [round(s, 2) for s in self.samples_ms]}


def measure(fn: Callable[[], object], n: int = 20, warmup: int = 3) -> Timing:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        samples.append(1000.0 * (time.perf_counter() - t0))
    s = sorted(samples)
    p95 = s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))]
    return Timing(n=n, warmup=warmup, p50_ms=statistics.median(s), p95_ms=p95,
                  mean_ms=statistics.fmean(s), min_ms=s[0], max_ms=s[-1], samples_ms=samples)


def peak_rss_mb() -> float | None:
    """Peak resident set of this process in MB (Windows: peak working set)."""
    try:
        import psutil
        info = psutil.Process(os.getpid()).memory_info()
        peak = getattr(info, "peak_wset", None) or getattr(info, "rss", None)
        return None if peak is None else peak / 1e6
    except Exception:
        return None
