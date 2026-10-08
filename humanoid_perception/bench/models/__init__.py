"""The models under test, each as a ``ModelSpec`` with a ``run`` that returns a CensusResult.

Heavy imports (torch, lerobot, transformers) happen inside each ``run``, so listing the
models costs nothing and a missing optional extra fails only for the row that needs it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..results import CensusResult


@dataclass(frozen=True)
class ModelSpec:
    name: str
    role: str
    source: str
    licence: str
    run: Callable[..., CensusResult]     # run(n, warmup, threads, precision) -> CensusResult


def _smolvla(**kw) -> CensusResult:
    from .smolvla import run
    return run(**kw)


MODELS: dict[str, ModelSpec] = {
    "smolvla_base": ModelSpec(
        name="smolvla_base", role="vla", source="lerobot/smolvla_base", licence="Apache-2.0", run=_smolvla),
}
