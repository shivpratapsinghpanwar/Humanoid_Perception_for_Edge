import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def sim():
    """One compiled standing robot for the whole session: compiling the meshes is the slow part."""
    from humanoid_perception.sim.stand import StandingSim

    s = StandingSim()
    yield s
    s.close()
