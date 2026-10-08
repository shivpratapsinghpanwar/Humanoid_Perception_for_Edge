"""The standing robot holds its posture and obeys the action stream."""

import numpy as np
import pytest


def test_holds_home_without_drift(sim):
    sim.set_targets(np.zeros(sim.layout.size))
    worst = 0.0
    for _ in range(50 * 10):          # 10 s at the 50 Hz control rate
        sim.control_step()
        worst = max(worst, float(np.abs(sim.joint_offsets()).max()))
    assert np.degrees(worst) < 1.0


@pytest.mark.slow
def test_holds_home_for_a_minute(sim):
    sim.set_targets(np.zeros(sim.layout.size))
    worst = 0.0
    for _ in range(50 * 60):
        sim.control_step()
        worst = max(worst, float(np.abs(sim.joint_offsets()).max()))
    assert np.degrees(worst) < 1.0


def test_targets_are_tracked_in_contract_order(sim):
    """Command one joint through the stream and only that joint moves (plus its mimics)."""
    target = np.zeros(sim.layout.size)
    i = sim.layout.index("head_pitch_joint")
    target[i] = 0.2
    sim.set_targets(target)
    for _ in range(50 * 3):
        sim.control_step()
    q = sim.joint_offsets()
    assert q[i] == pytest.approx(0.2, abs=np.radians(1.0))
    others = np.delete(q, i)
    assert np.degrees(np.abs(others).max()) < 1.0
    sim.set_targets(np.zeros(sim.layout.size))
    for _ in range(50 * 3):
        sim.control_step()


def test_wrong_stream_size_is_refused(sim):
    with pytest.raises(ValueError):
        sim.set_targets(np.zeros(17 if sim.layout.size != 17 else 16))
