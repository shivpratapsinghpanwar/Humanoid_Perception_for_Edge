"""The census harness: statistics, machine state and the rendered document behave as stated."""

import json
import time

from humanoid_perception.bench import machine, results, timing


def test_measure_excludes_warmup_and_reports_percentiles():
    calls = []

    def fn():
        calls.append(1)
        time.sleep(0.002 if len(calls) > 2 else 0.05)   # the two warm-up calls are slow

    t = timing.measure(fn, n=10, warmup=2)
    assert len(calls) == 12 and t.n == 10 and t.warmup == 2
    assert t.p50_ms < 30 and t.max_ms < 30                # the slow warm-ups are not in the samples
    assert t.min_ms <= t.p50_ms <= t.p95_ms <= t.max_ms
    assert len(t.samples_ms) == 10


def test_machine_state_captures_power_and_clock():
    s = machine.capture()
    assert s.python and s.timestamp_utc.endswith("+00:00")
    assert s.on_ac_power in (True, False, None)
    d = s.to_dict()
    assert d["publishable"] == (s.on_ac_power is True)


def _fake_result(tmp_path, model="m", on_ac=True, p50=100.0):
    t = timing.Timing(n=5, warmup=1, p50_ms=p50, p95_ms=p50 * 1.2, mean_ms=p50, min_ms=p50 * 0.9,
                      max_ms=p50 * 1.3, samples_ms=[p50] * 5)
    s = machine.capture()
    s.on_ac_power = on_ac
    r = results.CensusResult(model=model, role="vla", source="org/" + model, revision="abc1234", licence="Apache-2.0",
                             runtime="torch-cpu", precision="fp32", threads=4,
                             what_is_timed="obs -> 50-action chunk", input_shape={"image": [480, 640]},
                             timing=t, peak_rss_mb=1234.5, machine=s)
    return r.save(tmp_path)


def test_results_round_trip_and_render(tmp_path):
    p1 = _fake_result(tmp_path, "alpha", on_ac=True, p50=250.0)
    p2 = _fake_result(tmp_path, "beta", on_ac=False, p50=900.0)
    assert p1.name == "alpha__torch-cpu__fp32__t4.json"
    loaded = results.load_all(tmp_path)
    assert [r["model"] for r in loaded] == ["alpha", "beta"]
    assert json.loads(p1.read_text())["machine"]["publishable"] is True
    md = results.render_markdown(loaded)
    main_table = md.split("## Runs taken on battery")[0]
    assert "| alpha |" in main_table and "**250**" in main_table
    assert "| beta |" not in main_table                    # battery runs never enter the main table
    assert "| beta |" in md                                # but they are listed
    assert "org/alpha" in md and "abc1234" in md
