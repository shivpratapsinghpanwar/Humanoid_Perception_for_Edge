"""One census result = one JSON file; docs/BENCHMARKS.md is rendered from the set of them.

File name: ``<model>__<runtime>__<precision>__t<threads>.json`` under benchmarks/census/ (a
tracked folder, unlike output/). The files are committed: they are the evidence behind every
number in the document.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .machine import MachineState
from .timing import Timing

RESULTS_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "census"


@dataclass
class CensusResult:
    model: str                 # short id, e.g. "smolvla_base"
    role: str                  # "vla" | "vla_backbone" | "grounder"
    source: str                # HF repo id or URL
    revision: str | None       # HF commit hash when known
    licence: str
    runtime: str               # "torch-cpu" | "openvino" | "vla.cpp" | ...
    precision: str             # "fp32" | "int8" | "q8_0" | ...
    threads: int | None
    what_is_timed: str         # the exact quantity, in words
    input_shape: dict          # image size, state dim, instruction, ...
    timing: Timing
    peak_rss_mb: float | None
    machine: MachineState
    extra: dict = field(default_factory=dict)

    @property
    def filename(self) -> str:
        return f"{self.model}__{self.runtime}__{self.precision}__t{self.threads}.json"

    def save(self, directory: Path = RESULTS_DIR) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        d = asdict(self)
        d["timing"] = self.timing.to_dict()
        d["machine"] = self.machine.to_dict()
        path = directory / self.filename
        path.write_text(json.dumps(d, indent=2), encoding="utf-8", newline="\n")
        return path


def load_all(directory: Path = RESULTS_DIR) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(directory.glob("*.json"))]


def render_markdown(results: list[dict]) -> str:
    """docs/BENCHMARKS.md from the JSON results. Unpublishable rows (on battery) are listed
    separately and never in the main table."""
    pub = [r for r in results if r["machine"].get("publishable")]
    unpub = [r for r in results if not r["machine"].get("publishable")]
    lines = ["# CPU latency census", ""]
    lines += ["Every row below was produced by `scripts/census.py` and read back from",
              "`benchmarks/census/*.json`; nothing here is typed by hand. The quantity timed is named",
              "per row. SmolVLA's actions mean nothing on our body — its row is a latency probe.", ""]
    if pub:
        m = pub[0]["machine"]
        lines += ["## Machine", "",
                  f"- {m.get('cpu') or 'CPU'} — {m.get('cores_physical')} cores / {m.get('cores_logical')} threads, "
                  f"max {m.get('cpu_mhz_max')} MHz, {round(m.get('ram_gb') or 0, 1)} GB RAM, {m.get('os')}",
                  f"- Python {m.get('python')}, torch {m.get('torch')}", ""]
    lines += ["## Results (mains power)", "",
              "| model | role | runtime | precision | threads | timed quantity | p50 ms | p95 ms | mean ms | n | peak RSS MB | clock MHz | date (UTC) | git |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in pub:
        t, m = r["timing"], r["machine"]
        lines.append(f"| {r['model']} | {r['role']} | {r['runtime']} | {r['precision']} | {r['threads']} | "
                     f"{r['what_is_timed']} | **{t['p50_ms']:.0f}** | {t['p95_ms']:.0f} | {t['mean_ms']:.0f} | {t['n']} | "
                     f"{'' if r['peak_rss_mb'] is None else round(r['peak_rss_mb'])} | {m.get('cpu_mhz_now')} | "
                     f"{m.get('timestamp_utc', '')[:10]} | {m.get('git_revision') or ''} |")
    if not pub:
        lines.append("| — | no publishable run yet | | | | | | | | | | | | |")
    if unpub:
        lines += ["", "## Runs taken on battery (recorded, not publishable)", "",
                  "| model | runtime | precision | threads | p50 ms | clock MHz | battery % | date |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in unpub:
            t, m = r["timing"], r["machine"]
            lines.append(f"| {r['model']} | {r['runtime']} | {r['precision']} | {r['threads']} | {t['p50_ms']:.0f} | "
                         f"{m.get('cpu_mhz_now')} | {m.get('battery_percent')} | {m.get('timestamp_utc', '')[:10]} |")
    lines += ["", "## Sources and licences", "", "| model | source | revision | licence |", "|---|---|---|---|"]
    seen = set()
    for r in results:
        if r["model"] in seen:
            continue
        seen.add(r["model"])
        lines.append(f"| {r['model']} | {r['source']} | {r['revision'] or ''} | {r['licence']} |")
    return "\n".join(lines) + "\n"
