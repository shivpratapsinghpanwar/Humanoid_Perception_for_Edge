"""M1: the CPU latency census. One model per process; one JSON per run; the doc is rendered.

    uv run scripts/census.py list                         # what can be measured
    uv run scripts/census.py run smolvla_base --threads 4  # one row, in this process
    uv run scripts/census.py all --threads 4 8             # every model x thread count, each in a subprocess
    uv run scripts/census.py render                        # benchmarks/census/*.json -> docs/BENCHMARKS.md

A run on battery is recorded with publishable = false and never enters the main table;
pass --allow-battery to run it anyway (for development), or plug the laptop in.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from humanoid_perception.bench import machine as machine_state  # noqa: E402
from humanoid_perception.bench.models import MODELS  # noqa: E402
from humanoid_perception.bench.results import RESULTS_DIR, load_all, render_markdown  # noqa: E402


def cmd_list(_: argparse.Namespace) -> int:
    for name, spec in MODELS.items():
        print(f"{name:18s} {spec.role:12s} {spec.source}  [{spec.licence}]")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    spec = MODELS[args.model]
    state = machine_state.capture()
    if not state.publishable and not args.allow_battery:
        print(f"refusing: machine is on battery ({state.battery_percent}%, {state.cpu_mhz_now} MHz). "
              f"Plug in, or pass --allow-battery to record an unpublishable run.")
        return 2
    import torch
    torch.set_num_threads(args.threads)
    result = spec.run(n=args.n, warmup=args.warmup, threads=args.threads, precision=args.precision)
    result.machine = machine_state.capture()
    path = result.save(RESULTS_DIR)
    t = result.timing
    print(f"{result.model} {result.runtime} {result.precision} t{result.threads}: "
          f"p50 {t.p50_ms:.0f} ms, p95 {t.p95_ms:.0f} ms, n {t.n}, peak RSS {result.peak_rss_mb and round(result.peak_rss_mb)} MB"
          f"{'' if result.machine.publishable else '  [ON BATTERY - not publishable]'}  -> {path}")
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    rc = 0
    for name in (args.models or list(MODELS)):
        for threads in args.threads:
            cmd = [sys.executable, str(Path(__file__).resolve()), "run", name, "--threads", str(threads),
                   "--n", str(args.n), "--warmup", str(args.warmup), "--precision", args.precision]
            if args.allow_battery:
                cmd.append("--allow-battery")
            print("$", " ".join(cmd[1:]), flush=True)
            rc |= subprocess.call(cmd)
    return rc


def cmd_render(_: argparse.Namespace) -> int:
    results = load_all(RESULTS_DIR)
    out = ROOT / "docs" / "BENCHMARKS.md"
    out.write_text(render_markdown(results), encoding="utf-8", newline="\n")
    print(f"{out}: {len(results)} result files rendered")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").set_defaults(func=cmd_list)
    r = sub.add_parser("run")
    r.add_argument("model", choices=list(MODELS))
    r.add_argument("--threads", type=int, default=4)
    r.add_argument("--n", type=int, default=20)
    r.add_argument("--warmup", type=int, default=3)
    r.add_argument("--precision", default="fp32")
    r.add_argument("--allow-battery", action="store_true")
    r.set_defaults(func=cmd_run)
    a = sub.add_parser("all")
    a.add_argument("--models", nargs="*", choices=list(MODELS))
    a.add_argument("--threads", type=int, nargs="+", default=[4, 8])
    a.add_argument("--n", type=int, default=20)
    a.add_argument("--warmup", type=int, default=3)
    a.add_argument("--precision", default="fp32")
    a.add_argument("--allow-battery", action="store_true")
    a.set_defaults(func=cmd_all)
    sub.add_parser("render").set_defaults(func=cmd_render)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
