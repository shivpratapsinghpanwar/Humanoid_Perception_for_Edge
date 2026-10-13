# CPU latency census

Every row below was produced by `scripts/census.py` and read back from
`benchmarks/census/*.json`; nothing here is typed by hand. The quantity timed is named
per row. SmolVLA's actions mean nothing on our body — its row is a latency probe.
Precision is the dtype the weights actually ran in, checked at run time. A row is
publishable only if the machine stayed on mains power for the whole run (sampled every 5 s).
No clock is shown: Windows' clock counters report idle or package-average values that
say nothing about the clock under load.

## Machine

- Intel64 Family 6 Model 142 Stepping 12, GenuineIntel — 4 cores / 8 threads, 16.9 GB RAM, Windows 11
- Python 3.12.13, torch 2.10.0+cpu

## Results (mains power)

| model | role | runtime | precision | threads | timed quantity | p50 ms | p95 ms | mean ms | n | peak RSS MB | date (UTC) | git |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| smolvla_base | vla | torch-cpu | fp32 | 4 | observation -> 50-action chunk, LeRobot pre/post-processors included | **9619** | 11546 | 10085 | 20 | 6664 | 2026-10-13 | 3a82f18 |

## Sources and licences

| model | source | revision | licence |
|---|---|---|---|
| smolvla_base | lerobot/smolvla_base | 5e8d12a | Apache-2.0 |
