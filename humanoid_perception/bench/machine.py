"""What machine, in what state, produced a measurement.

A latency number without this is not a result. The two fields that bite on a laptop are
``on_ac_power`` and ``cpu_mhz_now``: on battery this machine drops to ~0.8 GHz and every
model runs 2-3x slower. A census run on battery is recorded but marked unpublishable.
"""

from __future__ import annotations

import datetime as _dt
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field


@dataclass
class MachineState:
    hostname: str
    os: str
    cpu: str
    cores_physical: int | None
    cores_logical: int | None
    cpu_mhz_now: float | None
    cpu_mhz_max: float | None
    ram_gb: float | None
    on_ac_power: bool | None
    battery_percent: float | None
    python: str
    torch: str | None
    torch_threads: int | None
    git_revision: str | None
    timestamp_utc: str
    notes: dict = field(default_factory=dict)

    @property
    def publishable(self) -> bool:
        """A number is publishable only when the machine was on mains power."""
        return self.on_ac_power is True

    def to_dict(self) -> dict:
        d = asdict(self)
        d["publishable"] = self.publishable
        return d


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _windows_cpu_and_power() -> tuple[float | None, float | None, bool | None, float | None]:
    """Current / max CPU MHz and AC state via WMI through PowerShell (no extra packages)."""
    if sys.platform != "win32":
        return None, None, None, None
    script = ("$p=Get-CimInstance Win32_Processor | Select-Object -First 1; "
              "$b=Get-CimInstance Win32_Battery | Select-Object -First 1; "
              "'{0}|{1}|{2}|{3}' -f $p.CurrentClockSpeed, $p.MaxClockSpeed, "
              "$(if ($b) { $b.BatteryStatus } else { 'none' }), $(if ($b) { $b.EstimatedChargeRemaining } else { '' })")
    try:
        out = subprocess.check_output(["powershell", "-NoProfile", "-Command", script], text=True, timeout=20).strip()
        now, mx, status, pct = out.split("|")
        # Win32_Battery.BatteryStatus: 1 = discharging, 2 = on AC; no battery = desktop = mains.
        on_ac = True if status == "none" else (status.strip() == "2")
        return float(now), float(mx), on_ac, (float(pct) if pct.strip() else None)
    except Exception:
        return None, None, None, None


def capture(notes: dict | None = None) -> MachineState:
    try:
        import psutil
        phys, logi = psutil.cpu_count(logical=False), psutil.cpu_count(logical=True)
        ram = psutil.virtual_memory().total / 1e9
        freq = psutil.cpu_freq()
        mhz_now, mhz_max = (freq.current, freq.max) if freq else (None, None)
        batt = psutil.sensors_battery()
        on_ac = None if batt is None else bool(batt.power_plugged)
        pct = None if batt is None else float(batt.percent)
    except Exception:
        phys = logi = None
        ram = mhz_now = mhz_max = pct = None
        on_ac = None
    w_now, w_max, w_ac, w_pct = _windows_cpu_and_power()
    # WMI reads the live clock on Windows; psutil's cpu_freq there reports the nominal value.
    if w_now is not None:
        mhz_now, mhz_max = w_now, w_max
    if w_ac is not None:
        on_ac, pct = w_ac, (w_pct if w_pct is not None else pct)
    try:
        import torch
        torch_v, threads = torch.__version__, torch.get_num_threads()
    except Exception:
        torch_v, threads = None, None
    return MachineState(
        hostname=platform.node(), os=f"{platform.system()} {platform.release()}",
        cpu=platform.processor() or "", cores_physical=phys, cores_logical=logi,
        cpu_mhz_now=mhz_now, cpu_mhz_max=mhz_max, ram_gb=ram, on_ac_power=on_ac,
        battery_percent=pct, python=sys.version.split()[0], torch=torch_v, torch_threads=threads,
        git_revision=_git_revision(), timestamp_utc=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        notes=dict(notes or {}),
    )
