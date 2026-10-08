"""Write or check the hash manifest of a vendored robot folder.

    uv run scripts/vendor_manifest.py assets/robots/tiangong2pro            # check (exit 1 on mismatch)
    uv run scripts/vendor_manifest.py assets/robots/tiangong2pro --write    # (re)write after vendoring

Run with --write exactly once per vendoring, then update PROVENANCE.md with the upstream commit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from humanoid_perception.robot.assets import check_manifest, write_manifest  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("asset_dir", type=Path)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    if args.write:
        n = write_manifest(args.asset_dir)
        print(f"wrote MANIFEST.sha256 for {n} files under {args.asset_dir}")
        return 0
    missing, changed, unlisted = check_manifest(args.asset_dir)
    for label, items in (("missing", missing), ("changed", changed), ("unlisted", unlisted)):
        for rel in items:
            print(f"{label}: {rel}")
    ok = not (missing or changed or unlisted)
    print("manifest OK" if ok else "manifest MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
