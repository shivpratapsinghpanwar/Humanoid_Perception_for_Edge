"""Hash manifests for vendored robot files: written once when copying, checked by a test.

A vendored robot must be exactly what PROVENANCE.md says it is. Text files are hashed
with line endings normalised, because a Windows checkout rewrites LF as CRLF and that is
not a change to the robot. Everything else is hashed as bytes.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

MANIFEST_NAME = "MANIFEST.sha256"
EXCLUDED = {MANIFEST_NAME, "PROVENANCE.md"}
TEXT_SUFFIXES = {".xml", ".yaml", ".yml", ".urdf", ".xacro", ".csv", ".md", ".txt", ".json", ""}


def file_digest(path: Path) -> str:
    data = path.read_bytes()
    if path.suffix.lower() in TEXT_SUFFIXES:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def _files(asset_dir: Path):
    for p in sorted(asset_dir.rglob("*")):
        if p.is_file() and p.name not in EXCLUDED:
            yield p


def write_manifest(asset_dir: Path) -> int:
    """Write MANIFEST.sha256 for every file under ``asset_dir``; returns the file count."""
    lines = [f"{file_digest(p)}  {p.relative_to(asset_dir).as_posix()}" for p in _files(asset_dir)]
    (asset_dir / MANIFEST_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return len(lines)


def read_manifest(asset_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in (asset_dir / MANIFEST_NAME).read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, rel = line.split(None, 1)
            out[rel.strip()] = digest.strip().lower()
    return out


def check_manifest(asset_dir: Path) -> tuple[list[str], list[str], list[str]]:
    """(missing, changed, unlisted): files the manifest names but are absent or differ,
    and files present but not in the manifest."""
    manifest = read_manifest(asset_dir)
    missing, changed = [], []
    for rel, digest in manifest.items():
        f = asset_dir / rel
        if not f.is_file():
            missing.append(rel)
        elif file_digest(f) != digest:
            changed.append(rel)
    unlisted = [p.relative_to(asset_dir).as_posix() for p in _files(asset_dir)
                if p.relative_to(asset_dir).as_posix() not in manifest]
    return missing, changed, unlisted
