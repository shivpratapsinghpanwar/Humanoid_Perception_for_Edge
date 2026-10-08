"""The vendored robot files are exactly what PROVENANCE.md says they are.

MANIFEST.sha256 was written when the files were copied from the upstream release
(scripts/vendor_manifest.py --write). A mismatch means a file was edited by hand (never do
that) or a new drop was vendored without regenerating the manifest.
"""

from pathlib import Path

from humanoid_perception.robot.assets import check_manifest, read_manifest

ASSET_DIR = Path(__file__).resolve().parents[1] / "assets" / "robots" / "tiangong2pro"


def test_vendored_robot_matches_its_manifest():
    assert len(read_manifest(ASSET_DIR)) > 60
    missing, changed, unlisted = check_manifest(ASSET_DIR)
    assert not missing and not changed and not unlisted, (missing[:5], changed[:5], unlisted[:5])
