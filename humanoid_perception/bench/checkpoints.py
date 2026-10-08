"""Model checkpoints live in a plain folder, not in the Hugging Face symlink cache.

On Windows, huggingface_hub's cache layout needs symlink privileges that a normal account
does not have (WinError 1314). Downloading with ``local_dir`` writes ordinary files, which
every loader accepts as a ``pretrained_path``. The folder is gitignored (``checkpoints/``).
"""

from __future__ import annotations

import os
from pathlib import Path

# <repo>/checkpoints by default; HUMANOID_PERCEPTION_CHECKPOINTS points elsewhere (e.g. one
# shared folder for several worktrees, so a 1 GB checkpoint is downloaded once).
CHECKPOINTS = Path(os.environ.get("HUMANOID_PERCEPTION_CHECKPOINTS",
                                  Path(__file__).resolve().parents[2] / "checkpoints"))


def fetch(repo_id: str, revision: str | None = None) -> Path:
    """Download (or reuse) ``repo_id`` under checkpoints/<org>__<name>; returns the folder."""
    from huggingface_hub import snapshot_download

    target = CHECKPOINTS / repo_id.replace("/", "__")
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id, revision=revision, local_dir=str(target))
    return target


def revision_of(repo_id: str) -> str | None:
    """Short commit hash of the repository's current revision on the Hub, if reachable."""
    try:
        from huggingface_hub import HfApi
        return HfApi().model_info(repo_id).sha[:7]
    except Exception:
        return None
