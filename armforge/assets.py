"""SO-101 MJCF asset resolution for ArmForge."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download

_SO101_PATTERN = "SO101/*"
_ASSETS_REPO = "Genesis-Intelligence/assets"


def so101_mjcf_path() -> Path:
    """Download (if needed) and return the calibrated SO-101 MJCF path."""
    asset_root = Path(
        snapshot_download(
            repo_type="dataset",
            repo_id=_ASSETS_REPO,
            allow_patterns=[_SO101_PATTERN],
        )
    )
    path = asset_root / "SO101" / "so101_new_calib.xml"
    if not path.is_file():
        raise FileNotFoundError(f"SO-101 MJCF not found at {path}")
    return path
