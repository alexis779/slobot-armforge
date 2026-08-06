"""SO-101 MJCF asset resolution for ArmForge."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path

from huggingface_hub import snapshot_download

_SO101_PATTERN = "SO101/*"
_ASSETS_REPO = "Genesis-Intelligence/assets"


def so101_mjcf_path() -> Path:
    """Download (if needed) and return a SO-101 MJCF with base-link collision geoms.

    Upstream ``so101_new_calib.xml`` only attaches *visual* meshes on ``base``. In
    collision visualization that looks like a missing base, and the shoulder
    collision hulls sit on the table with nothing connecting them visually.

    Set ``ARMFORGE_SO101_ROOT`` to a local SO101 directory to skip Hub download
    (useful on air-gapped / firewalled cloud hosts).
    """
    override = os.environ.get("ARMFORGE_SO101_ROOT", "").strip()
    if override:
        src = Path(override).expanduser().resolve() / "so101_new_calib.xml"
    else:
        asset_root = Path(
            snapshot_download(
                repo_type="dataset",
                repo_id=_ASSETS_REPO,
                allow_patterns=[_SO101_PATTERN],
            )
        )
        src = asset_root / "SO101" / "so101_new_calib.xml"
    if not src.is_file():
        raise FileNotFoundError(f"SO-101 MJCF not found at {src}")
    return _mjcf_with_base_collision(src)


def _mjcf_with_base_collision(src: Path) -> Path:
    """Write a sibling MJCF that mirrors base visual meshes as collision geoms."""
    out = src.with_name("so101_new_calib_base_col.xml")
    if out.is_file() and out.stat().st_mtime >= src.stat().st_mtime:
        return out

    tree = ET.parse(src)
    root = tree.getroot()
    base = root.find(".//body[@name='base']")
    if base is None:
        raise RuntimeError(f"No body name='base' in {src}")

    # Only direct-child visual geoms on the base (not nested links).
    visuals = [
        g
        for g in list(base)
        if g.tag == "geom" and "visual" in (g.get("class") or "")
    ]
    # Avoid duplicating if we re-run on an already-patched tree.
    existing_col = {
        (g.get("mesh"), g.get("pos"), g.get("quat"))
        for g in base.findall("geom")
        if "collision" in (g.get("class") or "")
    }
    for vis in visuals:
        key = (vis.get("mesh"), vis.get("pos"), vis.get("quat"))
        if key in existing_col:
            continue
        col = ET.Element("geom", dict(vis.attrib))
        col.set("class", "collision")
        # Insert collision geom right after its visual twin.
        idx = list(base).index(vis)
        base.insert(idx + 1, col)

    tree.write(out, encoding="utf-8", xml_declaration=True)
    return out
