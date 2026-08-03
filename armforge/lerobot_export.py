"""LeRobot-compatible episode recording helpers for ArmForge."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def write_dataset_meta(root: Path, robot: str = "so101", task: str = "cube_disk") -> None:
    meta_dir = root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    info = {
        "codebase_version": "v2.1",
        "robot_type": robot,
        "total_episodes": 0,
        "total_frames": 0,
        "fps": 50,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.npz",
        "features": {
            "observation.images.episode": {"dtype": "image", "shape": [3, 256, 256]},
            "observation.state": {"dtype": "float32", "shape": [7], "names": ["ee_pose"]},
            "observation.object_pose": {"dtype": "float32", "shape": [7]},
            "action": {
                "dtype": "float32",
                "shape": [7],
                "names": ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"],
            },
            "observation.qpos": {"dtype": "float32", "shape": [6]},
        },
        "task": task,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "notes": (
            "ArmForge episodes are stored as NPZ for fast Genesis BC training. "
            "Each file contains rgb (episode camera CHW float, training resolution), ee_pose, object_pose, action, qpos. "
            "Use export_to_lerobot_stub() or convert offline to full LeRobot parquet+mp4 if needed."
        ),
    }
    with open(meta_dir / "info.json", "w", encoding="ascii") as f:
        json.dump(info, f, indent=2)


class EpisodeRecorder:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.data_dir = self.root / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._steps: list[dict] = []
        self._episode_idx = self._next_episode_index()

    def _next_episode_index(self) -> int:
        existing = list(self.data_dir.glob("episode_*.npz"))
        if not existing:
            return 0
        return max(int(p.stem.split("_")[1]) for p in existing) + 1

    def start_episode(self) -> None:
        self._steps = []

    def add_step(
        self,
        *,
        rgb: np.ndarray,
        ee_pose: np.ndarray,
        object_pose: np.ndarray,
        action: np.ndarray,
        qpos: np.ndarray,
    ) -> None:
        self._steps.append(
            {
                "rgb": np.asarray(rgb, dtype=np.float32),
                "ee_pose": np.asarray(ee_pose, dtype=np.float32),
                "object_pose": np.asarray(object_pose, dtype=np.float32),
                "action": np.asarray(action, dtype=np.float32),
                "qpos": np.asarray(qpos, dtype=np.float32),
            }
        )

    def save_episode(self) -> Path | None:
        if not self._steps:
            return None
        path = self.data_dir / f"episode_{self._episode_idx:06d}.npz"
        np.savez_compressed(
            path,
            rgb=np.stack([s["rgb"] for s in self._steps], axis=0),
            ee_pose=np.stack([s["ee_pose"] for s in self._steps], axis=0),
            object_pose=np.stack([s["object_pose"] for s in self._steps], axis=0),
            action=np.stack([s["action"] for s in self._steps], axis=0),
            qpos=np.stack([s["qpos"] for s in self._steps], axis=0),
        )
        self._update_meta_counts(n_frames=len(self._steps))
        self._episode_idx += 1
        self._steps = []
        return path

    def _update_meta_counts(self, n_frames: int) -> None:
        meta_path = self.root / "meta" / "info.json"
        if not meta_path.is_file():
            return
        with open(meta_path, encoding="ascii") as f:
            info = json.load(f)
        info["total_episodes"] = int(info.get("total_episodes", 0)) + 1
        info["total_frames"] = int(info.get("total_frames", 0)) + n_frames
        with open(meta_path, "w", encoding="ascii") as f:
            json.dump(info, f, indent=2)


def export_to_lerobot_stub(dataset_root: str | Path, out_dir: str | Path | None = None) -> Path:
    """Write a conversion stub README for full LeRobot Hub upload."""
    root = Path(dataset_root)
    out = Path(out_dir) if out_dir else root / "LEROBOT_EXPORT.md"
    out.write_text(
        (
            "# Export ArmForge NPZ episodes to LeRobot\n\n"
            "ArmForge stores teleop episodes as NPZ under `data/episode_XXXXXX.npz` with fields:\n"
            "`rgb` (T,6,H,W), `ee_pose` (T,7), `object_pose` (T,7), `action` (T,7), `qpos` (T,6).\n\n"
            "To convert for `lerobot` training on a real SO-101:\n\n"
            "```python\n"
            "from pathlib import Path\n"
            "import numpy as np\n"
            "# write observation.images.episode parquet + mp4 per LeRobot v2.1\n"
            "```\n\n"
            "See https://huggingface.co/docs/lerobot/so101 and meta/info.json in this dataset.\n"
        ),
        encoding="ascii",
    )
    return out
