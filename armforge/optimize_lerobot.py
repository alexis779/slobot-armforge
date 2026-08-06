"""Drop idle (all-zero) teleop frames and verify the episode still succeeds.

Single-env only: keep nonzero key-actions in order, replay once, write NPZ if OK.

Example
-------
  python armforge/optimize_lerobot.py --root datasets/so101_cube_disk --episode 0 \\
      --backend cpu --out datasets/so101_cube_disk/optimized/episode_000.npz
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse

import numpy as np

from backend import add_backend_arg, init_genesis
from key_action_replay import BatchedKeyActionScene, load_actions_npz, load_episode_actions, report_success


def drop_zero_actions(actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(kept_actions, kept_indices)`` for frames with any nonzero bit."""
    actions = np.asarray(actions, dtype=np.float32)
    kept_indices = np.nonzero(actions.any(axis=1))[0].astype(np.int64)
    return actions[kept_indices], kept_indices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--actions-npz",
        type=str,
        default=None,
        help="Load actions from NPZ (key 'action') instead of a LeRobot dataset.",
    )
    parser.add_argument("--repo-id", type=str, default="local/so101_cube_disk")
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Local LeRobot dataset root. Omit to load --repo-id from the Hub.",
    )
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Write optimized NPZ. Default: <root>/optimized/episode_XXX.npz",
    )
    parser.add_argument("--no-write", action="store_true", help="Verify only; do not write NPZ")
    add_backend_arg(parser)
    args = parser.parse_args()

    root = Path(args.root) if args.root else None
    if args.actions_npz:
        actions = load_actions_npz(args.actions_npz)
        src = args.actions_npz
    else:
        if root is None and args.repo_id == "local/so101_cube_disk":
            default_root = Path("datasets/so101_cube_disk")
            if default_root.exists():
                root = default_root
        actions = load_episode_actions(args.repo_id, args.episode, root=root)
        src = str(root) if root is not None else args.repo_id

    kept_actions, kept_indices = drop_zero_actions(actions)
    n_idle = len(actions) - len(kept_indices)
    print(
        f"Loaded episode {args.episode}: {len(actions)} frames from {src} "
        f"→ drop {n_idle} idle → {len(kept_indices)} actions "
        f"({100.0 * len(kept_indices) / max(1, len(actions)):.1f}% kept)"
    )
    if len(kept_indices) == 0:
        raise RuntimeError("No nonzero actions in episode.")

    init_genesis(backend=args.backend, performance_mode=True, logging_level="warning")
    scene = BatchedKeyActionScene(num_envs=1, fps=args.fps, show_viewer=False)
    ok = scene.evaluate_subsequences(actions, [kept_indices.tolist()])
    info_ok, info = scene.success_info(0)
    report_success(bool(ok[0]), info, args.episode)
    if not bool(ok[0]):
        raise SystemExit(1)

    if args.no_write:
        return

    out = Path(args.out) if args.out else None
    if out is None:
        base = root if root is not None else Path("optimized")
        out = base / "optimized" / f"episode_{args.episode:03d}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        action=kept_actions.astype(np.float32),
        kept_indices=kept_indices,
        original_length=np.int64(len(actions)),
        episode=np.int64(args.episode),
        repo_id=np.asarray(args.repo_id),
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
