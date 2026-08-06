"""Replay a key-action episode recorded by ``record_lerobot.py``.

Loads the multi-hot teleop action buffer from a LeRobot dataset and drives the
same Cartesian/gripper controllers in Genesis (viewer on). At episode end,
checks whether the cube rests on the cylinder (same criteria as
``SO101CubeDiskEnv._on_disk``).

Example
-------
  python armforge/replay_lerobot.py --root datasets/so101_cube_disk --episode 0
  python armforge/replay_lerobot.py --root datasets/so101_cube_disk --episode 0 \\
      --video datasets/so101_cube_disk/videos/episode_000.mp4
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse

import numpy as np

import genesis as gs

from backend import add_backend_arg, init_genesis
from key_action_replay import BatchedKeyActionScene, load_actions_npz, load_episode_actions, report_success


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--actions-npz",
        type=str,
        default=None,
        help="Replay actions from an NPZ (key 'action') instead of a LeRobot dataset.",
    )
    parser.add_argument("--repo-id", type=str, default="local/so101_cube_disk")
    parser.add_argument("--root", type=str, default="datasets/so101_cube_disk")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--loop", action="store_true", help="Replay the episode repeatedly")
    parser.add_argument(
        "--settle-frames",
        type=int,
        default=0,
        help=(
            "Extra idle sim frames after each buffered action (default: 0). "
            "Use >0 only for older episodes that omitted idle/zero actions."
        ),
    )
    parser.add_argument(
        "--video",
        nargs="?",
        const="",
        default=None,
        help="Write an MP4 of the replay. Optional path; default "
        "<root>/videos/episode_<ep:03d>.mp4",
    )
    parser.add_argument("--no-viewer", action="store_true", help="Hide the interactive viewer")
    add_backend_arg(parser)
    args = parser.parse_args()

    root = Path(args.root) if args.root else None
    if args.actions_npz:
        actions = load_actions_npz(args.actions_npz)
        src = args.actions_npz
    else:
        actions = load_episode_actions(args.repo_id, args.episode, root=root)
        src = root if root is not None else args.repo_id
    print(f"Loaded episode {args.episode}: {len(actions)} key-actions from {src}")

    video_path: Path | None = None
    if args.video is not None:
        base = root if root is not None else Path(".")
        video_path = Path(args.video) if args.video else base / "videos" / f"episode_{args.episode:03d}.mp4"

    init_genesis(backend=args.backend, performance_mode=False, logging_level="info")
    scene = BatchedKeyActionScene(
        num_envs=1,
        fps=args.fps,
        show_viewer=not args.no_viewer,
        add_rgb_triad=True,
        video_path=video_path,
    )
    if video_path is not None:
        print(f"Recording video → {video_path}")

    settle = max(0, int(args.settle_frames))
    print(
        f"Replaying {len(actions)} actions @ {args.fps} FPS "
        f"(settle={settle} frames/action; close viewer to stop)"
    )

    def sim_tick(action: np.ndarray | None = None):
        if action is None or not np.asarray(action).any():
            scene.step(None)
        else:
            scene.step(np.asarray(action, dtype=np.float32).reshape(1, -1))

    def check_success() -> bool:
        ok, info = scene.success_info(0)
        report_success(ok, info, args.episode)
        return ok

    scene.reset_all()
    step_i = 0
    checked = False
    try:
        while True:
            if step_i < len(actions):
                act = actions[step_i]
                sim_tick(act if act.any() else None)
                step_i += 1
                if act.any():
                    for _ in range(settle):
                        sim_tick(None)
            elif not checked:
                for _ in range(scene.settle_frames()):
                    sim_tick(None)
                check_success()
                checked = True
                if video_path is not None and not args.loop:
                    break
            elif args.loop:
                print("Looping episode")
                scene.reset_all()
                step_i = 0
                checked = False
            else:
                sim_tick(None)
    except gs.GenesisException as e:
        if "Viewer closed" not in str(e):
            raise
        if not checked and step_i >= len(actions):
            check_success()
        print(f"Viewer closed after {step_i}/{len(actions)} actions")
    finally:
        if video_path is not None:
            scene.stop_recording()
            print(f"Wrote video ({step_i}/{len(actions)} actions) → {video_path}")


if __name__ == "__main__":
    main()
