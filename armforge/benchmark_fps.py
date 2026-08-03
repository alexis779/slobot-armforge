"""Benchmark parallel SO-101 sim throughput for ArmForge / AMD ROCm reports."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
import time

import torch

from backend import add_backend_arg, init_genesis
from so101_env import SO101KitchenEnv
from configs import get_task_cfgs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-B", "--num_envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--task", type=str, default="cube_disk")
    parser.add_argument("--out", type=str, default="logs/armforge_bench.json")
    parser.add_argument("--with_cameras", action="store_true", help="Include stereo RGB reads each step")
    add_backend_arg(parser)
    args = parser.parse_args()

    init_genesis(backend=args.backend, performance_mode=True)

    import genesis as gs

    env_cfg, reward_cfg, robot_cfg = get_task_cfgs(args.task)
    env_cfg["num_envs"] = args.num_envs
    env = SO101KitchenEnv(env_cfg=env_cfg, reward_cfg=reward_cfg, robot_cfg=robot_cfg, show_viewer=False)
    env.reset()

    actions = torch.zeros(args.num_envs, env.num_actions, device=gs.device)
    # Warmup
    for _ in range(20):
        env.step(actions)
        if args.with_cameras:
            env.get_rgb_images(normalize=True)

    t0 = time.perf_counter()
    for _ in range(args.steps):
        env.step(actions)
        if args.with_cameras:
            env.get_rgb_images(normalize=True)
    elapsed = time.perf_counter() - t0

    env_steps = args.num_envs * args.steps
    result = {
        "backend": str(gs.backend),
        "device": str(gs.device),
        "num_envs": args.num_envs,
        "steps": args.steps,
        "with_cameras": args.with_cameras,
        "wall_time_s": elapsed,
        "env_steps_per_s": env_steps / elapsed,
        "sim_fps_per_env": args.steps / elapsed,
    }
    print(json.dumps(result, indent=2))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="ascii") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
