"""Evaluate ArmForge SO-101 RL or BC policies."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import pickle
import re
from importlib import metadata

import torch

try:
    if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
        raise ImportError
except (metadata.PackageNotFoundError, ImportError, ValueError) as e:
    raise ImportError("Please install 'rsl-rl-lib>=5.0.0'.") from e

from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from backend import add_backend_arg, init_genesis
from behavior_cloning import BehaviorCloning
from so101_env import SO101KitchenEnv


def load_rl_policy(env, train_cfg, log_dir):
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    checkpoint_files = [f for f in log_dir.iterdir() if re.match(r"model_\d+\.pt", f.name)]
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {log_dir}")
    last_ckpt = max(checkpoint_files, key=lambda f: int(re.search(r"\d+", f.stem).group()))
    # HF Jobs checkpoints are CUDA tensors; map to local device for CPU / ROCm eval.
    runner.load(last_ckpt, map_location=str(gs.device))
    print(f"Loaded RL checkpoint from {last_ckpt}")
    return runner.get_inference_policy(device=gs.device)


def load_bc_policy(env, bc_cfg, log_dir):
    bc_runner = BehaviorCloning(env, bc_cfg, None, device=gs.device)
    checkpoint_files = [f for f in log_dir.iterdir() if re.match(r"checkpoint_\d+\.pt", f.name)]
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {log_dir}")
    last_ckpt = max(checkpoint_files, key=lambda f: int(re.search(r"\d+", f.stem).group()))
    print(f"Loaded BC checkpoint from {last_ckpt}")
    bc_runner.load(str(last_ckpt))
    return bc_runner._policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="armforge_so101")
    parser.add_argument("--stage", type=str, default="rl", choices=["rl", "bc"])
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    add_backend_arg(parser)
    args = parser.parse_args()

    init_genesis(backend=args.backend, performance_mode=True, logging_level="info")

    log_dir = Path("logs") / f"{args.exp_name}_{args.stage}"
    with open(log_dir / "cfgs.pkl", "rb") as f:
        env_cfg, reward_cfg, robot_cfg, rl_train_cfg, bc_train_cfg = pickle.load(f)

    env_cfg["num_envs"] = 1 if (args.vis or args.record) else 10
    env_cfg["box_fixed"] = False
    env_cfg["visualize_camera"] = args.vis
    if args.record:
        env_cfg["record_video"] = {
            "vis_cam": str(log_dir / "eval.mp4"),
            "episode_cam": str(log_dir / "episode.mp4"),
        }
        env_cfg["visualize_camera"] = True
        env_cfg["enable_cameras"] = True

    env = SO101KitchenEnv(
        env_cfg=env_cfg,
        reward_cfg=reward_cfg,
        robot_cfg=robot_cfg,
        show_viewer=args.vis,
    )

    if args.stage == "rl":
        policy = load_rl_policy(env, rl_train_cfg, log_dir)
    else:
        policy = load_bc_policy(env, bc_train_cfg, log_dir)
        policy.eval()

    successes = 0
    max_sim_step = int(env_cfg["episode_length_s"] / env_cfg["ctrl_dt"])
    with torch.no_grad():
        for ep in range(args.episodes):
            obs_dict = env.reset()
            ep_success = torch.zeros(env.num_envs, device=gs.device, dtype=torch.bool)
            for _ in range(max_sim_step):
                if args.stage == "rl":
                    actions = policy(obs_dict)
                else:
                    rgb_obs = env.get_rgb_images(normalize=True).float()
                    ee_pose = env.robot.ee_pose.float()
                    actions = policy(rgb_obs, ee_pose)
                obs_dict, _rews, _dones, infos = env.step(actions)
                ep_success |= infos["success"] > 0.5
            successes += int(ep_success.float().mean().item() * env.num_envs)
            print(f"Episode {ep + 1}: success_rate={ep_success.float().mean().item():.2f}")

    total = args.episodes * env.num_envs
    print(f"Overall success: {successes}/{total} ({100.0 * successes / total:.1f}%)")
    if args.record:
        env.scene.stop_recording()


if __name__ == "__main__":
    main()
