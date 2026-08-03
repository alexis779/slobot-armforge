"""Train SO-101 ArmForge policies (privileged PPO teacher + vision BC student)."""

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

try:
    if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
        raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please install 'rsl-rl-lib>=5.0.0'.") from e

from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from backend import add_backend_arg, init_genesis
from behavior_cloning import BehaviorCloning
from configs import get_task_cfgs, get_train_cfg
from so101_env import SO101KitchenEnv


def load_teacher_policy(env, rl_train_cfg, exp_name):
    log_dir = Path("logs") / f"{exp_name}_rl"
    assert log_dir.exists(), f"Log directory {log_dir} does not exist"
    checkpoint_files = [f for f in log_dir.iterdir() if re.match(r"model_\d+\.pt", f.name)]
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {log_dir}")
    last_ckpt = max(checkpoint_files, key=lambda f: int(re.search(r"\d+", f.stem).group()))
    runner = OnPolicyRunner(env, rl_train_cfg, log_dir, device=gs.device)
    runner.load(last_ckpt)
    print(f"Loaded teacher policy from {last_ckpt}")
    return runner.get_inference_policy(device=gs.device)


def main():
    parser = argparse.ArgumentParser(description="ArmForge SO-101 training")
    parser.add_argument("-e", "--exp_name", type=str, default="armforge_so101")
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-B", "--num_envs", type=int, default=512)
    parser.add_argument("--max_iterations", type=int, default=300)
    parser.add_argument("--stage", type=str, default="rl", choices=["rl", "bc"])
    parser.add_argument("--task", type=str, default="cube_disk", choices=["cube_disk"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--human_demo_dir", type=str, default=None, help="Directory of teleop .npz episodes")
    add_backend_arg(parser)
    args = parser.parse_args()

    init_genesis(backend=args.backend, seed=args.seed, performance_mode=True)

    env_cfg, reward_scales, robot_cfg = get_task_cfgs(args.task)
    rl_train_cfg, bc_train_cfg = get_train_cfg(args.exp_name)
    if args.human_demo_dir:
        bc_train_cfg["human_demo_dir"] = args.human_demo_dir

    log_dir = Path("logs") / f"{args.exp_name}_{args.stage}"
    log_dir.mkdir(parents=True, exist_ok=True)

    env_cfg["num_envs"] = args.num_envs if args.stage == "rl" else 8

    with open(log_dir / "cfgs.pkl", "wb") as f:
        pickle.dump((env_cfg, reward_scales, robot_cfg, rl_train_cfg, bc_train_cfg), f)

    env = SO101KitchenEnv(
        env_cfg=env_cfg,
        reward_cfg=reward_scales,
        robot_cfg=robot_cfg,
        show_viewer=args.vis,
    )

    if args.stage == "bc":
        teacher_policy = load_teacher_policy(env, rl_train_cfg, args.exp_name)
        runner = BehaviorCloning(env, bc_train_cfg, teacher_policy, device=gs.device)
        runner.learn(num_learning_iterations=args.max_iterations, log_dir=str(log_dir))
    else:
        runner = OnPolicyRunner(env, rl_train_cfg, log_dir, device=gs.device)
        runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
