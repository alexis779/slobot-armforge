"""Evaluate ArmForge SO-101 RL (PPO/SAC/DQN) or BC policies."""

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

import numpy as np
import torch

import genesis as gs

from backend import add_backend_arg, init_genesis
from so101_env import SO101KitchenEnv


def _load_cfgs(log_dir: Path):
    with open(log_dir / "cfgs.pkl", "rb") as f:
        raw = pickle.load(f)
    if len(raw) == 5:
        env_cfg, reward_cfg, robot_cfg, rl_train_cfg, bc_train_cfg = raw
        return env_cfg, reward_cfg, robot_cfg, rl_train_cfg, bc_train_cfg, None, None, "ppo"
    if len(raw) == 7:
        env_cfg, reward_cfg, robot_cfg, rl_train_cfg, bc_train_cfg, sac_cfg, algo = raw
        return env_cfg, reward_cfg, robot_cfg, rl_train_cfg, bc_train_cfg, sac_cfg, None, algo
    if len(raw) == 8:
        env_cfg, reward_cfg, robot_cfg, rl_train_cfg, bc_train_cfg, sac_cfg, dqn_cfg, algo = raw
        return env_cfg, reward_cfg, robot_cfg, rl_train_cfg, bc_train_cfg, sac_cfg, dqn_cfg, algo
    raise ValueError(f"Unrecognized cfgs.pkl layout ({len(raw)} entries) in {log_dir}")


def load_ppo_policy(env, train_cfg, log_dir):
    try:
        if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
            raise ImportError
    except (metadata.PackageNotFoundError, ImportError, ValueError) as e:
        raise ImportError("Please install 'rsl-rl-lib>=5.0.0'.") from e
    from rsl_rl.runners import OnPolicyRunner

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    checkpoint_files = [f for f in log_dir.iterdir() if re.match(r"model_\d+\.pt", f.name)]
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {log_dir}")
    last_ckpt = max(checkpoint_files, key=lambda f: int(re.search(r"\d+", f.stem).group()))
    # HF Jobs checkpoints are CUDA tensors; map to local device for CPU / ROCm eval.
    runner.load(last_ckpt, map_location=str(gs.device))
    print(f"Loaded PPO checkpoint from {last_ckpt}")
    return runner.get_inference_policy(device=gs.device)


def load_sac_policy(log_dir: Path):
    from stable_baselines3 import SAC

    candidates = list(log_dir.glob("sac_model*.zip"))
    if not candidates:
        raise FileNotFoundError(f"No sac_model*.zip found in {log_dir}")
    # Prefer final checkpoint when present.
    finals = [p for p in candidates if "final" in p.name]
    ckpt = finals[0] if finals else max(candidates, key=lambda p: p.stat().st_mtime)
    device = str(gs.device)
    model = SAC.load(str(ckpt), device=device)
    print(f"Loaded SAC checkpoint from {ckpt} (device={device})")
    return model


def load_dqn_policy(log_dir: Path):
    from stable_baselines3 import DQN

    candidates = list(log_dir.glob("dqn_model*.zip"))
    if not candidates:
        raise FileNotFoundError(f"No dqn_model*.zip found in {log_dir}")
    finals = [p for p in candidates if "final" in p.name]
    ckpt = finals[0] if finals else max(candidates, key=lambda p: p.stat().st_mtime)
    device = str(gs.device)
    model = DQN.load(str(ckpt), device=device)
    print(f"Loaded DQN checkpoint from {ckpt} (device={device})")
    return model


def load_bc_policy(env, bc_cfg, log_dir):
    from behavior_cloning import BehaviorCloning

    bc_runner = BehaviorCloning(env, bc_cfg, None, device=gs.device)
    checkpoint_files = [f for f in log_dir.iterdir() if re.match(r"checkpoint_\d+\.pt", f.name)]
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {log_dir}")
    last_ckpt = max(checkpoint_files, key=lambda f: int(re.search(r"\d+", f.stem).group()))
    print(f"Loaded BC checkpoint from {last_ckpt}")
    bc_runner.load(str(last_ckpt))
    return bc_runner._policy


def _stage_dir(stage: str, algo: str | None) -> str:
    if stage in ("sac", "dqn", "bc"):
        return stage
    if algo == "sac":
        return "sac"
    if algo == "dqn":
        return "dqn"
    return stage


def _eval_dqn(log_dir: Path, args) -> None:
    from discrete_key_env import DiscreteKeyVecEnv

    num_envs = args.num_envs if args.num_envs is not None else (1 if args.vis else 10)
    vec = DiscreteKeyVecEnv(num_envs, show_viewer=args.vis)
    policy = load_dqn_policy(log_dir)
    max_sim_step = int(vec.max_episode_length)
    successes = 0
    print(f"[ArmForge] Evaluating algo=dqn episodes={args.episodes} num_envs={vec.num_envs}")
    for ep in range(args.episodes):
        obs = vec.reset()
        ep_success = np.zeros(vec.num_envs, dtype=bool)
        for _ in range(max_sim_step):
            actions, _ = policy.predict(obs, deterministic=True)
            obs, _rews, dones, infos = vec.step(actions)
            for i, info in enumerate(infos):
                if dones[i] and info.get("is_success", False):
                    ep_success[i] = True
                elif float(info.get("success", 0.0)) > 0.5:
                    ep_success[i] = True
        successes += int(ep_success.sum())
        print(f"Episode {ep + 1}: success_rate={ep_success.mean():.2f}")
    total = args.episodes * vec.num_envs
    print(f"Overall success: {successes}/{total} ({100.0 * successes / total:.1f}%)")
    vec.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="armforge_so101")
    parser.add_argument("--stage", type=str, default="rl", choices=["rl", "bc", "sac", "dqn"])
    parser.add_argument(
        "--algo",
        type=str,
        default=None,
        choices=["ppo", "sac", "dqn"],
        help="Override algo for --stage rl",
    )
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("-B", "--num_envs", type=int, default=None)
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    add_backend_arg(parser)
    args = parser.parse_args()

    init_genesis(backend=args.backend, performance_mode=True, logging_level="info")

    stage_dir = _stage_dir(args.stage, args.algo)
    log_dir = Path("logs") / f"{args.exp_name}_{stage_dir}"
    env_cfg, reward_cfg, robot_cfg, rl_train_cfg, bc_train_cfg, _sac_cfg, _dqn_cfg, cfg_algo = _load_cfgs(
        log_dir
    )
    if args.stage == "sac":
        algo = "sac"
    elif args.stage == "dqn":
        algo = "dqn"
    else:
        algo = args.algo or (cfg_algo or "ppo")

    if algo == "dqn":
        _eval_dqn(log_dir, args)
        return

    if args.num_envs is not None:
        env_cfg["num_envs"] = args.num_envs
    else:
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

    if args.stage == "bc":
        policy = load_bc_policy(env, bc_train_cfg, log_dir)
        policy.eval()
        algo = "bc"
    elif algo == "sac":
        policy = load_sac_policy(log_dir)
    else:
        policy = load_ppo_policy(env, rl_train_cfg, log_dir)

    successes = 0
    max_sim_step = int(env_cfg["episode_length_s"] / env_cfg["ctrl_dt"])
    print(f"[ArmForge] Evaluating algo={algo} episodes={args.episodes} num_envs={env.num_envs}")
    with torch.no_grad():
        for ep in range(args.episodes):
            obs_dict = env.reset()
            ep_success = torch.zeros(env.num_envs, device=gs.device, dtype=torch.bool)
            for _ in range(max_sim_step):
                if algo == "bc":
                    rgb_obs = env.get_rgb_images(normalize=True).float()
                    ee_pose = env.robot.ee_pose.float()
                    actions = policy(rgb_obs, ee_pose)
                elif algo == "sac":
                    obs_np = obs_dict["policy"].detach().cpu().numpy()
                    actions_np, _ = policy.predict(obs_np, deterministic=True)
                    actions = torch.as_tensor(actions_np, device=gs.device, dtype=gs.tc_float)
                else:
                    actions = policy(obs_dict)
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
