"""Train SO-101 ArmForge policies (privileged PPO/SAC/DQN teacher + vision BC student)."""

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


def _require_rsl_rl():
    try:
        if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
            raise ImportError
    except (metadata.PackageNotFoundError, ImportError) as e:
        raise ImportError("Please install 'rsl-rl-lib>=5.0.0'.") from e
    from rsl_rl.runners import OnPolicyRunner

    return OnPolicyRunner


import genesis as gs

from backend import add_backend_arg, init_genesis
from configs import get_dqn_cfg, get_sac_cfg, get_task_cfgs, get_train_cfg
from so101_env import SO101KitchenEnv


def _rl_stage_dir(algo: str) -> str:
    if algo == "sac":
        return "sac"
    if algo == "dqn":
        return "dqn"
    return "rl"


def load_teacher_policy(env, rl_train_cfg, exp_name):
    OnPolicyRunner = _require_rsl_rl()
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


def _train_ppo(env, rl_train_cfg, log_dir, max_iterations: int):
    OnPolicyRunner = _require_rsl_rl()
    runner = OnPolicyRunner(env, rl_train_cfg, log_dir, device=gs.device)
    steps_per_iter = int(rl_train_cfg["num_steps_per_env"]) * int(env.num_envs)
    total_env_steps = steps_per_iter * int(max_iterations)
    print(
        f"[ArmForge] Starting PPO training: max_iterations={max_iterations} "
        f"num_envs={env.num_envs} steps_per_env={rl_train_cfg['num_steps_per_env']} "
        f"steps_per_iter={steps_per_iter} total_env_steps={total_env_steps} "
        f"log_dir={log_dir}"
    )
    runner.learn(num_learning_iterations=max_iterations, init_at_random_ep_len=True)


def _train_sac(env, sac_cfg: dict, log_dir: Path, max_iterations: int):
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

    from sb3_env import GenesisVecEnv, SuccessLoggingCallback

    vec = GenesisVecEnv(env)
    steps_per_iter = int(sac_cfg["num_steps_per_env"]) * int(env.num_envs)
    total_timesteps = steps_per_iter * int(max_iterations)
    device = str(gs.device)
    print(
        f"[ArmForge] Starting SAC training: max_iterations={max_iterations} "
        f"num_envs={env.num_envs} steps_per_env={sac_cfg['num_steps_per_env']} "
        f"steps_per_iter={steps_per_iter} total_timesteps={total_timesteps} "
        f"device={device} log_dir={log_dir}"
    )
    model = SAC(
        "MlpPolicy",
        vec,
        learning_rate=sac_cfg["learning_rate"],
        buffer_size=sac_cfg["buffer_size"],
        learning_starts=sac_cfg["learning_starts"],
        batch_size=sac_cfg["batch_size"],
        tau=sac_cfg["tau"],
        gamma=sac_cfg["gamma"],
        train_freq=sac_cfg["train_freq"],
        gradient_steps=sac_cfg["gradient_steps"],
        ent_coef=sac_cfg["ent_coef"],
        policy_kwargs=sac_cfg["policy_kwargs"],
        tensorboard_log=str(log_dir),
        device=device,
        verbose=1,
    )
    save_freq = max(steps_per_iter * 100, 1)
    callbacks = CallbackList(
        [
            CheckpointCallback(
                save_freq=save_freq,
                save_path=str(log_dir),
                name_prefix="sac_model",
                save_replay_buffer=False,
                save_vecnormalize=False,
            ),
            SuccessLoggingCallback(
                log_freq=steps_per_iter,
                window=max(env.num_envs * 4, 1000),
                algo="sac",
            ),
        ]
    )
    model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=False)
    final_path = log_dir / "sac_model_final.zip"
    model.save(str(final_path))
    print(f"[ArmForge] Saved SAC model to {final_path}")


def _train_dqn(dqn_cfg: dict, log_dir: Path, max_iterations: int, num_envs: int, show_viewer: bool):
    from stable_baselines3 import DQN
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

    from discrete_key_env import DiscreteKeyVecEnv
    from sb3_env import SuccessLoggingCallback

    vec = DiscreteKeyVecEnv(num_envs, show_viewer=show_viewer)
    steps_per_iter = int(dqn_cfg["num_steps_per_env"]) * int(num_envs)
    total_timesteps = steps_per_iter * int(max_iterations)
    device = str(gs.device)
    print(
        f"[ArmForge] Starting DQN training: max_iterations={max_iterations} "
        f"num_envs={num_envs} steps_per_env={dqn_cfg['num_steps_per_env']} "
        f"steps_per_iter={steps_per_iter} total_timesteps={total_timesteps} "
        f"device={device} log_dir={log_dir} action_space=Discrete(14)"
    )
    model = DQN(
        "MlpPolicy",
        vec,
        learning_rate=dqn_cfg["learning_rate"],
        buffer_size=dqn_cfg["buffer_size"],
        learning_starts=dqn_cfg["learning_starts"],
        batch_size=dqn_cfg["batch_size"],
        tau=dqn_cfg["tau"],
        gamma=dqn_cfg["gamma"],
        train_freq=dqn_cfg["train_freq"],
        gradient_steps=dqn_cfg["gradient_steps"],
        target_update_interval=dqn_cfg["target_update_interval"],
        exploration_fraction=dqn_cfg["exploration_fraction"],
        exploration_initial_eps=dqn_cfg["exploration_initial_eps"],
        exploration_final_eps=dqn_cfg["exploration_final_eps"],
        policy_kwargs=dqn_cfg["policy_kwargs"],
        tensorboard_log=str(log_dir),
        device=device,
        verbose=1,
    )
    save_freq = max(steps_per_iter * 100, 1)
    callbacks = CallbackList(
        [
            CheckpointCallback(
                save_freq=save_freq,
                save_path=str(log_dir),
                name_prefix="dqn_model",
                save_replay_buffer=False,
                save_vecnormalize=False,
            ),
            SuccessLoggingCallback(
                log_freq=steps_per_iter,
                window=max(num_envs * 4, 1000),
                algo="dqn",
            ),
        ]
    )
    model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=False)
    final_path = log_dir / "dqn_model_final.zip"
    model.save(str(final_path))
    print(f"[ArmForge] Saved DQN model to {final_path}")


def main():
    parser = argparse.ArgumentParser(description="ArmForge SO-101 training")
    parser.add_argument("-e", "--exp_name", type=str, default="armforge_so101")
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-B", "--num_envs", type=int, default=512)
    parser.add_argument("--max_iterations", type=int, default=300)
    parser.add_argument("--stage", type=str, default="rl", choices=["rl", "bc"])
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "sac", "dqn"])
    parser.add_argument("--task", type=str, default="cube_disk", choices=["cube_disk"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--human_demo_dir", type=str, default=None, help="Directory of teleop .npz episodes")
    add_backend_arg(parser)
    args = parser.parse_args()

    init_genesis(backend=args.backend, seed=args.seed, performance_mode=True)

    env_cfg, reward_scales, robot_cfg = get_task_cfgs(args.task)
    rl_train_cfg, bc_train_cfg = get_train_cfg(args.exp_name)
    sac_cfg = get_sac_cfg(args.exp_name)
    dqn_cfg = get_dqn_cfg(args.exp_name)
    if args.human_demo_dir:
        bc_train_cfg["human_demo_dir"] = args.human_demo_dir

    stage_dir = args.stage if args.stage == "bc" else _rl_stage_dir(args.algo)
    log_dir = Path("logs") / f"{args.exp_name}_{stage_dir}"
    log_dir.mkdir(parents=True, exist_ok=True)

    env_cfg["num_envs"] = args.num_envs if args.stage == "rl" else 8
    # RL teacher is privileged-state only; skip cameras for throughput on cloud GPUs.
    if args.stage == "rl":
        env_cfg["enable_cameras"] = False

    with open(log_dir / "cfgs.pkl", "wb") as f:
        pickle.dump(
            (
                env_cfg,
                reward_scales,
                robot_cfg,
                rl_train_cfg,
                bc_train_cfg,
                sac_cfg,
                dqn_cfg,
                args.algo,
            ),
            f,
        )

    if args.stage == "bc":
        env = SO101KitchenEnv(
            env_cfg=env_cfg,
            reward_cfg=reward_scales,
            robot_cfg=robot_cfg,
            show_viewer=args.vis,
        )
        from behavior_cloning import BehaviorCloning

        teacher_policy = load_teacher_policy(env, rl_train_cfg, args.exp_name)
        runner = BehaviorCloning(env, bc_train_cfg, teacher_policy, device=gs.device)
        print(
            f"[ArmForge] Starting BC training: max_iterations={args.max_iterations} "
            f"num_envs={env.num_envs} log_dir={log_dir}"
        )
        runner.learn(num_learning_iterations=args.max_iterations, log_dir=str(log_dir))
    elif args.algo == "dqn":
        _train_dqn(dqn_cfg, log_dir, args.max_iterations, args.num_envs, args.vis)
    elif args.algo == "sac":
        env = SO101KitchenEnv(
            env_cfg=env_cfg,
            reward_cfg=reward_scales,
            robot_cfg=robot_cfg,
            show_viewer=args.vis,
        )
        _train_sac(env, sac_cfg, log_dir, args.max_iterations)
    else:
        env = SO101KitchenEnv(
            env_cfg=env_cfg,
            reward_cfg=reward_scales,
            robot_cfg=robot_cfg,
            show_viewer=args.vis,
        )
        _train_ppo(env, rl_train_cfg, log_dir, args.max_iterations)


if __name__ == "__main__":
    main()
