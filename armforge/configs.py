"""Shared ArmForge task and training hyperparameters."""

from __future__ import annotations


def get_train_cfg(exp_name: str):
    rl_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.0003,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
            "obs_normalization": True,
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
            "obs_normalization": True,
        },
        "obs_groups": {
            "actor": ["policy"],
            "critic": ["policy"],
        },
        "num_steps_per_env": 24,
        "save_interval": 100,
        "run_name": exp_name,
        "logger": "tensorboard",
    }

    bc_cfg_dict = {
        "num_steps_per_env": 24,
        "learning_rate": 0.001,
        "num_epochs": 5,
        "num_mini_batches": 10,
        "max_grad_norm": 1.0,
        "policy": {
            "vision_encoder": {
                "conv_layers": [
                    {"in_channels": 3, "out_channels": 8, "kernel_size": 3, "stride": 1, "padding": 1},
                    {"in_channels": 8, "out_channels": 16, "kernel_size": 3, "stride": 2, "padding": 1},
                    {"in_channels": 16, "out_channels": 32, "kernel_size": 3, "stride": 2, "padding": 1},
                ],
                "pooling": "adaptive_avg",
            },
            "action_head": {
                "state_obs_dim": 7,
                "hidden_dims": [128, 128, 64],
            },
            "pose_head": {
                "hidden_dims": [64, 64],
            },
        },
        "buffer_size": 1000,
        "log_freq": 10,
        "save_freq": 50,
        "eval_freq": 50,
        "human_mix_ratio": 0.3,
    }
    return rl_cfg_dict, bc_cfg_dict


def get_sac_cfg(exp_name: str) -> dict:
    """Stable-Baselines3 SAC hyperparameters (privileged cube/disk teacher)."""
    return {
        "run_name": exp_name,
        "algorithm": "SAC",
        # Matched to PPO: total_timesteps = max_iterations * num_steps_per_env * num_envs
        "num_steps_per_env": 24,
        "learning_rate": 3e-4,
        "buffer_size": 1_000_000,
        "learning_starts": 10_000,
        "batch_size": 256,
        "tau": 0.005,
        "gamma": 0.99,
        "train_freq": 1,
        "gradient_steps": 1,
        "ent_coef": "auto",
        "policy_kwargs": {
            "net_arch": [256, 256, 128],
        },
        "save_interval_timesteps": None,  # set in train from iteration budget
    }


def get_dqn_cfg(exp_name: str) -> dict:
    """Stable-Baselines3 DQN hyperparameters (Discrete teleop-key teacher)."""
    return {
        "run_name": exp_name,
        "algorithm": "DQN",
        # Matched to PPO/SAC: total_timesteps = max_iterations * num_steps_per_env * num_envs
        "num_steps_per_env": 24,
        "learning_rate": 1e-4,
        "buffer_size": 1_000_000,
        "learning_starts": 10_000,
        "batch_size": 256,
        "tau": 1.0,
        "gamma": 0.99,
        "train_freq": 4,
        "gradient_steps": 1,
        "target_update_interval": 1000,
        "exploration_fraction": 0.2,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
        "policy_kwargs": {
            "net_arch": [256, 256, 128],
        },
        "save_interval_timesteps": None,
    }


def get_task_cfgs(task: str = "cube_disk"):
    env_cfg = {
        "num_envs": 10,
        # SafeSort-style hybrid: world-frame Δxyz only (scripted attach/release).
        "control_mode": "hybrid_cartesian",
        "num_actions": 3,
        "action_scales": [0.0075, 0.0075, 0.0075],
        "episode_length_s": 8.0,
        "ctrl_dt": 0.02,
        "box_size": [0.03, 0.03, 0.03],
        "box_fixed": False,
        "table_height": 0.0,
        # Disk diameter = 2× cube side ⇒ radius = cube side; height = half cube side.
        "disk_radius": 0.03,
        "disk_height": 0.015,
        "success_hold_s": 0.3,
        # Fixed spawn poses (no randomization) to simplify learning / teleop.
        "cube_pos_xy": (0.18, 0.0),
        "disk_pos_xy": (0.24, 0.08),
        "grasp_dist": 0.05,
        "lift_height": 0.03,
        "release_xy": 0.035,
        "release_z_tol": 0.04,
        "carry_z_offset": -0.02,
        "image_resolution": (256, 256),
        "episode_resolution": (1280, 960),
        "visualize_camera": False,
        "task": task,
    }
    # SafeSort-style potentials + event bonuses (absolute; not × dt except legacy joint terms).
    reward_scales = {
        "step": -0.01,
        "approach": 8.0,
        "attach": 8.0,
        "carry": 10.0,
        "success": 30.0,
    }
    robot_cfg = {
        "ee_link_name": "gripper",
        "jaw_link_name": "moving_jaw_so101_v1",
        # Mount on the ground plane (no tabletop).
        "base_pos": (0.0, 0.0, 0.0),
        # MJCF gripper range is ~[-0.175, 1.745].
        "gripper_open": 1.745,
        "gripper_close": -0.174,
    }
    return env_cfg, reward_scales, robot_cfg
