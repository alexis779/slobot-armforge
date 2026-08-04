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
            "hidden_dims": [256, 256, 128],
            "activation": "relu",
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [256, 256, 128],
            "activation": "relu",
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


def get_task_cfgs(task: str = "cube_disk"):
    env_cfg = {
        "num_envs": 10,
        "num_actions": 7,
        # Cartesian EE deltas; SO-101 workspace is small so keep steps moderate.
        "action_scales": [0.025, 0.025, 0.025, 0.05, 0.05, 0.05, 1.0],
        "episode_length_s": 6.0,
        "ctrl_dt": 0.02,
        "box_size": [0.03, 0.03, 0.03],
        # Must be free so the policy can push / place the cube (eval already forced False).
        "box_fixed": False,
        # Raised tabletop: floor-height cubes sit below the gripper frame reach envelope.
        "table_height": 0.10,
        "disk_radius": 0.06,
        # Require success to hold before episode end so credit is not diluted by wander.
        "success_hold_s": 0.4,
        "min_cube_disk_sep": 0.07,
        "image_resolution": (256, 256),
        "episode_resolution": (1280, 960),
        "visualize_camera": False,
        "task": task,
    }
    # Dense reach/place shaping + strong success; scales are later multiplied by ctrl_dt.
    reward_scales = {
        "reach": 1.0,
        "place": 2.0,
        "success": 8.0,
    }
    robot_cfg = {
        "ee_link_name": "gripper",
        "jaw_link_name": "moving_jaw_so101_v1",
        # Home pose aimed at the raised tabletop (gripper ~ table height).
        "default_arm_dof": [0.0, -1.2, 1.5, 1.2, 0.0],
        "default_gripper_dof": [1.7],
        "ik_method": "dls_ik",
        "gripper_open": 1.7,
        "gripper_close": 0.0,
    }
    return env_cfg, reward_scales, robot_cfg
