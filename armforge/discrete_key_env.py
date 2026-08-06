"""SB3 VecEnv: Discrete(14) teleop keys over BatchedKeyActionScene + cube/disk rewards."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

import genesis as gs

from configs import get_task_cfgs
from key_action_replay import BatchedKeyActionScene
from record_lerobot import ACTION_NAMES, GRIPPER_CLOSE, GRIPPER_OPEN

NUM_KEY_ACTIONS = len(ACTION_NAMES)  # 14: 12 EE/RPY + grip_open + grip_close


class DiscreteKeyVecEnv(VecEnv):
    """Parallel cube/disk env with teleop-key Discrete actions for DQN.

    Action indices map 1:1 onto ``ACTION_NAMES`` (``grip_open`` / ``grip_close``
    are the two gripper values). Control uses Cartesian IK via
    ``BatchedKeyActionScene``; rewards mirror ``SO101KitchenEnv`` staging.
    """

    def __init__(
        self,
        num_envs: int,
        *,
        fps: int = 30,
        show_viewer: bool = False,
        episode_length_s: float | None = None,
        success_hold_s: float | None = None,
        reward_scales: dict | None = None,
        grasp_dist: float | None = None,
        lift_height: float | None = None,
    ) -> None:
        env_cfg, default_rewards, _robot = get_task_cfgs("cube_disk")
        self.num_envs = int(num_envs)
        self.fps = int(fps)
        self.dt = 1.0 / float(self.fps)
        self.episode_length_s = float(
            episode_length_s if episode_length_s is not None else env_cfg["episode_length_s"]
        )
        self.max_episode_length = max(1, int(round(self.episode_length_s * self.fps)))
        hold_s = float(success_hold_s if success_hold_s is not None else env_cfg["success_hold_s"])
        self.success_hold_steps = max(1, int(round(hold_s * self.fps)))
        self.grasp_dist = float(grasp_dist if grasp_dist is not None else env_cfg["grasp_dist"])
        self.lift_height = float(lift_height if lift_height is not None else env_cfg["lift_height"])

        scales = dict(reward_scales if reward_scales is not None else default_rewards)
        self.reward_scales: dict[str, float] = {}
        for name, scale in scales.items():
            # Dense terms are rates (× dt); success is a one-shot held-solve bonus.
            self.reward_scales[name] = float(scale) if name == "success" else float(scale) * self.dt

        self.scene = BatchedKeyActionScene(
            self.num_envs,
            fps=self.fps,
            show_viewer=show_viewer,
            add_rgb_triad=False,
            video_path=None,
        )
        self.geom = self.scene.geom
        self.table_height = float(self.geom["table_h"])
        self.disk_height = float(self.geom["disk_h"])
        self.disk_radius = float(self.geom["disk_radius"])
        self.cube_half_z = float(self.geom["cube_half"])
        self.device = gs.device

        self._jaw_link = self.scene.robot.get_link("moving_jaw_so101_v1")
        self._grip_span = max(GRIPPER_OPEN - GRIPPER_CLOSE, 1e-6)

        # Obs: finger-cube(3) + finger_quat(4) + cube_pose(7) + cube-disk(3) + disk(3) + qpos(6) + last_onehot(14)
        self.obs_dim = 3 + 4 + 7 + 3 + 3 + 6 + NUM_KEY_ACTIONS
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        action_space = spaces.Discrete(NUM_KEY_ACTIONS)
        super().__init__(self.num_envs, observation_space, action_space)

        self._actions: np.ndarray | None = None
        self._obs_np = np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)
        self._last_actions = torch.zeros(
            (self.num_envs, NUM_KEY_ACTIONS), device=self.device, dtype=gs.tc_float
        )
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.success_hold_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self._did_lift = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._is_held = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._disk_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self._sync_disk_pos()

    def _sync_disk_pos(self) -> None:
        disk_xy = torch.as_tensor(self.scene.disk_xy, device=self.device, dtype=gs.tc_float)
        z = self.table_height + 0.5 * self.disk_height
        self._disk_pos[:, 0] = disk_xy[:, 0]
        self._disk_pos[:, 1] = disk_xy[:, 1]
        self._disk_pos[:, 2] = z

    def _center_finger_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        g_pos = self.scene.ee_link.get_pos().reshape(self.num_envs, 3)
        g_quat = self.scene.ee_link.get_quat().reshape(self.num_envs, 4)
        j_pos = self._jaw_link.get_pos().reshape(self.num_envs, 3)
        center = (g_pos + j_pos) * 0.5
        return center, g_quat

    def _qpos(self) -> torch.Tensor:
        return self.scene.robot.get_dofs_position(self.scene.all_dofs).reshape(self.num_envs, 6)

    def _gripper_openness(self) -> torch.Tensor:
        q = self._qpos()[:, 5]
        return ((q - GRIPPER_CLOSE) / self._grip_span).clamp(0.0, 1.0)

    def _finger_cube_dist(self) -> torch.Tensor:
        tip_offset = torch.tensor(
            [0.0, 0.0, -0.02], device=self.device, dtype=gs.tc_float
        ).expand(self.num_envs, -1)
        finger, _ = self._center_finger_pose()
        cube = self.scene.cube.get_pos().reshape(self.num_envs, 3)
        return torch.norm(finger + tip_offset - cube, dim=-1)

    def _is_grasping(self) -> torch.Tensor:
        closed = self._gripper_openness() < 0.45
        return (self._finger_cube_dist() < self.grasp_dist) & closed

    def _is_lifted(self) -> torch.Tensor:
        cube_z = self.scene.cube.get_pos().reshape(self.num_envs, 3)[:, 2]
        return cube_z > (self.table_height + self.cube_half_z + self.lift_height)

    def _is_released(self) -> torch.Tensor:
        return self._gripper_openness() > 0.55

    def _on_disk(self) -> torch.Tensor:
        cube_pos = self.scene.cube.get_pos().reshape(self.num_envs, 3)
        xy_dist = torch.norm(cube_pos[:, :2] - self._disk_pos[:, :2], dim=-1)
        disk_top = self.table_height + self.disk_height
        z_ok = (cube_pos[:, 2] > disk_top + self.cube_half_z - 0.01) & (
            cube_pos[:, 2] < disk_top + self.cube_half_z + 0.025
        )
        return (xy_dist < self.disk_radius) & z_ok

    def _success_mask(self) -> torch.Tensor:
        return self._on_disk() & self._is_released() & self._did_lift

    def _compute_rewards(self) -> torch.Tensor:
        finger_dist = self._finger_cube_dist()
        reach = torch.exp(-8.0 * finger_dist)
        near = torch.exp(-20.0 * finger_dist)
        closed = (1.0 - self._gripper_openness()).clamp(0.0, 1.0)
        grasp = near * closed
        cube_pos = self.scene.cube.get_pos().reshape(self.num_envs, 3)
        height = (cube_pos[:, 2] - (self.table_height + self.cube_half_z)).clamp(min=0.0)
        lift = self._is_grasping().to(dtype=gs.tc_float) * (height / 0.08).clamp(0.0, 1.0)
        xy_dist = torch.norm(cube_pos[:, :2] - self._disk_pos[:, :2], dim=-1)
        disk_top = self.table_height + self.disk_height
        above = (cube_pos[:, 2] > disk_top).to(dtype=gs.tc_float)
        place = (
            torch.exp(-8.0 * xy_dist)
            * (0.35 + 0.65 * above)
            * self._did_lift.to(dtype=gs.tc_float)
        )
        success = self._is_held.to(dtype=gs.tc_float)
        return (
            reach * self.reward_scales["reach"]
            + grasp * self.reward_scales["grasp"]
            + lift * self.reward_scales["lift"]
            + place * self.reward_scales["place"]
            + success * self.reward_scales["success"]
        )

    def _get_obs(self) -> np.ndarray:
        finger, finger_quat = self._center_finger_pose()
        cube_pos = self.scene.cube.get_pos().reshape(self.num_envs, 3)
        cube_quat = self.scene.cube.get_quat().reshape(self.num_envs, 4)
        qpos = self._qpos()
        obs = torch.cat(
            [
                finger - cube_pos,
                finger_quat,
                cube_pos,
                cube_quat,
                cube_pos - self._disk_pos,
                self._disk_pos,
                qpos,
                self._last_actions,
            ],
            dim=-1,
        )
        obs_cpu = obs.detach()
        if obs_cpu.device.type != "cpu":
            obs_cpu = obs_cpu.cpu()
        np.copyto(self._obs_np, obs_cpu.numpy())
        return self._obs_np

    def _indices_to_onehot(self, actions: np.ndarray) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.int64).reshape(self.num_envs)
        if actions.min() < 0 or actions.max() >= NUM_KEY_ACTIONS:
            raise ValueError(
                f"Discrete actions must be in [0, {NUM_KEY_ACTIONS}), got "
                f"[{actions.min()}, {actions.max()}]"
            )
        onehot = np.zeros((self.num_envs, NUM_KEY_ACTIONS), dtype=np.float32)
        onehot[np.arange(self.num_envs), actions] = 1.0
        return onehot

    def _reset_envs(self, mask: np.ndarray) -> None:
        mask = np.asarray(mask, dtype=bool).reshape(self.num_envs)
        if not mask.any():
            return
        self.scene.reset_mask(mask)
        mask_t = torch.as_tensor(mask, device=self.device, dtype=torch.bool)
        self.episode_length_buf.masked_fill_(mask_t, 0)
        self.success_hold_buf.masked_fill_(mask_t, 0)
        self._did_lift.masked_fill_(mask_t, False)
        self._is_held.masked_fill_(mask_t, False)
        self._last_actions.masked_fill_(mask_t.unsqueeze(-1), 0.0)
        self._sync_disk_pos()

    def reset(self) -> np.ndarray:
        self.scene.reset_all()
        self.episode_length_buf.zero_()
        self.success_hold_buf.zero_()
        self._did_lift.zero_()
        self._is_held.zero_()
        self._last_actions.zero_()
        self._sync_disk_pos()
        return self._get_obs()

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = actions

    def step_wait(self):
        assert self._actions is not None
        onehot = self._indices_to_onehot(self._actions)
        self._last_actions = torch.as_tensor(onehot, device=self.device, dtype=gs.tc_float)
        self.scene.step(onehot)

        self.episode_length_buf += 1
        self._did_lift |= self._is_grasping() & self._is_lifted()

        is_success = self._success_mask()
        self.success_hold_buf = torch.where(
            is_success,
            self.success_hold_buf + 1,
            torch.zeros_like(self.success_hold_buf),
        )
        is_timeout = self.episode_length_buf > self.max_episode_length
        is_held = self.success_hold_buf >= self.success_hold_steps
        self._is_held = is_held

        physics_failed = torch.as_tensor(
            self.scene.physics_failed_mask(), device=self.device, dtype=torch.bool
        )
        dones_t = is_timeout | is_held | physics_failed
        rewards_t = self._compute_rewards()

        # Auto-reset finished envs before returning obs (SB3 VecEnv convention).
        done_mask = dones_t.detach().cpu().numpy().astype(bool)
        trunc = is_timeout.detach().cpu().numpy().astype(bool)
        succ = is_held.detach().cpu().numpy().astype(np.float32)
        # Capture terminal obs before reset for truncated episodes.
        obs_before = self._get_obs().copy()
        self._reset_envs(done_mask)
        obs = self._get_obs()

        rewards = rewards_t.detach().float().cpu().numpy()
        infos: list[dict] = []
        for i in range(self.num_envs):
            info: dict[str, Any] = {"success": float(succ[i])}
            if done_mask[i]:
                info["is_success"] = bool(succ[i] > 0.5)
                if trunc[i]:
                    info["TimeLimit.truncated"] = True
                    info["terminal_observation"] = obs_before[i].copy()
            infos.append(info)
        return obs, rewards, done_mask, infos

    def close(self) -> None:
        return None

    def get_attr(self, attr_name: str, indices=None):
        return [getattr(self, attr_name) for _ in self._get_indices(indices)]

    def set_attr(self, attr_name: str, value, indices=None) -> None:
        setattr(self, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
        method = getattr(self, method_name)
        return [method(*method_args, **method_kwargs) for _ in self._get_indices(indices)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]

    def seed(self, seed: int | None = None):
        return [seed for _ in range(self.num_envs)]

    def _get_indices(self, indices):
        if indices is None:
            return range(self.num_envs)
        if isinstance(indices, int):
            return [indices]
        return indices
