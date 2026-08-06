"""Stable-Baselines3 VecEnv adapter around the batched cube/disk Genesis env."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

import genesis as gs


class SuccessLoggingCallback(BaseCallback):
    """Track cube-on-disk success from env infos; log to console + TensorBoard."""

    def __init__(
        self,
        log_freq: int,
        window: int = 1000,
        verbose: int = 1,
        algo: str = "rl",
    ):
        super().__init__(verbose)
        self.log_freq = max(int(log_freq), 1)
        self.algo = str(algo).upper()
        self._recent = deque(maxlen=max(int(window), 1))
        self._total_done = 0
        self._total_success = 0
        self._last_log_step = 0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")
        if dones is None or infos is None:
            return True

        for done, info in zip(dones, infos):
            if not done:
                continue
            # Prefer explicit flags; fall back to ArmForge ``success`` float.
            if "is_success" in info:
                ok = float(bool(info["is_success"]))
            else:
                ok = float(info.get("success", 0.0) > 0.5)
            self._recent.append(ok)
            self._total_done += 1
            self._total_success += int(ok)

        steps = int(self.num_timesteps)
        if steps - self._last_log_step < self.log_freq or self._total_done == 0:
            return True
        self._last_log_step = steps

        window_rate = float(np.mean(self._recent)) if self._recent else 0.0
        lifetime_rate = self._total_success / max(self._total_done, 1)
        self.logger.record("rollout/success_rate", window_rate)
        self.logger.record("rollout/success_rate_lifetime", lifetime_rate)
        self.logger.record("rollout/success_episodes", float(self._total_success))
        self.logger.record("rollout/done_episodes", float(self._total_done))
        if self.verbose >= 1:
            print(
                f"[ArmForge][{self.algo}] steps={steps} "
                f"success_rate={window_rate:.4f} (window={len(self._recent)}) "
                f"lifetime={lifetime_rate:.4f} "
                f"({self._total_success}/{self._total_done})"
            )
        return True


class GenesisVecEnv(VecEnv):
    """Wrap a parallel cube/disk env (``num_envs == B``) as an SB3 ``VecEnv``.

    Note: SB3's API is NumPy-based, so each ``step`` / ``reset`` copies obs/reward/done
    to host. Prefer keeping ``B`` modest if D2H becomes the bottleneck.
    """

    def __init__(self, env: Any) -> None:
        self.env = env
        self.num_envs = int(env.num_envs)
        self.device = gs.device

        # Probe obs dim from a reset (privileged policy vector).
        obs_td = env.get_observations()
        obs = obs_td["policy"]
        if obs.ndim != 2:
            raise ValueError(f"Expected policy obs (B, D), got {tuple(obs.shape)}")
        obs_dim = int(obs.shape[-1])
        act_dim = int(env.num_actions)

        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)
        super().__init__(self.num_envs, observation_space, action_space)

        self._actions: np.ndarray | None = None
        self._obs_np = np.zeros((self.num_envs, obs_dim), dtype=np.float32)

    def _obs_to_numpy(self, obs_td) -> np.ndarray:
        obs = obs_td["policy"].detach()
        if obs.device.type != "cpu":
            obs = obs.cpu()
        np.copyto(self._obs_np, obs.numpy())
        return self._obs_np

    def reset(self) -> np.ndarray:
        obs_td = self.env.reset()
        return self._obs_to_numpy(obs_td)

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = actions

    def step_wait(self):
        assert self._actions is not None
        actions_t = torch.as_tensor(self._actions, device=self.device, dtype=gs.tc_float)
        obs_td, rewards_t, dones_t, extras = self.env.step(actions_t)

        obs = self._obs_to_numpy(obs_td)
        rewards = rewards_t.detach().float().cpu().numpy()
        dones = dones_t.detach().cpu().numpy().astype(bool)
        timeouts = extras.get("time_outs")
        if timeouts is None:
            trunc = np.zeros(self.num_envs, dtype=bool)
        else:
            trunc = timeouts.detach().cpu().numpy().astype(bool) > 0
        success = extras.get("success")
        succ = (
            success.detach().cpu().numpy().astype(np.float32)
            if success is not None
            else np.zeros(self.num_envs, dtype=np.float32)
        )

        infos: list[dict] = []
        for i in range(self.num_envs):
            info: dict[str, Any] = {"success": float(succ[i])}
            if dones[i]:
                info["is_success"] = bool(succ[i] > 0.5)
                # Env already auto-reset; mark time-limit truncations for bootstrapping.
                if trunc[i]:
                    info["TimeLimit.truncated"] = True
                    info["terminal_observation"] = obs[i].copy()
            infos.append(info)
        return obs, rewards, dones, infos

    def close(self) -> None:
        return None

    def get_attr(self, attr_name: str, indices=None):
        return [getattr(self.env, attr_name) for _ in self._get_indices(indices)]

    def set_attr(self, attr_name: str, value, indices=None) -> None:
        setattr(self.env, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
        method = getattr(self.env, method_name)
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
