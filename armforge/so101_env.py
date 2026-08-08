"""SO-ARM-101 cube-disk pick-and-place environment."""

from __future__ import annotations

import math
from functools import partial

import torch
import torch.nn.functional as F
from tensordict import TensorDict

import genesis as gs
from genesis.options.sensors import BatchRendererCameraOptions, RasterizerCameraOptions

from so101_manipulator import SO101Manipulator

try:
    import gs_madrona

    _ENABLE_MADRONA = True
except ImportError:
    _ENABLE_MADRONA = False


class SO101KitchenEnv:
    """Gym-style parallel env: pick a cube and place it on a raised disk.

    ``control_mode``:
    - ``hybrid_cartesian`` (default): SafeSort-style Δxyz + kinematic attach/release
    - ``joint``: legacy joint deltas + learned gripper
    """

    def __init__(
        self,
        env_cfg: dict,
        reward_cfg: dict,
        robot_cfg: dict,
        show_viewer: bool = False,
    ) -> None:
        self.num_envs = env_cfg["num_envs"]
        self.num_actions = env_cfg["num_actions"]
        self.cfg = env_cfg
        self.device = gs.device
        self.control_mode = str(env_cfg.get("control_mode", "hybrid_cartesian"))

        self.ctrl_dt = env_cfg["ctrl_dt"]
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.ctrl_dt)
        self.success_hold_steps = max(1, math.ceil(env_cfg.get("success_hold_s", 0.4) / self.ctrl_dt))
        self.grasp_dist = float(env_cfg.get("grasp_dist", 0.035))
        self.lift_height = float(env_cfg.get("lift_height", 0.03))
        self.release_xy = float(env_cfg.get("release_xy", 0.035))
        self.release_z_tol = float(env_cfg.get("release_z_tol", 0.04))
        self.carry_z_offset = float(env_cfg.get("carry_z_offset", -0.02))
        self.cube_pos_xy = tuple(env_cfg.get("cube_pos_xy", (0.18, 0.0)))
        self.disk_pos_xy = tuple(env_cfg.get("disk_pos_xy", (0.24, 0.08)))
        self.env_cfg = env_cfg
        self.reward_scales = dict(reward_cfg)
        self.action_scales = torch.tensor(env_cfg["action_scales"], device=self.device)

        self.image_width = env_cfg["image_resolution"][0]
        self.image_height = env_cfg["image_resolution"][1]
        self.episode_width, self.episode_height = env_cfg.get("episode_resolution", (1280, 960))

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.ctrl_dt, substeps=2),
            rigid_options=gs.options.RigidOptions(
                dt=self.ctrl_dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                noslip_iterations=2,
            ),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=list(range(min(10, self.num_envs))),
                env_separate_rigid=True,
            ),
            viewer_options=gs.options.ViewerOptions(
                res=(1280, 960),
                camera_pos=(0.8, -0.6, 0.5),
                camera_lookat=(0.15, 0.0, 0.1),
                camera_fov=55,
            ),
            profiling_options=gs.options.ProfilingOptions(show_FPS=False),
            show_viewer=show_viewer,
        )

        self.scene.add_entity(gs.morphs.Plane())
        self.table_height = float(env_cfg.get("table_height", 0.0))
        self.disk_height = float(env_cfg.get("disk_height", 0.015))
        self.disk_radius = float(env_cfg.get("disk_radius", 0.06))
        self.robot = SO101Manipulator(
            num_envs=self.num_envs,
            scene=self.scene,
            args={
                **robot_cfg,
                "base_pos": (0.0, 0.0, self.table_height),
                "control_mode": self.control_mode,
            },
            device=gs.device,
        )

        cube_size = env_cfg.get("box_size", [0.03, 0.03, 0.03])
        self.cube_half_z = 0.5 * float(cube_size[2])
        self.object = self.scene.add_entity(
            material=gs.materials.Rigid(rho=500.0, friction=1.5),
            morph=gs.morphs.Box(
                size=cube_size,
                fixed=env_cfg.get("box_fixed", False),
                batch_fixed_verts=True,
            ),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(color=(0.9, 0.15, 0.1)),
            ),
        )
        self.disk = self.scene.add_entity(
            gs.morphs.Cylinder(
                radius=self.disk_radius,
                height=self.disk_height,
                fixed=True,
            ),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(color=(0.15, 0.55, 0.9)),
            ),
        )

        if _ENABLE_MADRONA and gs.backend == gs.cuda:
            CameraOptions = BatchRendererCameraOptions
            cam_kwargs = dict(use_rasterizer=True)
        else:
            CameraOptions = RasterizerCameraOptions
            cam_kwargs = {}

        self.episode_cam = None
        self.vis_cam = None
        if env_cfg.get("enable_cameras", True):
            self.episode_cam = self.scene.add_sensor(
                CameraOptions(
                    res=(self.episode_width, self.episode_height),
                    pos=(0.9, 0.0, 0.45),
                    lookat=(0.15, 0.0, 0.08),
                    fov=50,
                    **cam_kwargs,
                )
            )
            self.vis_cam = self.episode_cam

            def _read_episode_cam(cam):
                rgb = cam.read(envs_idx=0).rgb
                if isinstance(rgb, torch.Tensor):
                    rgb = rgb.detach()
                if rgb.ndim == 4:
                    rgb = rgb[0]
                return rgb[..., :3]

            record_video = env_cfg.get("record_video", {})
            for cam_name, filename in record_video.items():
                cam = getattr(self, cam_name)
                self.scene.start_recording(
                    data_func=partial(_read_episode_cam, cam),
                    rec_options=gs.recorders.VideoFile(filename=filename),
                )

        self.scene.build(n_envs=env_cfg["num_envs"], env_spacing=(1.0, 1.0))
        self.robot.set_pd_gains()
        if self.control_mode == "hybrid_cartesian":
            self.robot.init_cartesian_buffers()

        # Event / potential terms stay absolute; only legacy dense joint rates use × dt.
        self._dt_scaled_rewards = {"reach", "grasp", "lift", "place"}
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            if name in self._dt_scaled_rewards:
                self.reward_scales[name] *= self.ctrl_dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        self._init_buffers()
        self.reset()

    def _init_buffers(self) -> None:
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.reset_buf = torch.ones(self.num_envs, dtype=gs.tc_bool, device=gs.device)
        self.success_hold_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.goal_pose = torch.zeros(self.num_envs, 7, device=gs.device, dtype=gs.tc_float)
        self.disk_pos = torch.zeros(self.num_envs, 3, device=gs.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, device=gs.device, dtype=gs.tc_float)
        self._is_held = torch.zeros(self.num_envs, dtype=gs.tc_bool, device=gs.device)
        self._did_lift = torch.zeros(self.num_envs, dtype=gs.tc_bool, device=gs.device)
        self._holding = torch.zeros(self.num_envs, dtype=gs.tc_bool, device=gs.device)
        self._attach_offset = torch.zeros(self.num_envs, 3, device=gs.device, dtype=gs.tc_float)
        self._prev_finger_cube = torch.zeros(self.num_envs, device=gs.device, dtype=gs.tc_float)
        self._prev_cube_disk_xy = torch.zeros(self.num_envs, device=gs.device, dtype=gs.tc_float)
        self._attached_this_step = torch.zeros(self.num_envs, dtype=gs.tc_bool, device=gs.device)
        self._released_this_step = torch.zeros(self.num_envs, dtype=gs.tc_bool, device=gs.device)
        self.extras = dict()

    def _reset_idx(self, envs_idx=None) -> None:
        self.robot.reset(envs_idx)

        x = torch.full((self.num_envs,), float(self.cube_pos_xy[0]), device=self.device)
        y = torch.full((self.num_envs,), float(self.cube_pos_xy[1]), device=self.device)
        z = torch.full((self.num_envs,), self.table_height + self.cube_half_z, device=self.device)
        cube_pos = torch.stack([x, y, z], dim=-1)

        disk_z = self.table_height + 0.5 * self.disk_height
        disk_pos = torch.stack(
            [
                torch.full((self.num_envs,), float(self.disk_pos_xy[0]), device=self.device),
                torch.full((self.num_envs,), float(self.disk_pos_xy[1]), device=self.device),
                torch.full((self.num_envs,), disk_z, device=self.device),
            ],
            dim=-1,
        )

        q_identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).expand(self.num_envs, -1)
        goal_pose = torch.cat([cube_pos, q_identity], dim=-1)

        if envs_idx is None:
            self.goal_pose.copy_(goal_pose)
            self.disk_pos.copy_(disk_pos)
            self.object.set_pos(cube_pos, zero_velocity=True, skip_forward=True)
            self.object.set_quat(q_identity, zero_velocity=True, skip_forward=True)
            self.disk.set_pos(disk_pos, skip_forward=False)
            self.episode_length_buf.zero_()
            self.success_hold_buf.zero_()
            self._did_lift.zero_()
            self._holding.zero_()
            self._attach_offset.zero_()
            self.reset_buf.fill_(True)
        else:
            torch.where(envs_idx[:, None], goal_pose, self.goal_pose, out=self.goal_pose)
            torch.where(envs_idx[:, None], disk_pos, self.disk_pos, out=self.disk_pos)
            self.object.set_pos(cube_pos, envs_idx=envs_idx, zero_velocity=True, skip_forward=True)
            self.object.set_quat(q_identity, envs_idx=envs_idx, zero_velocity=True, skip_forward=True)
            self.disk.set_pos(disk_pos, envs_idx=envs_idx, skip_forward=False)
            self.episode_length_buf.masked_fill_(envs_idx, 0)
            self.success_hold_buf.masked_fill_(envs_idx, 0)
            self._did_lift.masked_fill_(envs_idx, False)
            self._holding.masked_fill_(envs_idx, False)
            self._attach_offset.masked_fill_(envs_idx[:, None], 0.0)
            self.reset_buf.masked_fill_(envs_idx, True)

        # Refresh potential baselines after reset poses are applied.
        finger_d = self._finger_cube_dist()
        xy_d = torch.norm(self.object.get_pos()[:, :2] - self.disk_pos[:, :2], dim=-1)
        if envs_idx is None:
            self._prev_finger_cube.copy_(finger_d)
            self._prev_cube_disk_xy.copy_(xy_d)
            self.last_actions.zero_()
        else:
            self._prev_finger_cube = torch.where(envs_idx, finger_d, self._prev_finger_cube)
            self._prev_cube_disk_xy = torch.where(envs_idx, xy_d, self._prev_cube_disk_xy)
            self.last_actions.masked_fill_(envs_idx[:, None], 0.0)

        if self.episode_cam is not None:
            self.episode_cam._stale = True

        n_envs = envs_idx.sum() if envs_idx is not None else self.num_envs
        self.extras["episode"] = {}
        for key, value in self.episode_sums.items():
            if envs_idx is None:
                mean = value.mean()
            else:
                mean = torch.where(n_envs > 0, value[envs_idx].sum() / n_envs, 0.0)
            self.extras["episode"]["rew_" + key] = mean / self.env_cfg["episode_length_s"]
            if envs_idx is None:
                value.zero_()
            else:
                value.masked_fill_(envs_idx, 0.0)

        if envs_idx is None:
            self.extras["episode"]["success_rate"] = self._is_held.float().mean()
        else:
            n = n_envs.clamp(min=1).to(dtype=gs.tc_float)
            self.extras["episode"]["success_rate"] = (self._is_held & envs_idx).sum().to(dtype=gs.tc_float) / n

    def reset(self) -> TensorDict:
        self._reset_idx()
        return self.get_observations()

    def _update_hybrid_attach_release(self) -> None:
        self._attached_this_step.zero_()
        self._released_this_step.zero_()

        finger_d = self._finger_cube_dist()
        can_attach = (~self._holding) & (finger_d < self.grasp_dist)
        if can_attach.any():
            ee = self.robot.center_finger_pose[:, :3]
            cube = self.object.get_pos()
            # Prefer a small downward offset so the cube hangs under the fingers.
            offset = cube - ee
            offset[:, 2] = self.carry_z_offset
            self._attach_offset = torch.where(can_attach.unsqueeze(-1), offset, self._attach_offset)
            self._holding = self._holding | can_attach
            self._attached_this_step = can_attach

        # Kinematic carry while holding.
        if self._holding.any():
            ee = self.robot.center_finger_pose[:, :3]
            ee_quat = self.robot.center_finger_pose[:, 3:7]
            carry_pos = ee + self._attach_offset
            hold_idx = self._holding.nonzero(as_tuple=False).flatten()
            self.object.set_pos(carry_pos, envs_idx=hold_idx, zero_velocity=True, skip_forward=True)
            self.object.set_quat(ee_quat, envs_idx=hold_idx, zero_velocity=True, skip_forward=True)
            self._did_lift |= self._holding & self._is_lifted()

        cube_pos = self.object.get_pos()
        xy_dist = torch.norm(cube_pos[:, :2] - self.disk_pos[:, :2], dim=-1)
        disk_top = self.table_height + self.disk_height
        near_disk = xy_dist < self.release_xy
        low_enough = cube_pos[:, 2] < (disk_top + self.cube_half_z + self.release_z_tol)
        can_release = self._holding & near_disk & low_enough & self._did_lift
        if can_release.any():
            self._holding = self._holding & ~can_release
            self._released_this_step = can_release

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        actions = self.rescale_action(actions)
        self.last_actions.copy_(actions)

        if self.control_mode == "hybrid_cartesian":
            # Gripper follows previous holding state; attach/release updates after the step.
            self.robot.apply_cartesian_action(actions[:, :3], self._holding)
            self.scene.step()
            self._update_hybrid_attach_release()
        else:
            self.robot.apply_action(actions)
            self.scene.step()
            self._did_lift |= self._is_grasping() & self._is_lifted()

        self.episode_length_buf += 1

        is_success = self._success_mask()
        self.success_hold_buf = torch.where(is_success, self.success_hold_buf + 1, torch.zeros_like(self.success_hold_buf))
        is_timeout = self.episode_length_buf > self.max_episode_length
        is_held = self.success_hold_buf >= self.success_hold_steps
        self._is_held = is_held

        self.reset_buf = is_timeout | is_held
        self.reset_buf |= self.scene.rigid_solver.get_error_envs_mask()
        self.extras["time_outs"] = is_timeout.to(dtype=gs.tc_float)
        self.extras["success"] = is_held.to(dtype=gs.tc_float)

        reward = torch.zeros(self.num_envs, device=gs.device, dtype=gs.tc_float)
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            reward += rew
            self.episode_sums[name] += rew

        # Update potentials after reward (SafeSort uses prev−curr from last step).
        finger_d = self._finger_cube_dist()
        xy_d = torch.norm(self.object.get_pos()[:, :2] - self.disk_pos[:, :2], dim=-1)
        self._prev_finger_cube = finger_d.detach()
        self._prev_cube_disk_xy = xy_d.detach()

        self._reset_idx(self.reset_buf)
        return self.get_observations(), reward, self.reset_buf, self.extras

    def get_observations(self) -> TensorDict:
        finger_pos, finger_quat = (
            self.robot.center_finger_pose[:, :3],
            self.robot.center_finger_pose[:, 3:7],
        )
        target_pos = self.object.get_pos()
        target_quat = self.object.get_quat()
        hold = self._holding.to(dtype=gs.tc_float).unsqueeze(-1)
        self.obs_buf = torch.cat(
            [
                finger_pos - target_pos,
                finger_quat,
                target_pos,
                target_quat,
                target_pos - self.disk_pos,
                self.disk_pos,
                self.robot.qpos,
                hold,
                self.last_actions,
            ],
            dim=-1,
        )
        return TensorDict({"policy": self.obs_buf}, batch_size=[self.num_envs])

    def rescale_action(self, action: torch.Tensor) -> torch.Tensor:
        return torch.clamp(action, -1.0, 1.0) * self.action_scales

    def get_rgb_images(self, normalize: bool = True) -> torch.Tensor:
        """High-res episode camera, downscaled to training resolution. Shape (B, 3, H, W)."""
        if self.episode_cam is None:
            raise RuntimeError("Cameras disabled (enable_cameras=False); cannot read RGB.")
        rgb = self.episode_cam.read().rgb  # (B, H_hi, W_hi, 3)
        rgb = rgb.permute(0, 3, 1, 2).float()
        if normalize:
            rgb = rgb / 255.0
        if rgb.shape[-2] != self.image_height or rgb.shape[-1] != self.image_width:
            rgb = F.interpolate(
                rgb,
                size=(self.image_height, self.image_width),
                mode="bilinear",
                align_corners=False,
            )
        return rgb

    def get_stereo_rgb_images(self, normalize: bool = True) -> torch.Tensor:
        """Backward-compatible alias for get_rgb_images (single mono view)."""
        return self.get_rgb_images(normalize=normalize)

    def _finger_cube_dist(self) -> torch.Tensor:
        tip_offset = torch.tensor([0.0, 0.0, -0.02], device=self.device, dtype=gs.tc_float).repeat(self.num_envs, 1)
        finger_tip = self.robot.center_finger_pose[:, :3] + tip_offset
        return torch.norm(finger_tip - self.object.get_pos(), dim=-1)

    def _is_grasping(self) -> torch.Tensor:
        if self.control_mode == "hybrid_cartesian":
            return self._holding
        closed = self.robot.gripper_openness < 0.45
        return (self._finger_cube_dist() < self.grasp_dist) & closed

    def _is_lifted(self) -> torch.Tensor:
        cube_z = self.object.get_pos()[:, 2]
        return cube_z > (self.table_height + self.cube_half_z + self.lift_height)

    def _is_released(self) -> torch.Tensor:
        if self.control_mode == "hybrid_cartesian":
            return ~self._holding
        return self.robot.gripper_openness > 0.55

    def _on_disk(self) -> torch.Tensor:
        cube_pos = self.object.get_pos()
        xy_dist = torch.norm(cube_pos[:, :2] - self.disk_pos[:, :2], dim=-1)
        disk_top = self.table_height + self.disk_height
        z_ok = (cube_pos[:, 2] > disk_top + self.cube_half_z - 0.01) & (
            cube_pos[:, 2] < disk_top + self.cube_half_z + 0.025
        )
        return (xy_dist < self.disk_radius) & z_ok

    def _success_mask(self) -> torch.Tensor:
        return self._on_disk() & self._is_released() & self._did_lift

    # --- Hybrid SafeSort-style rewards ---

    def _reward_step(self) -> torch.Tensor:
        return torch.ones(self.num_envs, device=self.device, dtype=gs.tc_float)

    def _reward_approach(self) -> torch.Tensor:
        """Potential: improvement in finger→cube distance while not holding."""
        cur = self._finger_cube_dist()
        delta = self._prev_finger_cube - cur
        return torch.where(~self._holding, delta, torch.zeros_like(delta))

    def _reward_attach(self) -> torch.Tensor:
        return self._attached_this_step.to(dtype=gs.tc_float)

    def _reward_carry(self) -> torch.Tensor:
        """Potential: improvement in cube→disk XY while holding."""
        cur = torch.norm(self.object.get_pos()[:, :2] - self.disk_pos[:, :2], dim=-1)
        delta = self._prev_cube_disk_xy - cur
        return torch.where(self._holding, delta, torch.zeros_like(delta))

    # --- Legacy joint-mode rewards ---

    def _reward_reach(self) -> torch.Tensor:
        return torch.exp(-8.0 * self._finger_cube_dist())

    def _reward_grasp(self) -> torch.Tensor:
        near = torch.exp(-20.0 * self._finger_cube_dist())
        closed = (1.0 - self.robot.gripper_openness).clamp(0.0, 1.0)
        return near * closed

    def _reward_lift(self) -> torch.Tensor:
        cube_z = self.object.get_pos()[:, 2]
        height = (cube_z - (self.table_height + self.cube_half_z)).clamp(min=0.0)
        return self._is_grasping().to(dtype=gs.tc_float) * (height / 0.08).clamp(0.0, 1.0)

    def _reward_place(self) -> torch.Tensor:
        cube_pos = self.object.get_pos()
        xy_dist = torch.norm(cube_pos[:, :2] - self.disk_pos[:, :2], dim=-1)
        disk_top = self.table_height + self.disk_height
        above = (cube_pos[:, 2] > disk_top).to(dtype=gs.tc_float)
        return torch.exp(-8.0 * xy_dist) * (0.35 + 0.65 * above) * self._did_lift.to(dtype=gs.tc_float)

    def _reward_success(self) -> torch.Tensor:
        return self._is_held.to(dtype=gs.tc_float)

    def is_task_success(self) -> torch.Tensor:
        return self._success_mask()
