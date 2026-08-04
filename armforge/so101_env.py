"""SO-ARM-101 cube-disk manipulation environment."""

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
    """Gym-style parallel env: place a cube on a disk with SO-101 joint-space control."""

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

        self.ctrl_dt = env_cfg["ctrl_dt"]
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.ctrl_dt)
        self.success_hold_steps = max(1, math.ceil(env_cfg.get("success_hold_s", 0.4) / self.ctrl_dt))
        self.min_cube_disk_sep = float(env_cfg.get("min_cube_disk_sep", 0.08))
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
        self.table_height = float(env_cfg.get("table_height", 0.10))
        # Raised work surface so cube/disk sit inside the SO-101 gripper-frame reach envelope.
        self.scene.add_entity(
            gs.morphs.Box(
                pos=(0.22, 0.0, 0.5 * self.table_height),
                size=(0.45, 0.35, self.table_height),
                fixed=True,
            ),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(color=(0.55, 0.45, 0.35)),
            ),
        )
        self.robot = SO101Manipulator(
            num_envs=self.num_envs,
            scene=self.scene,
            args=robot_cfg,
            device=gs.device,
        )

        cube_size = env_cfg.get("box_size", [0.03, 0.03, 0.03])
        self.cube_half_z = 0.5 * float(cube_size[2])
        # Default free: a fixed cube cannot be placed, so place/success would only reflect spawn luck.
        self.object = self.scene.add_entity(
            material=gs.materials.Rigid(rho=200.0, friction=1.2),
            morph=gs.morphs.Box(
                size=cube_size,
                fixed=env_cfg.get("box_fixed", False),
                batch_fixed_verts=True,
            ),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(color=(0.9, 0.15, 0.1)),
            ),
        )
        disk_r = env_cfg.get("disk_radius", 0.06)
        self.disk = self.scene.add_entity(
            gs.morphs.Cylinder(
                radius=disk_r,
                height=0.01,
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
            # Single high-res episode camera. Training RGB is downscaled from this view.
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

        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
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
        self.extras = dict()

    def _sample_workspace_xy(self) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.rand(self.num_envs, device=self.device) * 0.12 + 0.12
        y = (torch.rand(self.num_envs, device=self.device) - 0.5) * 0.16
        return x, y

    def _sample_separated_disk_xy(self, cube_xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample disk XY at least min_cube_disk_sep from the cube so success is not free at spawn."""
        dx, dy = self._sample_workspace_xy()
        disk_xy = torch.stack([dx + 0.05, dy], dim=-1)
        for _ in range(8):
            sep = torch.norm(disk_xy - cube_xy, dim=-1)
            is_too_close = sep < self.min_cube_disk_sep
            if not is_too_close.any():
                break
            rdx, rdy = self._sample_workspace_xy()
            resample = torch.stack([rdx + 0.05, rdy], dim=-1)
            disk_xy = torch.where(is_too_close[:, None], resample, disk_xy)
        return disk_xy[:, 0], disk_xy[:, 1]

    def _reset_idx(self, envs_idx=None) -> None:
        self.robot.reset(envs_idx)

        x, y = self._sample_workspace_xy()
        z = torch.full((self.num_envs,), self.table_height + self.cube_half_z, device=self.device)
        cube_pos = torch.stack([x, y, z], dim=-1)
        cube_xy = cube_pos[:, :2]

        dx, dy = self._sample_separated_disk_xy(cube_xy)
        disk_pos = torch.stack(
            [dx, dy, torch.full((self.num_envs,), self.table_height + 0.005, device=self.device)],
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
            self.reset_buf.fill_(True)
        else:
            torch.where(envs_idx[:, None], goal_pose, self.goal_pose, out=self.goal_pose)
            torch.where(envs_idx[:, None], disk_pos, self.disk_pos, out=self.disk_pos)
            self.object.set_pos(cube_pos, envs_idx=envs_idx, zero_velocity=True, skip_forward=True)
            self.object.set_quat(q_identity, envs_idx=envs_idx, zero_velocity=True, skip_forward=True)
            self.disk.set_pos(disk_pos, envs_idx=envs_idx, skip_forward=False)
            self.episode_length_buf.masked_fill_(envs_idx, 0)
            self.success_hold_buf.masked_fill_(envs_idx, 0)
            self.reset_buf.masked_fill_(envs_idx, True)

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

    def reset(self) -> TensorDict:
        self._reset_idx()
        return self.get_observations()

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        actions = self.rescale_action(actions)
        self.robot.apply_action(actions)
        self.scene.step()
        self.episode_length_buf += 1

        is_success = self._success_mask()
        self.success_hold_buf = torch.where(is_success, self.success_hold_buf + 1, torch.zeros_like(self.success_hold_buf))
        is_timeout = self.episode_length_buf > self.max_episode_length
        is_held = self.success_hold_buf >= self.success_hold_steps

        self.reset_buf = is_timeout | is_held
        self.reset_buf |= self.scene.rigid_solver.get_error_envs_mask()
        # Only true timeouts bootstrap values; success-held episodes are real terminals.
        self.extras["time_outs"] = is_timeout.to(dtype=gs.tc_float)
        self.extras["success"] = is_success.to(dtype=gs.tc_float)

        reward = torch.zeros(self.num_envs, device=gs.device, dtype=gs.tc_float)
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            reward += rew
            self.episode_sums[name] += rew

        self._reset_idx(self.reset_buf)
        return self.get_observations(), reward, self.reset_buf, self.extras

    def get_observations(self) -> TensorDict:
        finger_pos, finger_quat = (
            self.robot.center_finger_pose[:, :3],
            self.robot.center_finger_pose[:, 3:7],
        )
        target_pos = self.object.get_pos()
        target_quat = self.object.get_quat()
        # Disk pose is required for place; without it the policy cannot know the goal.
        self.obs_buf = torch.cat(
            [
                finger_pos - target_pos,
                finger_quat,
                target_pos,
                target_quat,
                target_pos - self.disk_pos,
                self.disk_pos,
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

    def _success_mask(self) -> torch.Tensor:
        cube_pos = self.object.get_pos()
        xy_dist = torch.norm(cube_pos[:, :2] - self.disk_pos[:, :2], dim=-1)
        max_z = self.table_height + self.cube_half_z + 0.04
        return (xy_dist < self.env_cfg.get("disk_radius", 0.06)) & (cube_pos[:, 2] < max_z)

    def _reward_reach(self) -> torch.Tensor:
        tip_offset = torch.tensor([0.0, 0.0, -0.02], device=self.device, dtype=gs.tc_float).repeat(self.num_envs, 1)
        finger_tip = self.robot.center_finger_pose[:, :3] + tip_offset
        dist = torch.norm(finger_tip - self.object.get_pos(), dim=-1)
        return torch.exp(-8.0 * dist)

    def _reward_place(self) -> torch.Tensor:
        cube_pos = self.object.get_pos()
        xy_dist = torch.norm(cube_pos[:, :2] - self.disk_pos[:, :2], dim=-1)
        max_z = self.table_height + self.cube_half_z + 0.05
        is_low = (cube_pos[:, 2] < max_z).to(dtype=gs.tc_float)
        # Sharp near-disk kernel so place credit only arrives close to success.
        return torch.exp(-25.0 * xy_dist) * (0.25 + 0.75 * is_low)

    def _reward_success(self) -> torch.Tensor:
        # Sparse: only credit while the cube is on-disk (and held until episode end).
        # Scale in configs makes a short hold outweigh a full episode of place farming.
        return self._success_mask().to(dtype=gs.tc_float)

    def is_task_success(self) -> torch.Tensor:
        return self._success_mask()
