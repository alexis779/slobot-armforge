"""SO-ARM-101 manipulator: joint deltas or SafeSort-style Cartesian IK + scripted grip."""

from __future__ import annotations

import torch

import genesis as gs

from assets import so101_mjcf_path


class SO101Manipulator:
    """5-DoF arm + 1-DoF gripper.

    Modes:
    - joint (legacy): scaled joint deltas + absolute gripper in [-1, 1]
    - hybrid_cartesian: world-frame Δxyz via batched IK; gripper open/close from holding mask
    """

    def __init__(self, num_envs: int, scene: gs.Scene, args: dict, device: str = "cpu"):
        self._device = device
        self._scene = scene
        self._num_envs = num_envs
        self._args = args
        self.control_mode = str(args.get("control_mode", "joint"))

        mjcf = args.get("mjcf_path") or str(so101_mjcf_path())
        morph = gs.morphs.MJCF(
            file=mjcf,
            pos=tuple(args.get("base_pos", (0.0, 0.0, 0.0))),
            quat=tuple(args.get("base_quat", (1.0, 0.0, 0.0, 0.0))),
            convexify=True,
            decompose_robot_error_threshold=0.0,
        )
        self._robot_entity = scene.add_entity(
            material=gs.materials.Rigid(gravity_compensation=1.0),
            morph=morph,
        )

        # Gripper hinge: closed near lower bound, open near upper bound (MJCF range).
        self._gripper_open_dof = float(args.get("gripper_open", 1.745))
        self._gripper_close_dof = float(args.get("gripper_close", -0.174))
        self._init()

    def set_pd_gains(self) -> None:
        # Moderate gains: enough for grasp/lift without batting the cube off the table.
        kp = torch.tensor([40.0, 40.0, 40.0, 30.0, 20.0, 20.0], device=self._device)
        kv = torch.tensor([2.0, 2.0, 2.0, 1.5, 1.0, 1.0], device=self._device)
        self._robot_entity.set_dofs_kp(kp)
        self._robot_entity.set_dofs_kv(kv)
        force = torch.tensor([5.0, 5.0, 5.0, 4.0, 3.0, 3.0], device=self._device)
        self._robot_entity.set_dofs_force_range(-force, force)

    def _init(self) -> None:
        self._arm_dof_dim = 5
        self._gripper_dim = 1
        self._arm_dof_idx = torch.arange(self._arm_dof_dim, device=self._device)
        self._gripper_dof = torch.tensor([self._arm_dof_dim], device=self._device)
        self._all_dofs = torch.arange(6, device=self._device)
        self._ee_link = self._robot_entity.get_link(self._args.get("ee_link_name", "gripper"))
        jaw_name = self._args.get("jaw_link_name", "moving_jaw_so101_v1")
        self._jaw_link = self._robot_entity.get_link(jaw_name)

        self._init_qpos = torch.tensor(
            [0.0] * self._arm_dof_dim + [self._gripper_open_dof],
            dtype=torch.float32,
            device=self._device,
        )
        # Filled after scene.build() via init_cartesian_buffers().
        self.target_pos: torch.Tensor | None = None
        self.target_quat: torch.Tensor | None = None
        self._q_lo: torch.Tensor | None = None
        self._q_hi: torch.Tensor | None = None

    def init_cartesian_buffers(self) -> None:
        """Call after ``scene.build`` so EE pose / DOF limits are valid."""
        b = self._num_envs
        pos = self._ee_link.get_pos().reshape(b, 3).to(dtype=gs.tc_float)
        quat = self._ee_link.get_quat().reshape(b, 4).to(dtype=gs.tc_float)
        self.target_pos = pos.clone()
        self.target_quat = quat.clone()
        try:
            lim = self._robot_entity.get_dofs_limit(self._all_dofs)
            if isinstance(lim, tuple):
                lo, hi = lim
            else:
                lim_t = torch.as_tensor(lim, device=self._device, dtype=gs.tc_float)
                if lim_t.shape[0] == 2:
                    lo, hi = lim_t[0], lim_t[1]
                else:
                    lo, hi = lim_t[:, 0], lim_t[:, 1]
            self._q_lo = torch.as_tensor(lo, device=self._device, dtype=gs.tc_float).reshape(-1)[:6]
            self._q_hi = torch.as_tensor(hi, device=self._device, dtype=gs.tc_float).reshape(-1)[:6]
        except Exception:
            self._q_lo = torch.tensor(
                [-1.92, -1.75, -1.69, -1.69, -2.74, self._gripper_close_dof],
                device=self._device,
                dtype=gs.tc_float,
            )
            self._q_hi = torch.tensor(
                [1.92, 1.75, 1.69, 1.69, 2.74, self._gripper_open_dof],
                device=self._device,
                dtype=gs.tc_float,
            )

    def reset(self, envs_idx=None, skip_forward=True) -> None:
        self._robot_entity.set_qpos(
            self._init_qpos,
            envs_idx=envs_idx,
            zero_velocity=True,
            skip_forward=skip_forward,
        )
        if self.target_pos is not None:
            pos = self._ee_link.get_pos().reshape(self._num_envs, 3).to(dtype=gs.tc_float)
            quat = self._ee_link.get_quat().reshape(self._num_envs, 4).to(dtype=gs.tc_float)
            if envs_idx is None:
                self.target_pos.copy_(pos)
                self.target_quat.copy_(quat)
            else:
                self.target_pos[envs_idx] = pos[envs_idx]
                self.target_quat[envs_idx] = quat[envs_idx]

    def apply_action(self, action: torch.Tensor) -> None:
        """Apply scaled arm joint deltas (5) + absolute gripper command (1) in [-1, 1]."""
        q_pos = self._robot_entity.get_qpos().clone()
        q_pos[:, : self._arm_dof_dim] = q_pos[:, : self._arm_dof_dim] + action[:, : self._arm_dof_dim]
        grip_cmd = action[:, self._arm_dof_dim]
        grip = 0.5 * (grip_cmd + 1.0) * (self._gripper_open_dof - self._gripper_close_dof) + self._gripper_close_dof
        q_pos[:, self._gripper_dof] = grip.unsqueeze(-1)
        self._robot_entity.control_dofs_position(position=q_pos)

    def apply_cartesian_action(self, dxyz: torch.Tensor, holding: torch.Tensor) -> None:
        """World-frame EE Δxyz (meters) + scripted gripper from ``holding`` mask."""
        if self.target_pos is None:
            raise RuntimeError("Call init_cartesian_buffers() after scene.build()")
        self.target_pos = self.target_pos + dxyz.to(dtype=gs.tc_float)
        # Keep a reachable workspace around the tabletop task.
        self.target_pos[:, 0].clamp_(0.05, 0.35)
        self.target_pos[:, 1].clamp_(-0.20, 0.20)
        self.target_pos[:, 2].clamp_(0.02, 0.25)

        # Position-only IK (fixed approach orientation from reset) — matches teleop EE moves.
        q_arm = self._robot_entity.inverse_kinematics(
            link=self._ee_link,
            pos=self.target_pos,
            dofs_idx_local=self._arm_dof_idx,
            max_samples=1,
            max_solver_iters=50,
            damping=0.05,
            pos_tol=1e-3,
            rot_tol=1e-2,
        ).reshape(self._num_envs, -1)

        q_cmd = self._robot_entity.get_qpos().clone()
        arm = q_arm[:, : self._arm_dof_dim]
        if self._q_lo is not None:
            arm = torch.clamp(arm, self._q_lo[:5], self._q_hi[:5])
        q_cmd[:, : self._arm_dof_dim] = arm

        closed = holding.to(dtype=torch.bool)
        grip = torch.where(
            closed,
            torch.full((self._num_envs,), self._gripper_close_dof, device=self._device, dtype=gs.tc_float),
            torch.full((self._num_envs,), self._gripper_open_dof, device=self._device, dtype=gs.tc_float),
        )
        q_cmd[:, self._arm_dof_dim] = grip
        self._robot_entity.control_dofs_position(position=q_cmd)

        # Snap unreachable targets toward achieved EE to avoid IK wind-up.
        link_pos = self._ee_link.get_pos().reshape(self._num_envs, 3).to(dtype=gs.tc_float)
        err = torch.linalg.norm(link_pos - self.target_pos, dim=-1)
        snap = err > 0.04
        if snap.any():
            self.target_pos = torch.where(snap.unsqueeze(-1), link_pos, self.target_pos)

    @property
    def entity(self):
        return self._robot_entity

    @property
    def ee_pose(self) -> torch.Tensor:
        pos, quat = self._ee_link.get_pos(), self._ee_link.get_quat()
        return torch.cat([pos, quat], dim=-1)

    @property
    def center_finger_pose(self) -> torch.Tensor:
        """Midpoint between fixed gripper body and moving jaw."""
        g_pos, g_quat = self._ee_link.get_pos(), self._ee_link.get_quat()
        j_pos = self._jaw_link.get_pos()
        center = (g_pos + j_pos) * 0.5
        return torch.cat([center, g_quat], dim=-1)

    @property
    def qpos(self) -> torch.Tensor:
        return self._robot_entity.get_qpos()

    @property
    def gripper_openness(self) -> torch.Tensor:
        """1 = fully open, 0 = fully closed."""
        q = self._robot_entity.get_qpos()[:, self._arm_dof_dim]
        span = max(self._gripper_open_dof - self._gripper_close_dof, 1e-6)
        return ((q - self._gripper_close_dof) / span).clamp(0.0, 1.0)
