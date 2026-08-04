"""SO-ARM-101 manipulator with joint-space arm deltas + gripper control."""

from __future__ import annotations

import torch

import genesis as gs

from assets import so101_mjcf_path


class SO101Manipulator:
    """5-DoF arm + 1-DoF gripper controlled via joint deltas (pick-and-place)."""

    def __init__(self, num_envs: int, scene: gs.Scene, args: dict, device: str = "cpu"):
        self._device = device
        self._scene = scene
        self._num_envs = num_envs
        self._args = args

        mjcf = args.get("mjcf_path") or str(so101_mjcf_path())
        morph = gs.morphs.MJCF(
            file=mjcf,
            pos=tuple(args.get("base_pos", (0.0, 0.0, 0.0))),
            quat=tuple(args.get("base_quat", (1.0, 0.0, 0.0, 0.0))),
        )
        self._robot_entity = scene.add_entity(
            material=gs.materials.Rigid(gravity_compensation=1.0),
            morph=morph,
        )

        # Gripper hinge: closed near lower bound, open near upper bound (MJCF range).
        self._gripper_open_dof = float(args.get("gripper_open", 1.7))
        self._gripper_close_dof = float(args.get("gripper_close", 0.0))
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
        self._ee_link = self._robot_entity.get_link(self._args.get("ee_link_name", "gripper"))
        jaw_name = self._args.get("jaw_link_name", "moving_jaw_so101_v1")
        self._jaw_link = self._robot_entity.get_link(jaw_name)

        default_arm = self._args.get("default_arm_dof", [0.0, -0.9, 1.1, 0.9, 0.0])
        default_grip = self._args.get("default_gripper_dof", [self._gripper_open_dof])
        self._init_qpos = torch.tensor(default_arm + default_grip, dtype=torch.float32, device=self._device)

    def reset(self, envs_idx=None, skip_forward=True) -> None:
        self._robot_entity.set_qpos(
            self._init_qpos,
            envs_idx=envs_idx,
            zero_velocity=True,
            skip_forward=skip_forward,
        )

    def apply_action(self, action: torch.Tensor) -> None:
        """Apply scaled arm joint deltas (5) + absolute gripper command (1) in [-1, 1]."""
        q_pos = self._robot_entity.get_qpos().clone()
        q_pos[:, : self._arm_dof_dim] = q_pos[:, : self._arm_dof_dim] + action[:, : self._arm_dof_dim]
        grip_cmd = action[:, self._arm_dof_dim]
        grip = 0.5 * (grip_cmd + 1.0) * (self._gripper_open_dof - self._gripper_close_dof) + self._gripper_close_dof
        q_pos[:, self._gripper_dof] = grip.unsqueeze(-1)
        self._robot_entity.control_dofs_position(position=q_pos)

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
