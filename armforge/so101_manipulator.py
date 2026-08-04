"""SO-ARM-101 manipulator with joint-space or Cartesian EE + gripper control."""

from __future__ import annotations

from typing import Literal

import torch

import genesis as gs
from genesis.utils.geom import transform_quat_by_quat, xyz_to_quat

from assets import so101_mjcf_path


class SO101Manipulator:
    """5-DoF arm + 1-DoF gripper.

    Default ``control_mode="joint"`` applies arm joint deltas + gripper directly
    (no IK). ``control_mode="ee"`` keeps Cartesian deltas via DLS / Genesis IK.
    """

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
        self._control_mode: Literal["joint", "ee"] = args.get("control_mode", "joint")
        self._ik_method: Literal["gs_ik", "dls_ik"] = args.get("ik_method", "dls_ik")
        self._init()

    def set_pd_gains(self) -> None:
        # STS3215-tuned gains from the MJCF; stiffened + higher force limit so the arm can push the cube.
        kp = torch.tensor([50.0, 50.0, 50.0, 40.0, 25.0, 25.0], device=self._device)
        kv = torch.tensor([2.5, 2.5, 2.5, 2.0, 1.2, 1.2], device=self._device)
        self._robot_entity.set_dofs_kp(kp)
        self._robot_entity.set_dofs_kv(kv)
        force = torch.tensor([8.0, 8.0, 8.0, 5.0, 4.0, 4.0], device=self._device)
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

        self._dls_solve_on_cpu = self._device == "mps" or str(self._device).startswith("mps")
        dls_device = "cpu" if self._dls_solve_on_cpu else self._device
        self._dls_lambda_matrix = (0.05**2) * torch.eye(6, device=dls_device)

    def reset(self, envs_idx=None, skip_forward=True) -> None:
        self._robot_entity.set_qpos(
            self._init_qpos,
            envs_idx=envs_idx,
            zero_velocity=True,
            skip_forward=skip_forward,
        )

    def apply_action(self, action: torch.Tensor) -> None:
        """Apply scaled action: joint deltas (5) + gripper, or EE deltas (6) + gripper."""
        if self._control_mode == "joint":
            q_pos = self._joint_delta(action[:, : self._arm_dof_dim])
            grip_cmd = action[:, self._arm_dof_dim]
        elif self._control_mode == "ee":
            ee_delta = action[:, :6]
            grip_cmd = action[:, 6]
            if self._ik_method == "gs_ik":
                q_pos = self._gs_ik(ee_delta)
            elif self._ik_method == "dls_ik":
                q_pos = self._dls_ik(ee_delta)
            else:
                raise ValueError(f"Invalid IK method: {self._ik_method}")
        else:
            raise ValueError(f"Invalid control mode: {self._control_mode}")

        grip = 0.5 * (grip_cmd + 1.0) * (self._gripper_open_dof - self._gripper_close_dof) + self._gripper_close_dof
        q_pos[:, self._gripper_dof] = grip.unsqueeze(-1)
        self._robot_entity.control_dofs_position(position=q_pos)

    def _joint_delta(self, delta_q: torch.Tensor) -> torch.Tensor:
        """Add arm joint deltas to current qpos (no IK)."""
        q = self._robot_entity.get_qpos().clone()
        q[:, : self._arm_dof_dim] = q[:, : self._arm_dof_dim] + delta_q
        return q

    def _gs_ik(self, action: torch.Tensor) -> torch.Tensor:
        delta_position = action[:, :3]
        delta_orientation = action[:, 3:6]
        target_position = delta_position + self._ee_link.get_pos()
        quat_rel = xyz_to_quat(delta_orientation, rpy=True, degrees=False)
        target_orientation = transform_quat_by_quat(quat_rel, self._ee_link.get_quat())
        return self._robot_entity.inverse_kinematics(
            link=self._ee_link,
            pos=target_position,
            quat=target_orientation,
            dofs_idx_local=self._arm_dof_idx,
        )

    def _dls_ik(self, action: torch.Tensor) -> torch.Tensor:
        delta_pose = action[:, :6]
        jacobian = self._robot_entity.get_jacobian(link=self._ee_link)[..., : self._arm_dof_dim]
        if self._dls_solve_on_cpu:
            jacobian = jacobian.cpu()
            delta_pose = delta_pose.cpu()
        A = torch.baddbmm(self._dls_lambda_matrix, jacobian, jacobian.mT)
        y = torch.linalg.solve(A, delta_pose)
        delta_joint_pos = (jacobian.mT @ y.unsqueeze(-1)).squeeze(-1)
        if self._dls_solve_on_cpu:
            delta_joint_pos = delta_joint_pos.to(self._device)
        q = self._robot_entity.get_qpos().clone()
        q[:, : self._arm_dof_dim] = q[:, : self._arm_dof_dim] + delta_joint_pos
        return q

    def go_to_goal(self, goal_pose: torch.Tensor, open_gripper: bool = True) -> None:
        q_pos = self._robot_entity.inverse_kinematics(
            link=self._ee_link,
            pos=goal_pose[:, :3],
            quat=goal_pose[:, 3:7],
            dofs_idx_local=self._arm_dof_idx,
        )
        grip = self._gripper_open_dof if open_gripper else self._gripper_close_dof
        q_pos[:, self._gripper_dof] = grip
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
