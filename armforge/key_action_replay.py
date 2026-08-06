"""Shared LeRobot key-action load + Genesis Cartesian replay.

Used by ``replay_lerobot.py`` and ``optimize_lerobot.py`` (drop idle frames).
Action layout and teleop deltas must match ``record_lerobot.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch

import genesis as gs
import genesis.utils.geom as gu

from assets import so101_mjcf_path
from configs import get_task_cfgs
from record_lerobot import (
    ACTION_NAMES,
    CUBE_SIZE,
    DISK_HEIGHT,
    DISK_RADIUS,
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    _to_numpy,
)


# Per-tick teleop deltas (must match record_lerobot.py).
DQ = 0.03
DPOS = 0.005
DROT = 0.025

# action_idx -> ("ee"|"rpy"|"grip", payload)
ACTION_EFFECTS = (
    ("ee", (0.0, 0.0, DPOS)),
    ("ee", (0.0, 0.0, -DPOS)),
    ("ee", (0.0, -DPOS, 0.0)),
    ("ee", (0.0, DPOS, 0.0)),
    ("ee", (-DPOS, 0.0, 0.0)),
    ("ee", (DPOS, 0.0, 0.0)),
    ("rpy", (DROT, 0.0, 0.0)),
    ("rpy", (-DROT, 0.0, 0.0)),
    ("rpy", (0.0, DROT, 0.0)),
    ("rpy", (0.0, -DROT, 0.0)),
    ("rpy", (0.0, 0.0, DROT)),
    ("rpy", (0.0, 0.0, -DROT)),
    ("grip", DQ),
    ("grip", -DQ),
)

# Back-compat alias used by older imports.
_ACTION_EFFECTS = ACTION_EFFECTS


def load_actions_npz(path: Path | str) -> np.ndarray:
    """Load ``(T, 14)`` actions from an NPZ with an ``action`` array."""
    path = Path(path)
    data = np.load(path, allow_pickle=True)
    if "action" not in data.files:
        raise ValueError(f"{path} missing 'action' array; has {data.files}")
    acts = np.asarray(data["action"], dtype=np.float32)
    if acts.ndim != 2 or acts.shape[1] != len(ACTION_NAMES):
        raise ValueError(f"Expected action (T, {len(ACTION_NAMES)}), got {acts.shape}")
    return acts


def load_episode_actions(
    repo_id: str,
    episode: int,
    *,
    root: Path | str | None = None,
) -> np.ndarray:
    """Load ``(T, 14)`` multi-hot actions from a local or Hub LeRobot dataset."""
    try:
        from lerobot.datasets import LeRobotDataset
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Loading LeRobot datasets requires lerobot (declared in pyproject.toml).\n"
            "  uv sync\n"
            "Or pass a pre-exported NPZ via optimize_lerobot --actions-npz."
        ) from e
    kwargs: dict = {"repo_id": repo_id}
    if root is not None:
        kwargs["root"] = Path(root)
    ds = LeRobotDataset(**kwargs)
    if episode < 0 or episode >= ds.meta.total_episodes:
        raise IndexError(f"episode {episode} out of range [0, {ds.meta.total_episodes})")
    ep = ds.meta.episodes[episode]
    start = int(ep["dataset_from_index"])
    end = int(ep["dataset_to_index"])
    actions = []
    for i in range(start, end):
        a = ds[i]["action"]
        actions.append(_to_numpy(a).reshape(-1).astype(np.float32))
    acts = np.stack(actions, axis=0)
    if acts.shape[1] != len(ACTION_NAMES):
        raise ValueError(f"Expected action dim {len(ACTION_NAMES)}, got {acts.shape[1]}")
    return acts


# Alias matching previous private name in replay_lerobot.py
_load_episode_actions = load_episode_actions


def cube_on_disk(
    cube_pos: np.ndarray,
    disk_xy: tuple[float, float],
    *,
    table_h: float,
    disk_h: float,
    disk_radius: float,
    cube_half: float,
) -> tuple[bool, dict[str, float]]:
    """Match ``SO101CubeDiskEnv._on_disk``: cube resting on the cylinder top face."""
    cube_pos = np.asarray(cube_pos, dtype=np.float64).reshape(-1)[:3]
    xy_dist = float(np.linalg.norm(cube_pos[:2] - np.asarray(disk_xy, dtype=np.float64)))
    disk_top = table_h + disk_h
    z = float(cube_pos[2])
    z_lo = disk_top + cube_half - 0.01
    z_hi = disk_top + cube_half + 0.025
    xy_ok = xy_dist < disk_radius
    z_ok = z_lo < z < z_hi
    info = {
        "xy_dist": xy_dist,
        "cube_z": z,
        "disk_top": disk_top,
        "z_lo": z_lo,
        "z_hi": z_hi,
        "disk_radius": disk_radius,
    }
    return bool(xy_ok and z_ok), info


_cube_on_disk = cube_on_disk


def report_success(ok: bool, info: dict[str, float], episode: int) -> None:
    status = "SUCCESS" if ok else "FAILURE"
    print(
        f"Episode {episode} {status}: cube on cylinder="
        f"{ok} (xy_dist={info['xy_dist']:.4f}m < {info['disk_radius']:.4f}m, "
        f"z={info['cube_z']:.4f}m in ({info['z_lo']:.4f}, {info['z_hi']:.4f}))"
    )


_report_success = report_success


def task_geometry() -> dict:
    """Shared cube/disk/table geometry from task configs."""
    env_cfg, _, _ = get_task_cfgs("cube_disk")
    table_h = float(env_cfg["table_height"])
    disk_h = float(env_cfg.get("disk_height", DISK_HEIGHT))
    disk_radius = float(env_cfg.get("disk_radius", DISK_RADIUS))
    box = env_cfg.get("box_size", [CUBE_SIZE, CUBE_SIZE, CUBE_SIZE])
    cube_size = float(box[2] if len(box) > 2 else box[0])
    return {
        "table_h": table_h,
        "disk_h": disk_h,
        "disk_radius": disk_radius,
        "cube_size": cube_size,
        "cube_half": 0.5 * cube_size,
        "cube_xy": tuple(env_cfg.get("cube_pos_xy", (0.18, 0.0))),
        "disk_xy": tuple(env_cfg.get("disk_pos_xy", (0.24, 0.08))),
    }


class BatchedKeyActionScene:
    """Genesis scene with ``n_envs`` parallel Cartesian key-action controllers."""

    def __init__(
        self,
        num_envs: int,
        *,
        fps: int = 30,
        show_viewer: bool = False,
        add_rgb_triad: bool = False,
        video_path: Path | str | None = None,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        self.num_envs = int(num_envs)
        self.fps = int(fps)
        self.geom = task_geometry()
        self.video_path = Path(video_path) if video_path is not None else None
        table_h = self.geom["table_h"]
        cube_half = self.geom["cube_half"]
        cube_size = self.geom["cube_size"]
        cube_xy = self.geom["cube_xy"]
        disk_xy = self.geom["disk_xy"]
        disk_h = self.geom["disk_h"]
        disk_radius = self.geom["disk_radius"]

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=1.0 / self.fps, substeps=8),
            rigid_options=gs.options.RigidOptions(
                enable_joint_limit=True,
                enable_collision=True,
                gravity=(0, 0, -9.8),
                noslip_iterations=8,
            ),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=list(range(min(10, self.num_envs))),
                env_separate_rigid=True,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(0.75, -0.55, 0.45),
                camera_lookat=(0.18, 0.0, table_h + 0.05),
                camera_fov=50,
            ),
            profiling_options=gs.options.ProfilingOptions(show_FPS=False),
            show_viewer=show_viewer,
        )
        self.scene.add_entity(gs.morphs.Plane())
        self.robot = self.scene.add_entity(
            material=gs.materials.Rigid(gravity_compensation=1),
            morph=gs.morphs.MJCF(
                file=str(so101_mjcf_path()),
                pos=(0.0, 0.0, table_h),
                convexify=True,
                decompose_robot_error_threshold=0.0,
            ),
        )
        self.cube = self.scene.add_entity(
            material=gs.materials.Rigid(rho=500, friction=1.5),
            morph=gs.morphs.Box(
                pos=(float(cube_xy[0]), float(cube_xy[1]), table_h + cube_half),
                size=(cube_size, cube_size, cube_size),
                batch_fixed_verts=True,
            ),
            surface=gs.surfaces.Default(color=(0.9, 0.2, 0.1)),
        )
        self.disk = self.scene.add_entity(
            gs.morphs.Cylinder(
                pos=(float(disk_xy[0]), float(disk_xy[1]), table_h + 0.5 * disk_h),
                radius=disk_radius,
                height=disk_h,
                fixed=True,
            ),
            surface=gs.surfaces.Default(color=(0.2, 0.5, 0.9)),
        )
        self.rgb_axes = None
        if add_rgb_triad:
            from record_lerobot import _add_rgb_triad

            self.rgb_axes = _add_rgb_triad(self.scene)

        self.episode_cam = None
        if self.video_path is not None:
            from functools import partial

            from genesis.options.sensors import RasterizerCameraOptions

            self.episode_cam = self.scene.add_sensor(
                RasterizerCameraOptions(
                    res=(1280, 960),
                    pos=(0.9, 0.0, table_h + 0.35),
                    lookat=(0.18, 0.0, table_h + 0.05),
                    fov=50,
                )
            )
            self.video_path.parent.mkdir(parents=True, exist_ok=True)

            def _read_cam_rgb(cam) -> np.ndarray:
                rgb = _to_numpy(cam.read(envs_idx=0).rgb)
                if rgb.ndim == 4:
                    rgb = rgb[0]
                return np.asarray(rgb[..., :3], dtype=np.uint8)

            self.scene.start_recording(
                data_func=partial(_read_cam_rgb, self.episode_cam),
                rec_options=gs.recorders.VideoFile(filename=str(self.video_path), fps=self.fps),
            )

        self.scene.build(n_envs=self.num_envs, env_spacing=(1.0, 1.0))

        self.all_dofs = np.arange(6)
        self.motors_dof = np.arange(5)
        self.ee_link = self.robot.get_link("gripper")
        try:
            lim = _to_numpy(self.robot.get_dofs_limit(self.all_dofs))
            lim = np.asarray(lim, dtype=np.float64)
            if lim.shape[0] == 2:
                self.q_lo, self.q_hi = lim[0], lim[1]
            else:
                self.q_lo, self.q_hi = lim[:, 0], lim[:, 1]
        except Exception:
            self.q_lo = np.array([-1.92, -1.75, -1.69, -1.69, -2.74, GRIPPER_CLOSE], dtype=np.float64)
            self.q_hi = np.array([1.92, 1.75, 1.69, 1.69, 2.74, GRIPPER_OPEN], dtype=np.float64)

        b = self.num_envs
        # Cartesian / joint targets live on-device to avoid per-step D2H sync.
        self.q_lo_t = torch.as_tensor(self.q_lo, device=gs.device, dtype=gs.tc_float)
        self.q_hi_t = torch.as_tensor(self.q_hi, device=gs.device, dtype=gs.tc_float)
        # Device targets — IK / control / post never ``_to_numpy`` the sim state.
        self.target_q = torch.zeros((b, 6), device=gs.device, dtype=gs.tc_float)
        self.target_pos = torch.zeros((b, 3), device=gs.device, dtype=gs.tc_float)
        self.target_quat = torch.zeros((b, 4), device=gs.device, dtype=gs.tc_float)
        self.target_quat[:, 0] = 1.0
        self.orient_cmd = np.zeros((b,), dtype=bool)
        self.joint_cmd = np.zeros((b,), dtype=bool)
        self._active = np.ones((b,), dtype=bool)
        self._physics_failed = np.zeros((b,), dtype=bool)
        self._ee_deltas = torch.zeros((6, 3), device=gs.device, dtype=gs.tc_float)
        self._rpy_deltas = torch.zeros((len(ACTION_EFFECTS), 3), device=gs.device, dtype=gs.tc_float)
        self._grip_deltas = torch.zeros((len(ACTION_EFFECTS),), device=gs.device, dtype=gs.tc_float)
        self._action_kinds = []
        for i, (kind, payload) in enumerate(ACTION_EFFECTS):
            self._action_kinds.append(kind)
            if kind == "ee":
                self._ee_deltas[i] = torch.as_tensor(payload, device=gs.device, dtype=gs.tc_float)
            elif kind == "rpy":
                self._rpy_deltas[i] = torch.as_tensor(payload, device=gs.device, dtype=gs.tc_float)
            else:
                self._grip_deltas[i] = float(payload)
        self.cube_xy = np.broadcast_to(
            np.asarray(self.geom["cube_xy"], dtype=np.float64).reshape(1, 2), (b, 2)
        ).copy()
        self.cube_yaw = np.zeros((b,), dtype=np.float64)
        self.disk_xy = np.broadcast_to(
            np.asarray(self.geom["disk_xy"], dtype=np.float64).reshape(1, 2), (b, 2)
        ).copy()
        self.profile: dict[str, float] | None = None
        self.reset_all()

    def enable_profile(self) -> None:
        self.profile = {
            "apply_s": 0.0,
            "ik_s": 0.0,
            "control_s": 0.0,
            "physics_s": 0.0,
            "sync_s": 0.0,
            "steps": 0.0,
        }

    def reset_all(
        self,
        *,
        cube_xy: np.ndarray | None = None,
        cube_yaw: np.ndarray | None = None,
        disk_xy: np.ndarray | None = None,
    ) -> None:
        self.reset_mask(
            np.ones(self.num_envs, dtype=bool),
            cube_xy=cube_xy,
            cube_yaw=cube_yaw,
            disk_xy=disk_xy,
        )

    def reset_mask(
        self,
        mask: np.ndarray,
        *,
        cube_xy: np.ndarray | None = None,
        cube_yaw: np.ndarray | None = None,
        disk_xy: np.ndarray | None = None,
    ) -> None:
        """Reset selected envs.

        Optional per-env poses (full ``num_envs`` length, only ``mask`` rows applied):
        - ``cube_xy``: ``(B, 2)`` meters
        - ``cube_yaw``: ``(B,)`` radians about +Z
        - ``disk_xy``: ``(B, 2)`` meters (cylinder)
        """
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        if mask.shape[0] != self.num_envs:
            raise ValueError(f"mask length {mask.shape[0]} != num_envs {self.num_envs}")
        envs_idx = np.nonzero(mask)[0]
        if envs_idx.size == 0:
            return
        table_h = self.geom["table_h"]
        cube_half = self.geom["cube_half"]
        disk_h = self.geom["disk_h"]

        if cube_xy is None:
            cube_xy_full = self.cube_xy
        else:
            cube_xy_full = np.asarray(cube_xy, dtype=np.float64).reshape(self.num_envs, 2)
            self.cube_xy[mask] = cube_xy_full[mask]
        if cube_yaw is None:
            yaw_full = self.cube_yaw
        else:
            yaw_full = np.asarray(cube_yaw, dtype=np.float64).reshape(self.num_envs)
            self.cube_yaw[mask] = yaw_full[mask]
        if disk_xy is None:
            disk_xy_full = self.disk_xy
        else:
            disk_xy_full = np.asarray(disk_xy, dtype=np.float64).reshape(self.num_envs, 2)
            self.disk_xy[mask] = disk_xy_full[mask]

        q6 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, GRIPPER_CLOSE], dtype=np.float32)
        q6 = np.clip(q6, self.q_lo, self.q_hi).astype(np.float32)
        q_batch = np.broadcast_to(q6, (envs_idx.size, 6)).copy()

        cube_pos = np.zeros((envs_idx.size, 3), dtype=np.float32)
        cube_pos[:, 0] = cube_xy_full[envs_idx, 0]
        cube_pos[:, 1] = cube_xy_full[envs_idx, 1]
        cube_pos[:, 2] = table_h + cube_half

        # Genesis quat is (w, x, y, z); yaw about +Z via RPY.
        yaw_batch = yaw_full[envs_idx].astype(np.float32)
        rpy = np.zeros((envs_idx.size, 3), dtype=np.float32)
        rpy[:, 2] = yaw_batch
        cube_quat = np.asarray(gu.xyz_to_quat(rpy, rpy=True), dtype=np.float32).reshape(envs_idx.size, 4)

        disk_pos = np.zeros((envs_idx.size, 3), dtype=np.float32)
        disk_pos[:, 0] = disk_xy_full[envs_idx, 0]
        disk_pos[:, 1] = disk_xy_full[envs_idx, 1]
        disk_pos[:, 2] = table_h + 0.5 * disk_h

        # Genesis expects tensor / array with env indexing.
        idx_t = torch.as_tensor(envs_idx, device=gs.device, dtype=torch.long)
        self.robot.set_dofs_position(
            torch.as_tensor(q_batch, device=gs.device, dtype=gs.tc_float),
            self.all_dofs,
            envs_idx=idx_t,
        )
        self.cube.set_pos(
            torch.as_tensor(cube_pos, device=gs.device, dtype=gs.tc_float),
            envs_idx=idx_t,
        )
        self.cube.set_quat(
            torch.as_tensor(cube_quat, device=gs.device, dtype=gs.tc_float),
            envs_idx=idx_t,
        )
        self.disk.set_pos(
            torch.as_tensor(disk_pos, device=gs.device, dtype=gs.tc_float),
            envs_idx=idx_t,
        )

        # Sync Cartesian targets from current EE pose on-device (no host copy in step).
        pos = self.ee_link.get_pos().reshape(self.num_envs, 3)
        quat = self.ee_link.get_quat().reshape(self.num_envs, 4)
        qpos = self.robot.get_dofs_position(self.all_dofs).reshape(self.num_envs, 6)
        mask_t = torch.as_tensor(mask, device=gs.device, dtype=torch.bool)
        self.target_q[mask_t] = qpos[mask_t].to(dtype=gs.tc_float)
        self.target_pos[mask_t] = pos[mask_t].to(dtype=gs.tc_float)
        self.target_quat[mask_t] = quat[mask_t].to(dtype=gs.tc_float)
        self.orient_cmd[mask] = False
        self.joint_cmd[mask] = False
        self._active[mask] = True
        self._physics_failed[mask] = False

    def _solver_error_mask(self) -> np.ndarray:
        """Boolean ``(B,)`` of envs with rigid-solver errno or NaN state."""
        b = self.num_envs
        bad = np.zeros(b, dtype=bool)
        try:
            rigid = self.scene.sim.rigid_solver
            mask = rigid.get_error_envs_mask()
            if torch.is_tensor(mask):
                bad |= mask.detach().cpu().numpy().astype(bool).reshape(b)
            else:
                bad |= np.asarray(mask, dtype=bool).reshape(b)
        except Exception:
            pass
        try:
            q = self.robot.get_dofs_position(self.all_dofs).reshape(b, -1)
            if torch.is_tensor(q):
                bad |= torch.isnan(q).any(dim=-1).detach().cpu().numpy()
            else:
                bad |= np.isnan(np.asarray(q).reshape(b, -1)).any(axis=1)
        except Exception:
            pass
        try:
            cpos = self.cube.get_pos().reshape(b, -1)
            if torch.is_tensor(cpos):
                bad |= torch.isnan(cpos).any(dim=-1).detach().cpu().numpy()
            else:
                bad |= np.isnan(np.asarray(cpos).reshape(b, -1)).any(axis=1)
        except Exception:
            pass
        return bad

    def quarantine_physics_failures(self) -> np.ndarray:
        """Reset + deactivate envs with solver NaN/errno. Returns newly failed mask.

        Genesis raises on the *next* ``scene.step`` if any env has errno set. Clearing
        via ``set_dofs_position`` (inside ``reset_mask``) zeroes errno so the batch can
        continue; quarantined envs count as failures.
        """
        newly = self._solver_error_mask() & ~self._physics_failed
        if not newly.any():
            return newly
        self._physics_failed[newly] = True
        # reset_mask clears errno and would re-activate — force inactive after.
        self.reset_mask(newly)
        self._active[newly] = False
        self._physics_failed[newly] = True
        n = int(newly.sum())
        ids = np.nonzero(newly)[0][:16].tolist()
        print(f"[physics] quarantined {n} env(s): {ids}{'…' if n > 16 else ''}", flush=True)
        return newly

    def _apply_action_env(self, env_i: int, action: np.ndarray) -> None:
        """Apply RPY / gripper bits to on-device targets for one env (EE is batched)."""
        for bit_i in range(6, len(ACTION_EFFECTS)):
            if action[bit_i] < 0.5:
                continue
            kind = self._action_kinds[bit_i]
            if kind == "rpy":
                dq_r = gu.xyz_to_quat(self._rpy_deltas[bit_i], rpy=True)
                self.target_quat[env_i] = gu.transform_quat_by_quat(dq_r, self.target_quat[env_i]).reshape(4)
                self.orient_cmd[env_i] = True
            else:
                self.target_q[env_i, 5] = torch.clamp(
                    self.target_q[env_i, 5] + self._grip_deltas[bit_i],
                    self.q_lo_t[5],
                    self.q_hi_t[5],
                )
                self.joint_cmd[env_i] = True

    def _apply_ee_batch(self, actions_t: torch.Tensor, active_t: torch.Tensor) -> None:
        """Batched EE-frame translations (action bits 0..5) on-device."""
        d_local = actions_t[:, :6] @ self._ee_deltas
        R = gu.quat_to_R(self.target_quat).reshape(self.num_envs, 3, 3)
        d_world = torch.bmm(R, d_local.unsqueeze(-1)).squeeze(-1)
        self.target_pos = self.target_pos + d_world * active_t.to(dtype=gs.tc_float).unsqueeze(-1)

    def step(self, actions: np.ndarray | None = None, *, active: np.ndarray | None = None) -> None:
        """Step all envs. ``actions`` shape ``(B, 14)``; None / zeros = idle tick.

        Envs with ``active[i] == False`` keep holding their last command (frozen).
        Hot path keeps targets / IK / post on ``gs.device`` (no per-step ``_to_numpy``).
        """
        import time as _time

        def _sync_device() -> None:
            if self.profile is None:
                return
            dev = gs.device
            if isinstance(dev, torch.device) and dev.type == "cuda":
                torch.cuda.synchronize(dev)
            elif str(dev).startswith("cuda"):
                torch.cuda.synchronize()

        b = self.num_envs
        prof = self.profile
        if active is None:
            active = self._active
        else:
            active = np.asarray(active, dtype=bool).reshape(b)
            self._active = active

        if actions is None:
            actions = np.zeros((b, len(ACTION_NAMES)), dtype=np.float32)
        else:
            actions = np.asarray(actions, dtype=np.float32).reshape(b, -1)

        t0 = _time.perf_counter() if prof is not None else 0.0
        active_t = torch.as_tensor(active, device=gs.device, dtype=torch.bool)
        actions_t = torch.as_tensor(actions, device=gs.device, dtype=gs.tc_float)
        self._apply_ee_batch(actions_t, active_t)
        for i in range(b):
            if not active[i]:
                continue
            act = actions[i]
            if act[6:].any():
                self._apply_action_env(i, act)

        use_orient = self.orient_cmd.copy()
        use_joint = self.joint_cmd.copy()
        self.orient_cmd[active] = False
        self.joint_cmd[active] = False
        _sync_device()
        t1 = _time.perf_counter() if prof is not None else 0.0

        ik_kwargs = dict(
            link=self.ee_link,
            pos=self.target_pos,
            dofs_idx_local=self.motors_dof,
            max_samples=1,
            max_solver_iters=50,
            damping=0.05,
            pos_tol=1e-3,
            rot_tol=1e-2,
        )
        if np.any(active) and np.any(use_orient[active]):
            ik_kwargs["quat"] = self.target_quat
        q_arm = self.robot.inverse_kinematics(**ik_kwargs).reshape(self.num_envs, -1)
        _sync_device()
        t2 = _time.perf_counter() if prof is not None else 0.0

        q_cmd = self.target_q.clone()
        q_arm_clipped = torch.clamp(q_arm[:, :5], self.q_lo_t[:5], self.q_hi_t[:5])
        q_cmd[:, :5] = torch.where(active_t.unsqueeze(-1), q_arm_clipped, q_cmd[:, :5])

        self.robot.control_dofs_position(q_cmd, self.all_dofs)
        _sync_device()
        t3 = _time.perf_counter() if prof is not None else 0.0
        try:
            self.scene.step()
        except Exception:
            # Genesis check_errno raises at the *start* of this step when a prior
            # tick left INVALID_*_NAN on any env. Quarantine those envs (reset
            # clears errno) and count them as failures, then finish this tick.
            newly = self.quarantine_physics_failures()
            if not newly.any():
                raise
            self.scene.step()
        _sync_device()
        t4 = _time.perf_counter() if prof is not None else 0.0

        link_pos = self.ee_link.get_pos().reshape(self.num_envs, 3)
        link_quat = self.ee_link.get_quat().reshape(self.num_envs, 4)
        q_now = self.robot.get_dofs_position(self.all_dofs).reshape(self.num_envs, 6)
        self.target_q = q_cmd

        if self.rgb_axes is not None:
            from record_lerobot import _sync_rgb_triad

            _sync_rgb_triad(
                self.rgb_axes,
                link_pos[0].detach().cpu().numpy(),
                link_quat[0].detach().cpu().numpy(),
            )

        use_joint_t = torch.as_tensor(use_joint, device=gs.device, dtype=torch.bool)
        use_orient_t = torch.as_tensor(use_orient, device=gs.device, dtype=torch.bool)
        joint_m = active_t & use_joint_t
        self.target_pos = torch.where(joint_m.unsqueeze(-1), link_pos, self.target_pos)
        self.target_quat = torch.where(joint_m.unsqueeze(-1), link_quat, self.target_quat)
        arm_m = joint_m.unsqueeze(-1).expand(-1, 5)
        self.target_q[:, :5] = torch.where(arm_m, q_now[:, :5], self.target_q[:, :5])

        cart_m = active_t & ~use_joint_t
        pos_err = torch.linalg.norm(link_pos - self.target_pos, dim=-1)
        snap_pos = cart_m & (pos_err > 0.04)
        self.target_pos = torch.where(snap_pos.unsqueeze(-1), link_pos, self.target_pos)
        snap_quat = cart_m & ~use_orient_t
        self.target_quat = torch.where(snap_quat.unsqueeze(-1), link_quat, self.target_quat)
        _sync_device()
        t5 = _time.perf_counter() if prof is not None else 0.0

        if prof is not None:
            prof["apply_s"] += t1 - t0
            prof["ik_s"] += t2 - t1
            prof["control_s"] += t3 - t2
            prof["physics_s"] += t4 - t3
            prof["sync_s"] += t5 - t4
            prof["steps"] += 1.0

    def success_mask(self) -> np.ndarray:
        """Per-env cube-on-disk success ``(B,)``. Physics-failed envs are False."""
        cube_pos = _to_numpy(self.cube.get_pos()).reshape(self.num_envs, 3)
        out = np.zeros(self.num_envs, dtype=bool)
        for i in range(self.num_envs):
            if self._physics_failed[i]:
                continue
            ok, _ = cube_on_disk(
                cube_pos[i],
                (float(self.disk_xy[i, 0]), float(self.disk_xy[i, 1])),
                table_h=self.geom["table_h"],
                disk_h=self.geom["disk_h"],
                disk_radius=self.geom["disk_radius"],
                cube_half=self.geom["cube_half"],
            )
            out[i] = ok
        return out

    def physics_failed_mask(self) -> np.ndarray:
        return self._physics_failed.copy()

    def success_info(self, env_i: int = 0) -> tuple[bool, dict[str, float]]:
        cube_pos = _to_numpy(self.cube.get_pos()).reshape(self.num_envs, 3)[env_i]
        disk_xy = (float(self.disk_xy[env_i, 0]), float(self.disk_xy[env_i, 1]))
        return cube_on_disk(
            cube_pos,
            disk_xy,
            table_h=self.geom["table_h"],
            disk_h=self.geom["disk_h"],
            disk_radius=self.geom["disk_radius"],
            cube_half=self.geom["cube_half"],
        )

    def settle_frames(self) -> int:
        return max(1, int(0.3 * self.fps))

    def evaluate_subsequences(
        self,
        full_actions: np.ndarray,
        keep_lists: Sequence[Sequence[int]],
        *,
        settle_frames: int | None = None,
    ) -> np.ndarray:
        """Replay each keep-index list in parallel; return success ``(len(keep_lists),)``.

        Pads to ``num_envs`` with inactive slots. Different lengths are lockstep-padded
        with idle ticks; each env's success is sampled after its own actions + settle.
        """
        full_actions = np.asarray(full_actions, dtype=np.float32)
        n = len(keep_lists)
        if n == 0:
            return np.zeros(0, dtype=bool)
        if n > self.num_envs:
            raise ValueError(f"Need {n} candidates but scene has only {self.num_envs} envs")

        settle = self.settle_frames() if settle_frames is None else max(0, int(settle_frames))
        lengths = [len(k) for k in keep_lists]
        max_len = max(lengths) if lengths else 0
        horizon = max_len + settle

        self.reset_all()
        finished = np.zeros(self.num_envs, dtype=bool)
        finished[n:] = True  # unused env slots stay frozen
        success = np.zeros(n, dtype=bool)

        zero = np.zeros((self.num_envs, full_actions.shape[1]), dtype=np.float32)
        for t in range(horizon):
            batch = zero.copy()
            stepping = np.zeros(self.num_envs, dtype=bool)
            for i, keep in enumerate(keep_lists):
                if finished[i]:
                    continue
                if t < lengths[i]:
                    batch[i] = full_actions[int(keep[t])]
                    stepping[i] = True
                elif t < lengths[i] + settle:
                    stepping[i] = True  # idle settle ticks
            self.step(batch, active=stepping)
            for i in range(n):
                if finished[i]:
                    continue
                if t + 1 >= lengths[i] + settle:
                    success[i] = bool(self.success_mask()[i])
                    finished[i] = True
            if finished[:n].all():
                break

        return success

    def stop_recording(self) -> None:
        if self.video_path is not None:
            self.scene.stop_recording()
