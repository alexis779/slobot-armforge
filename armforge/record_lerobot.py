"""Record SO-101 pick-and-place teleop as key-press action sequences.

Controls (Genesis viewer keybinds)
---------------------------------
End-effector (EE-frame translate; body-frame RPY)
  Arrow keys      : Up/Down = ±Z, Left/Right = ±Y
  N / M           : ∓X / ±X
  Q / E           : roll  ±
  T / G           : pitch ±
  Y / B           : yaw   ±

Gripper
  6 / ^           : open / close (Shift+6 on a US keyboard)

R                 : reset cube/disk and home pose (discards current key buffer)
Enter             : save episode (flush key buffer → LeRobot) and start a new one
Esc               : quit (finalizes the LeRobot dataset)

Avoids Genesis viewer defaults: I (instructions), O (camera mode), H (shadow).

Cartesian is primary: every frame the arm IK-tracks the EE target.
Home is all zeros with gripper closed.
An RGB triad (R=X, G=Y, B=Z) is drawn at the gripper link pose.

Recording
---------
Simulation runs every frame for teleop. Each sim step appends one multi-hot
``action`` vector (all zeros when no control key is held), so replay keeps the
same timing / pauses as the demonstration. Enter flushes that buffer into a
LeRobot episode.

Example
-------
  uv sync
  python armforge/record_lerobot.py --repo-id local/so101_cube_disk --root datasets/so101_cube_disk
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse

import numpy as np

import genesis as gs
import genesis.utils.geom as gu
from genesis.vis.keybindings import Key, KeyAction, Keybind

from assets import so101_mjcf_path
from backend import add_backend_arg, init_genesis
from configs import get_task_cfgs


CUBE_SIZE = 0.03
DISK_RADIUS = CUBE_SIZE  # diameter = 2× cube side
DISK_HEIGHT = 0.5 * CUBE_SIZE  # height = half cube side
# Full MJCF gripper hinge range.
GRIPPER_OPEN = 1.745
GRIPPER_CLOSE = -0.174
AXIS_LEN = 0.05
AXIS_THICK = 0.004

# Multi-hot action layout (policy action space = teleop keys).
ACTION_NAMES = (
    "move_+z",
    "move_-z",
    "move_-y",
    "move_+y",
    "move_-x",
    "move_+x",
    "roll_+",
    "roll_-",
    "pitch_+",
    "pitch_-",
    "yaw_+",
    "yaw_-",
    "grip_open",
    "grip_close",
)


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _ee_pose(ee_link) -> np.ndarray:
    pos = _to_numpy(ee_link.get_pos()).reshape(-1)[:3]
    quat = _to_numpy(ee_link.get_quat()).reshape(-1)[:4]
    return np.concatenate([pos, quat]).astype(np.float32)


def _add_rgb_triad(scene: gs.Scene) -> list:
    """Three thin boxes (R/G/B = X/Y/Z). Poses updated each frame to the EE link."""
    specs = [
        ((1.0, 0.1, 0.1), (AXIS_LEN, AXIS_THICK, AXIS_THICK)),  # X red
        ((0.1, 1.0, 0.1), (AXIS_THICK, AXIS_LEN, AXIS_THICK)),  # Y green
        ((0.1, 0.1, 1.0), (AXIS_THICK, AXIS_THICK, AXIS_LEN)),  # Z blue
    ]
    axes = []
    for color, size in specs:
        axes.append(
            scene.add_entity(
                gs.morphs.Box(pos=(0.0, 0.0, 0.5), size=size, collision=False, fixed=False),
                surface=gs.surfaces.Default(color=(*color, 1.0)),
            )
        )
    return axes


def _sync_rgb_triad(axes: list, pos: np.ndarray, quat: np.ndarray) -> None:
    """Place triad so its origin is at ``pos`` and axes align with ``quat`` (link frame)."""
    R = np.asarray(gu.quat_to_R(quat), dtype=np.float64).reshape(3, 3)
    p = np.asarray(pos, dtype=np.float64).reshape(3)
    local_centers = np.array(
        [
            [0.5 * AXIS_LEN, 0.0, 0.0],
            [0.0, 0.5 * AXIS_LEN, 0.0],
            [0.0, 0.0, 0.5 * AXIS_LEN],
        ],
        dtype=np.float64,
    )
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    for axis_ent, local in zip(axes, local_centers):
        world = p + R @ local
        axis_ent.set_pos(world)
        axis_ent.set_quat(q)


def _dataset_features() -> dict:
    return {
        "action": {
            "dtype": "float32",
            "shape": (len(ACTION_NAMES),),
            "names": list(ACTION_NAMES),
        },
    }


def main() -> None:
    try:
        from lerobot.datasets import LeRobotDataset
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "This script requires lerobot (declared in pyproject.toml).\n"
            "  uv sync"
        ) from e

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", type=str, default="local/so101_cube_disk")
    parser.add_argument("--root", type=str, default="datasets/so101_cube_disk")
    parser.add_argument("--task", type=str, default="pick the cube and place it on the disk")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--resume", action="store_true", help="Append to an existing dataset at --root")
    add_backend_arg(parser)
    args = parser.parse_args()

    env_cfg, _, _ = get_task_cfgs("cube_disk")
    table_h = float(env_cfg["table_height"])
    disk_h = float(env_cfg.get("disk_height", DISK_HEIGHT))
    disk_radius = float(env_cfg.get("disk_radius", DISK_RADIUS))
    box = env_cfg.get("box_size", [CUBE_SIZE, CUBE_SIZE, CUBE_SIZE])
    cube_size = float(box[2] if len(box) > 2 else box[0])
    cube_half = 0.5 * cube_size
    cube_xy = tuple(env_cfg.get("cube_pos_xy", (0.18, 0.0)))
    disk_xy = tuple(env_cfg.get("disk_pos_xy", (0.24, 0.08)))

    init_genesis(backend=args.backend, performance_mode=False, logging_level="info")

    root = Path(args.root)
    meta_info = root / "meta" / "info.json"
    if args.resume:
        if not meta_info.is_file():
            raise FileNotFoundError(f"--resume set but no dataset meta at {meta_info}")
        dataset = LeRobotDataset.resume(repo_id=args.repo_id, root=root, image_writer_threads=0)
        print(f"Resuming LeRobot dataset at {root} ({dataset.meta.total_episodes} episodes so far)")
    else:
        if root.exists():
            raise FileExistsError(
                f"Dataset root {root} already exists. "
                "Pass --resume to append, or remove/choose a new --root to start fresh."
            )
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            fps=args.fps,
            features=_dataset_features(),
            root=root,
            robot_type="so101",
            use_videos=False,
            image_writer_threads=0,
        )
        print(f"Created LeRobot dataset at {root}")
    print(f"Action space ({len(ACTION_NAMES)}): {list(ACTION_NAMES)}")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1.0 / args.fps, substeps=8),
        rigid_options=gs.options.RigidOptions(
            enable_joint_limit=True,
            enable_collision=True,
            gravity=(0, 0, -9.8),
            noslip_iterations=8,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.75, -0.55, 0.45),
            camera_lookat=(0.18, 0.0, table_h + 0.05),
            camera_fov=50,
        ),
        profiling_options=gs.options.ProfilingOptions(show_FPS=False),
        show_viewer=True,
    )
    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(
        material=gs.materials.Rigid(gravity_compensation=1),
        morph=gs.morphs.MJCF(
            file=str(so101_mjcf_path()),
            pos=(0.0, 0.0, table_h),
            convexify=True,
            # Default robot threshold is inf (hull-only). 0 forces CoACD on every link.
            decompose_robot_error_threshold=0.0,
        ),
    )
    cube = scene.add_entity(
        material=gs.materials.Rigid(rho=500, friction=1.5),
        morph=gs.morphs.Box(
            pos=(float(cube_xy[0]), float(cube_xy[1]), table_h + cube_half),
            size=(cube_size, cube_size, cube_size),
        ),
        surface=gs.surfaces.Default(color=(0.9, 0.2, 0.1)),
    )
    disk = scene.add_entity(
        gs.morphs.Cylinder(
            pos=(float(disk_xy[0]), float(disk_xy[1]), table_h + 0.5 * disk_h),
            radius=disk_radius,
            height=disk_h,
            fixed=True,
        ),
        surface=gs.surfaces.Default(color=(0.2, 0.5, 0.9)),
    )
    rgb_axes = _add_rgb_triad(scene)
    scene.build()

    all_dofs = np.arange(6)
    motors_dof = np.arange(5)
    ee_link = robot.get_link("gripper")

    dq = 0.03  # rad per HOLD tick (joints)
    dpos = 0.005
    drot = 0.025
    try:
        lim = _to_numpy(robot.get_dofs_limit(all_dofs))
        lim = np.asarray(lim, dtype=np.float64)
        if lim.shape[0] == 2:
            q_lo, q_hi = lim[0], lim[1]
        else:
            q_lo, q_hi = lim[:, 0], lim[:, 1]
    except Exception:
        q_lo = np.array([-1.92, -1.75, -1.69, -1.69, -2.74, GRIPPER_CLOSE], dtype=np.float64)
        q_hi = np.array([1.92, 1.75, 1.69, 1.69, 2.74, GRIPPER_OPEN], dtype=np.float64)

    def go_home():
        """Zero arm joints, gripper closed. Returns 6-DoF q and EE pose."""
        q6 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, GRIPPER_CLOSE], dtype=np.float32)
        q6 = np.clip(q6, q_lo, q_hi).astype(np.float32)
        robot.set_dofs_position(q6, all_dofs)
        pose = _ee_pose(ee_link)
        return q6.copy(), pose[:3].copy(), pose[3:7].copy()

    target_q, target_pos, target_quat = go_home()
    ee_cmd = False
    orient_cmd = False
    joint_cmd = False
    is_running = True
    # held[i] set by keybind callbacks during scene.step(); snapshotted after.
    held = np.zeros(len(ACTION_NAMES), dtype=np.float32)
    action_buffer: list[np.ndarray] = []
    request_reset = False
    request_save = False

    def apply_reset():
        nonlocal target_q, target_pos, target_quat, ee_cmd, orient_cmd, joint_cmd
        if action_buffer:
            action_buffer.clear()
            print("Discarded current key-action buffer")
        if dataset.has_pending_frames():
            dataset.clear_episode_buffer()
        target_q, target_pos, target_quat = go_home()
        ee_cmd = orient_cmd = joint_cmd = False
        held[:] = 0
        cube.set_pos((float(cube_xy[0]), float(cube_xy[1]), table_h + cube_half))
        disk.set_pos((float(disk_xy[0]), float(disk_xy[1]), table_h + 0.5 * disk_h))

    def apply_save():
        if not action_buffer:
            print("Empty episode — nothing to save")
            return
        n = len(action_buffer)
        for action in action_buffer:
            dataset.add_frame({"task": args.task, "action": action})
        action_buffer.clear()
        dataset.save_episode()
        print(f"Saved episode ({n} key-actions) → total episodes={dataset.meta.total_episodes}")

    def nudge_joint(joint_idx: int, delta: float):
        nonlocal target_q, joint_cmd
        target_q = target_q.copy()
        target_q[joint_idx] = float(
            np.clip(target_q[joint_idx] + delta, q_lo[joint_idx], q_hi[joint_idx])
        )
        joint_cmd = True

    def move_ee(delta_ee):
        """Translate by ``delta_ee`` in the current EE frame."""
        nonlocal target_pos, ee_cmd
        R = np.asarray(gu.quat_to_R(target_quat), dtype=np.float64).reshape(3, 3)
        d_world = R @ np.asarray(delta_ee, dtype=np.float64).reshape(3)
        target_pos = (np.asarray(target_pos, dtype=np.float64) + d_world).astype(np.float32)
        ee_cmd = True

    def rotate_rpy(rpy_delta):
        nonlocal target_quat, orient_cmd
        dq_r = gu.xyz_to_quat(np.asarray(rpy_delta, dtype=np.float32), rpy=True)
        target_quat = np.asarray(gu.transform_quat_by_quat(dq_r, target_quat), dtype=np.float32).reshape(4)
        orient_cmd = True

    def mark(action_idx: int, fn, *fn_args):
        """HOLD callback: mark action bit, then apply teleop."""

        def _cb(*_args):
            held[action_idx] = 1.0
            fn(*fn_args)

        return _cb

    def solve_ik(use_orient: bool):
        kwargs = dict(
            link=ee_link,
            pos=target_pos,
            dofs_idx_local=motors_dof,
            max_samples=1,
            max_solver_iters=50,
            damping=0.05,
            pos_tol=1e-3,
            rot_tol=1e-2,
        )
        if use_orient:
            kwargs["quat"] = target_quat
        q = robot.inverse_kinematics(**kwargs)
        return _to_numpy(q).reshape(-1)

    def queue_reset():
        nonlocal request_reset
        request_reset = True

    def queue_save():
        nonlocal request_save
        request_save = True

    def stop():
        nonlocal is_running
        is_running = False

    apply_reset()
    # Indices must match ACTION_NAMES order.
    keybinds = [
        Keybind("move_up", Key.UP, KeyAction.HOLD, callback=mark(0, move_ee, (0, 0, dpos))),
        Keybind("move_down", Key.DOWN, KeyAction.HOLD, callback=mark(1, move_ee, (0, 0, -dpos))),
        Keybind("move_left", Key.LEFT, KeyAction.HOLD, callback=mark(2, move_ee, (0, -dpos, 0))),
        Keybind("move_right", Key.RIGHT, KeyAction.HOLD, callback=mark(3, move_ee, (0, dpos, 0))),
        Keybind("move_forward", Key.N, KeyAction.HOLD, callback=mark(4, move_ee, (-dpos, 0, 0))),
        Keybind("move_back", Key.M, KeyAction.HOLD, callback=mark(5, move_ee, (dpos, 0, 0))),
        Keybind("roll_p", Key.Q, KeyAction.HOLD, callback=mark(6, rotate_rpy, (drot, 0, 0))),
        Keybind("roll_n", Key.E, KeyAction.HOLD, callback=mark(7, rotate_rpy, (-drot, 0, 0))),
        Keybind("pitch_p", Key.T, KeyAction.HOLD, callback=mark(8, rotate_rpy, (0, drot, 0))),
        Keybind("pitch_n", Key.G, KeyAction.HOLD, callback=mark(9, rotate_rpy, (0, -drot, 0))),
        Keybind("yaw_p", Key.Y, KeyAction.HOLD, callback=mark(10, rotate_rpy, (0, 0, drot))),
        Keybind("yaw_n", Key.B, KeyAction.HOLD, callback=mark(11, rotate_rpy, (0, 0, -drot))),
        Keybind("grip_open", Key._6, KeyAction.HOLD, callback=mark(12, nudge_joint, 5, dq)),
        Keybind("grip_close", Key.ASCIICIRCUM, KeyAction.HOLD, callback=mark(13, nudge_joint, 5, -dq)),
        Keybind("reset", Key.R, KeyAction.PRESS, callback=queue_reset),
        Keybind("save_ep", Key.ENTER, KeyAction.PRESS, callback=queue_save),
        Keybind("quit", Key.ESCAPE, KeyAction.RELEASE, callback=stop),
    ]
    scene.viewer.register_keybinds(*keybinds)

    print(__doc__)
    print(f"Home q={np.round(target_q, 3).tolist()} ee={target_pos.round(3).tolist()}")
    print("Recording: one action vector every sim step (idle = zeros)")
    print(f"Disk diameter={2 * disk_radius:.3f}m height={disk_h:.3f}m. Task: {args.task!r}")

    try:
        while is_running:
            use_orient = orient_cmd
            q_arm = solve_ik(use_orient)
            target_q = target_q.copy()
            target_q[:5] = np.clip(q_arm[:5], q_lo[:5], q_hi[:5]).astype(np.float32)

            robot.control_dofs_position(target_q, all_dofs)
            ee_cmd = False
            orient_cmd = False
            joint_cmd = False
            held[:] = 0
            scene.step()

            # One sample per sim step (zeros = idle) so replay timing matches teleop.
            action_buffer.append(held.copy())

            if request_save:
                request_save = False
                apply_save()
            if request_reset:
                request_reset = False
                apply_reset()
                continue

            link_pose = _ee_pose(ee_link)
            _sync_rgb_triad(rgb_axes, link_pose[:3], link_pose[3:7])
            if joint_cmd:
                target_pos = link_pose[:3].copy()
                target_quat = link_pose[3:7].copy()
                q_now = _to_numpy(robot.get_dofs_position(all_dofs)).reshape(-1)
                target_q[:5] = q_now[:5].astype(np.float32)
            else:
                pos_err = float(np.linalg.norm(link_pose[:3] - target_pos))
                if pos_err > 0.04:
                    target_pos = link_pose[:3].copy()
                if not orient_cmd and not use_orient:
                    target_quat = link_pose[3:7].copy()
    except gs.GenesisException as e:
        if "Viewer closed" not in str(e):
            raise
        print("Viewer closed")
    finally:
        try:
            if action_buffer:
                n = len(action_buffer)
                for action in action_buffer:
                    dataset.add_frame({"task": args.task, "action": action})
                action_buffer.clear()
                dataset.save_episode()
                print(f"Saved final episode ({n} key-actions)")
            dataset.finalize()
            print(f"Finalized LeRobot dataset at {root}")
        except Exception as e:
            print(f"Dataset cleanup warning: {e}")


if __name__ == "__main__":
    main()
