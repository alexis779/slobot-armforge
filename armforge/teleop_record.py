"""Cartesian keyboard teleop for SO-101 with LeRobot-compatible episode recording.

Controls
--------
Arrow keys / N M / J K : translate / rotate EE
Space                  : close gripper (release to open)
R                      : reset scene
Enter                  : save episode and start a new one
Esc                    : quit (saves current episode if non-empty)
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse

import numpy as np
import torch

import genesis as gs
import genesis.utils.geom as gu
from genesis.vis.keybindings import Key, KeyAction, Keybind

from assets import so101_mjcf_path
from backend import add_backend_arg, init_genesis
from lerobot_export import EpisodeRecorder, write_dataset_meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default="datasets/armforge_so101")
    parser.add_argument("--task", type=str, default="cube_disk", choices=["cube_disk"])
    parser.add_argument("--image_size", type=int, default=256)
    add_backend_arg(parser)
    args = parser.parse_args()

    init_genesis(backend=args.backend, performance_mode=False, logging_level="info")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_dataset_meta(out_dir, robot="so101", task=args.task)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(substeps=4),
        rigid_options=gs.options.RigidOptions(
            enable_joint_limit=True,
            enable_collision=True,
            gravity=(0, 0, -9.8),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.7, -0.5, 0.4),
            camera_lookat=(0.15, 0.0, 0.08),
            camera_fov=50,
        ),
        profiling_options=gs.options.ProfilingOptions(show_FPS=False),
        show_viewer=True,
    )
    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(
        material=gs.materials.Rigid(gravity_compensation=1),
        morph=gs.morphs.MJCF(file=str(so101_mjcf_path())),
    )
    cube = scene.add_entity(
        material=gs.materials.Rigid(rho=300),
        morph=gs.morphs.Box(pos=(0.18, 0.0, 0.02), size=(0.03, 0.03, 0.03)),
        surface=gs.surfaces.Default(color=(0.9, 0.2, 0.1)),
    )
    disk = scene.add_entity(
        gs.morphs.Cylinder(pos=(0.22, 0.08, 0.005), radius=0.04, height=0.01, fixed=True),
        surface=gs.surfaces.Default(color=(0.2, 0.5, 0.9)),
    )
    target_vis = scene.add_entity(
        gs.morphs.Mesh(file="meshes/axis.obj", scale=0.08, collision=False),
        surface=gs.surfaces.Default(color=(1, 0.5, 0.5, 1)),
    )

    from genesis.options.sensors import RasterizerCameraOptions

    episode_cam = scene.add_sensor(
        RasterizerCameraOptions(
            res=(1280, 960),
            pos=(0.9, 0.0, 0.45),
            lookat=(0.15, 0.0, 0.08),
            fov=50,
        )
    )

    scene.build()

    motors_dof = np.arange(5)
    gripper_dof = np.array([5])
    ee_link = robot.get_link("gripper")

    # Home pose (Cartesian)
    target_pos = np.array([0.18, 0.0, 0.18], dtype=np.float32)
    target_quat = gu.xyz_to_quat(np.array([0.0, np.pi, 0.0]))
    dpos = 0.003
    drot = 0.02
    gripper_open = True
    recorder = EpisodeRecorder(out_dir)
    is_running = True

    def reset_scene():
        nonlocal target_pos, target_quat, gripper_open
        target_pos = np.array([0.18, 0.0, 0.18], dtype=np.float32)
        target_quat = gu.xyz_to_quat(np.array([0.0, np.pi, 0.0]))
        gripper_open = True
        q = robot.inverse_kinematics(link=ee_link, pos=target_pos, quat=target_quat, dofs_idx_local=motors_dof)
        robot.set_qpos(q[:5] if hasattr(q, "__len__") else q, motors_dof)
        robot.set_dofs_position(np.array([1.7]), gripper_dof)
        cube.set_pos((np.random.uniform(0.14, 0.22), np.random.uniform(-0.06, 0.06), 0.02))
        disk.set_pos((np.random.uniform(0.18, 0.26), np.random.uniform(0.04, 0.10), 0.005))
        episode_cam._stale = True

    def move(delta):
        nonlocal target_pos
        target_pos = target_pos + np.asarray(delta, dtype=np.float32)

    def rotate(dz):
        nonlocal target_quat
        drot_quat = gu.xyz_to_quat(np.array([0.0, 0.0, dz]))
        target_quat = gu.transform_quat_by_quat(target_quat, drot_quat)

    def set_gripper(close: bool):
        nonlocal gripper_open
        gripper_open = not close

    def save_episode():
        path = recorder.save_episode()
        if path is not None:
            print(f"Saved episode: {path}")
        recorder.start_episode()

    def stop():
        nonlocal is_running
        is_running = False

    reset_scene()
    recorder.start_episode()

    scene.viewer.register_keybinds(
        Keybind("move_forward", Key.UP, KeyAction.HOLD, callback=move, args=(( -dpos, 0, 0),)),
        Keybind("move_back", Key.DOWN, KeyAction.HOLD, callback=move, args=((dpos, 0, 0),)),
        Keybind("move_left", Key.LEFT, KeyAction.HOLD, callback=move, args=((0, -dpos, 0),)),
        Keybind("move_right", Key.RIGHT, KeyAction.HOLD, callback=move, args=((0, dpos, 0),)),
        Keybind("move_up", Key.N, KeyAction.HOLD, callback=move, args=((0, 0, dpos),)),
        Keybind("move_down", Key.M, KeyAction.HOLD, callback=move, args=((0, 0, -dpos),)),
        Keybind("rotate_ccw", Key.J, KeyAction.HOLD, callback=rotate, args=(-drot,)),
        Keybind("rotate_cw", Key.K, KeyAction.HOLD, callback=rotate, args=(drot,)),
        Keybind("grip_close", Key.SPACE, KeyAction.PRESS, callback=set_gripper, args=(True,)),
        Keybind("grip_open", Key.SPACE, KeyAction.RELEASE, callback=set_gripper, args=(False,)),
        Keybind("reset", Key.R, KeyAction.PRESS, callback=reset_scene),
        Keybind("save_ep", Key.ENTER, KeyAction.PRESS, callback=save_episode),
        Keybind("quit", Key.ESCAPE, KeyAction.RELEASE, callback=stop),
    )

    print(__doc__)
    prev_ee = None
    while is_running:
        target_vis.set_qpos(np.concatenate([target_pos, target_quat]))
        q = robot.inverse_kinematics(link=ee_link, pos=target_pos, quat=target_quat, dofs_idx_local=motors_dof)
        if isinstance(q, torch.Tensor):
            q_np = q.detach().cpu().numpy()
        else:
            q_np = np.asarray(q)
        if q_np.ndim > 1:
            q_np = q_np[0]
        robot.control_dofs_position(q_np[:5], motors_dof)
        grip = 0.0 if not gripper_open else 1.7
        robot.control_dofs_position(np.array([grip]), gripper_dof)
        scene.step()

        ee_pos = ee_link.get_pos()
        ee_quat = ee_link.get_quat()
        if isinstance(ee_pos, torch.Tensor):
            ee_pos = ee_pos.detach().cpu().numpy().reshape(-1)[:3]
            ee_quat = ee_quat.detach().cpu().numpy().reshape(-1)[:4]
        ee_pose = np.concatenate([ee_pos, ee_quat]).astype(np.float32)

        # Approximate Cartesian action as EE delta; gripper as [-1,1].
        if prev_ee is None:
            action = np.zeros(7, dtype=np.float32)
        else:
            action = np.zeros(7, dtype=np.float32)
            action[:3] = ee_pose[:3] - prev_ee[:3]
            action[6] = -1.0 if not gripper_open else 1.0
        prev_ee = ee_pose.copy()

        episode_rgb = episode_cam.read(envs_idx=0).rgb
        if isinstance(episode_rgb, torch.Tensor):
            episode_rgb = episode_rgb.detach().cpu().numpy()
        episode_rgb = np.asarray(episode_rgb)
        if episode_rgb.ndim == 4:
            episode_rgb = episode_rgb[0]

        obj_pos = cube.get_pos()
        obj_quat = cube.get_quat()
        if isinstance(obj_pos, torch.Tensor):
            obj_pos = obj_pos.detach().cpu().numpy().reshape(-1)[:3]
            obj_quat = obj_quat.detach().cpu().numpy().reshape(-1)[:4]
        object_pose = np.concatenate([obj_pos, obj_quat]).astype(np.float32)

        # Downscale high-res episode RGB to training resolution for the dataset.
        rgb_t = torch.from_numpy(np.transpose(episode_rgb[..., :3], (2, 0, 1)).astype(np.float32) / 255.0).unsqueeze(0)
        rgb_t = torch.nn.functional.interpolate(
            rgb_t, size=(args.image_size, args.image_size), mode="bilinear", align_corners=False
        )
        rgb = rgb_t.squeeze(0).numpy()

        recorder.add_step(
            rgb=rgb,
            ee_pose=ee_pose,
            object_pose=object_pose,
            action=action,
            qpos=q_np[:6] if q_np.size >= 6 else np.concatenate([q_np[:5], [grip]]),
        )

    path = recorder.save_episode()
    if path is not None:
        print(f"Saved final episode: {path}")
    meta_path = out_dir / "meta" / "info.json"
    print(f"Dataset root: {out_dir} (meta at {meta_path})")


if __name__ == "__main__":
    main()
