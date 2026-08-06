"""Multi-env key-action replay throughput benchmark.

Replays the same teleop episode across ``-B`` parallel Genesis envs with slight
per-env spawn jitter (cube x/y/yaw, cylinder x/y) and reports wall-clock
throughput. Useful for comparing local CPU vs HF GPU Jobs.

Example
-------
  # Local CPU
  python armforge/benchmark_key_action.py --backend cpu -B 8 \\
      --actions-npz datasets/so101_cube_disk/optimized/episode_000_opt.npz

  # Sweep B=1..12 (subprocess per B; Genesis init is once-per-process)
  python armforge/benchmark_key_action.py --backend cpu --sweep 1-12 --profile \\
      --actions-npz datasets/so101_cube_disk/optimized/episode_000_opt.npz \\
      --out logs/key_action_sweep_cpu.json

  # HF GPU (via scripts/hf_jobs_bench.sh)
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
import os
import platform
import subprocess
import time

import numpy as np

from backend import add_backend_arg, init_genesis
from key_action_replay import BatchedKeyActionScene, load_actions_npz, load_episode_actions, task_geometry


def sample_pose_jitter(
    num_envs: int,
    *,
    cube_xy0: tuple[float, float],
    disk_xy0: tuple[float, float],
    cube_xy_noise: float,
    cube_yaw_noise: float,
    disk_xy_noise: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-env ``cube_xy (B,2)``, ``cube_yaw (B,)``, ``disk_xy (B,2)``."""
    rng = np.random.default_rng(seed)
    cube_xy = np.broadcast_to(np.asarray(cube_xy0, dtype=np.float64), (num_envs, 2)).copy()
    disk_xy = np.broadcast_to(np.asarray(disk_xy0, dtype=np.float64), (num_envs, 2)).copy()
    cube_yaw = np.zeros(num_envs, dtype=np.float64)
    if cube_xy_noise > 0:
        cube_xy += rng.uniform(-cube_xy_noise, cube_xy_noise, size=(num_envs, 2))
    if disk_xy_noise > 0:
        disk_xy += rng.uniform(-disk_xy_noise, disk_xy_noise, size=(num_envs, 2))
    if cube_yaw_noise > 0:
        cube_yaw += rng.uniform(-cube_yaw_noise, cube_yaw_noise, size=(num_envs,))
    # Env 0 keeps the nominal pose so a baseline success check is available.
    cube_xy[0] = cube_xy0
    disk_xy[0] = disk_xy0
    cube_yaw[0] = 0.0
    return cube_xy, cube_yaw, disk_xy


def parse_sweep(spec: str) -> list[int]:
    """Parse ``'1-12'`` or ``'1,2,4,8'`` into a list of batch sizes."""
    spec = spec.strip()
    if "-" in spec and "," not in spec:
        lo_s, hi_s = spec.split("-", 1)
        lo, hi = int(lo_s), int(hi_s)
        if lo < 1 or hi < lo:
            raise ValueError(f"invalid sweep range: {spec}")
        return list(range(lo, hi + 1))
    vals = [int(x) for x in spec.split(",") if x.strip()]
    if not vals or any(v < 1 for v in vals):
        raise ValueError(f"invalid sweep list: {spec}")
    return vals


def run_benchmark(
    actions: np.ndarray,
    *,
    num_envs: int,
    fps: int,
    settle_frames: int,
    cube_xy_noise: float,
    cube_yaw_noise: float,
    disk_xy_noise: float,
    seed: int,
    warmup: bool,
    profile: bool = False,
    jitter_num_envs: int | None = None,
    jitter_env_id: int | None = None,
) -> dict:
    import genesis as gs

    geom = task_geometry()
    scene = BatchedKeyActionScene(num_envs=num_envs, fps=fps, show_viewer=False)
    sample_B = int(jitter_num_envs) if jitter_num_envs is not None else num_envs
    if sample_B < 1:
        raise ValueError("jitter_num_envs must be >= 1")
    cube_xy_full, cube_yaw_full, disk_xy_full = sample_pose_jitter(
        sample_B,
        cube_xy0=geom["cube_xy"],
        disk_xy0=geom["disk_xy"],
        cube_xy_noise=cube_xy_noise,
        cube_yaw_noise=cube_yaw_noise,
        disk_xy_noise=disk_xy_noise,
        seed=seed,
    )
    if jitter_env_id is not None:
        eid = int(jitter_env_id)
        if eid < 0 or eid >= sample_B:
            raise ValueError(f"jitter_env_id {eid} out of range for jitter_num_envs={sample_B}")
        cube_xy = np.broadcast_to(cube_xy_full[eid], (num_envs, 2)).copy()
        cube_yaw = np.full((num_envs,), float(cube_yaw_full[eid]), dtype=np.float64)
        disk_xy = np.broadcast_to(disk_xy_full[eid], (num_envs, 2)).copy()
    elif sample_B != num_envs:
        raise ValueError("jitter_num_envs != num_envs requires --jitter-env-id")
    else:
        cube_xy, cube_yaw, disk_xy = cube_xy_full, cube_yaw_full, disk_xy_full

    T = int(actions.shape[0])
    settle = max(0, int(settle_frames))
    horizon = T + settle
    zero = np.zeros((num_envs, actions.shape[1]), dtype=np.float32)

    def replay_once() -> np.ndarray:
        scene.reset_all(cube_xy=cube_xy, cube_yaw=cube_yaw, disk_xy=disk_xy)
        for t in range(horizon):
            batch = zero.copy()
            if t < T:
                batch[:] = actions[t]
            scene.step(batch)
        return scene.success_mask().copy()

    if warmup:
        # One short warmup pass (first 30 ticks) to compile kernels / fill caches.
        scene.reset_all(cube_xy=cube_xy, cube_yaw=cube_yaw, disk_xy=disk_xy)
        warm_n = min(30, T)
        for t in range(warm_n):
            batch = zero.copy()
            batch[:] = actions[t]
            scene.step(batch)

    if profile:
        scene.enable_profile()

    t0 = time.perf_counter()
    success = replay_once()
    elapsed = time.perf_counter() - t0
    physics_failed = scene.physics_failed_mask()

    env_steps = num_envs * horizon
    result = {
        "backend": str(gs.backend),
        "device": str(gs.device),
        "platform": platform.platform(),
        "cpu_count_logical": os.cpu_count(),
        "num_envs": num_envs,
        "episode_frames": T,
        "settle_frames": settle,
        "horizon": horizon,
        "fps": fps,
        "cube_xy_noise_m": cube_xy_noise,
        "cube_yaw_noise_rad": cube_yaw_noise,
        "disk_xy_noise_m": disk_xy_noise,
        "seed": seed,
        "wall_time_s": elapsed,
        "env_steps_per_s": env_steps / elapsed if elapsed > 0 else 0.0,
        "episodes_per_s": num_envs / elapsed if elapsed > 0 else 0.0,
        "sim_fps_per_env": horizon / elapsed if elapsed > 0 else 0.0,
        "success_count": int(success.sum()),
        "success_rate": float(success.mean()),
        "success_env0": bool(success[0]),
        "physics_fail_count": int(physics_failed.sum()),
        "physics_fail_rate": float(physics_failed.mean()),
        "physics_fail_env_ids": np.nonzero(physics_failed)[0][:64].tolist(),
        "jitter_num_envs": sample_B,
        "jitter_env_id": None if jitter_env_id is None else int(jitter_env_id),
        "pose_cube_xy": cube_xy.tolist(),
        "pose_cube_yaw": cube_yaw.tolist(),
        "pose_disk_xy": disk_xy.tolist(),
    }
    if profile and scene.profile is not None:
        p = scene.profile
        steps = max(1.0, float(p["steps"]))
        total = sum(float(p[k]) for k in ("apply_s", "ik_s", "control_s", "physics_s", "sync_s"))
        result["profile"] = {
            "steps": int(steps),
            "apply_s": p["apply_s"],
            "ik_s": p["ik_s"],
            "control_s": p["control_s"],
            "physics_s": p["physics_s"],
            "sync_s": p["sync_s"],
            "profiled_s": total,
            "frac_apply": p["apply_s"] / total if total > 0 else 0.0,
            "frac_ik": p["ik_s"] / total if total > 0 else 0.0,
            "frac_control": p["control_s"] / total if total > 0 else 0.0,
            "frac_physics": p["physics_s"] / total if total > 0 else 0.0,
            "frac_sync": p["sync_s"] / total if total > 0 else 0.0,
            "ms_per_step": {
                "apply": 1000.0 * p["apply_s"] / steps,
                "ik": 1000.0 * p["ik_s"] / steps,
                "control": 1000.0 * p["control_s"] / steps,
                "physics": 1000.0 * p["physics_s"] / steps,
                "sync": 1000.0 * p["sync_s"] / steps,
            },
        }
    return result


def run_sweep_subprocess(args: argparse.Namespace, batch_sizes: list[int], actions_src: str) -> dict:
    """Run one fresh Python process per B (Genesis ``gs.init`` is once-per-process)."""
    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    script = Path(__file__).resolve()
    for b in batch_sizes:
        part = out_dir / f"_sweep_part_B{b}.json"
        cmd = [
            sys.executable,
            str(script),
            "--backend",
            args.backend,
            "-B",
            str(b),
            "--fps",
            str(args.fps),
            "--settle-frames",
            str(args.settle_frames),
            "--cube-xy-noise",
            str(args.cube_xy_noise),
            "--cube-yaw-noise",
            str(args.cube_yaw_noise),
            "--disk-xy-noise",
            str(args.disk_xy_noise),
            "--seed",
            str(args.seed),
            "--out",
            str(part),
        ]
        if args.actions_npz:
            cmd.extend(["--actions-npz", args.actions_npz])
        else:
            cmd.extend(["--repo-id", args.repo_id, "--root", args.root, "--episode", str(args.episode)])
        if args.profile:
            cmd.append("--profile")
        if args.no_warmup:
            cmd.append("--no-warmup")
        print(f"[sweep] B={b} …", flush=True)
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0 or not part.is_file():
            print(f"[sweep] B={b} FAILED (exit={proc.returncode})", flush=True)
            rows.append(
                {
                    "num_envs": b,
                    "error": f"subprocess_exit_{proc.returncode}",
                    "env_steps_per_s": None,
                    "wall_time_s": None,
                }
            )
            part.unlink(missing_ok=True)
            continue
        rows.append(json.loads(part.read_text(encoding="ascii")))
        part.unlink(missing_ok=True)

    summary = {
        "sweep": batch_sizes,
        "backend_arg": args.backend,
        "source": actions_src,
        "profile": bool(args.profile),
        "rows": [{k: v for k, v in r.items() if not str(k).startswith("pose_")} for r in rows],
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--actions-npz", type=str, default=None, help="NPZ with key 'action' (T, 14)")
    parser.add_argument("--repo-id", type=str, default="local/so101_cube_disk")
    parser.add_argument("--root", type=str, default="datasets/so101_cube_disk")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("-B", "--num_envs", type=int, default=8)
    parser.add_argument(
        "--sweep",
        type=str,
        default=None,
        help="Batch-size sweep, e.g. '1-12' or '1,2,4,8' (subprocess per B)",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--settle-frames", type=int, default=0)
    parser.add_argument(
        "--cube-xy-noise",
        type=float,
        default=0.01,
        help="Uniform ±noise (m) on cube x/y (env 0 fixed)",
    )
    parser.add_argument(
        "--cube-yaw-noise",
        type=float,
        default=0.1,
        help="Uniform ±noise (rad) on cube yaw (env 0 fixed)",
    )
    parser.add_argument(
        "--disk-xy-noise",
        type=float,
        default=0.01,
        help="Uniform ±noise (m) on cylinder x/y (env 0 fixed)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--jitter-num-envs",
        type=int,
        default=None,
        help="Sample pose jitter as if this many envs (for isolating a multi-env pose)",
    )
    parser.add_argument(
        "--jitter-env-id",
        type=int,
        default=None,
        help="Use only this env's pose from the jitter_num_envs sample (broadcast to -B)",
    )
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Break down step time into apply / IK / control / physics / sync",
    )
    parser.add_argument("--out", type=str, default="logs/key_action_bench.json")
    add_backend_arg(parser)
    args = parser.parse_args()

    if args.actions_npz:
        src = args.actions_npz
    else:
        root = Path(args.root) if args.root else None
        src = str(root if root is not None else args.repo_id)

    if args.sweep:
        batch_sizes = parse_sweep(args.sweep)
        summary = run_sweep_subprocess(args, batch_sizes, src)
        print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
        for r in summary["rows"]:
            if r.get("error"):
                print(json.dumps({"B": r["num_envs"], "error": r["error"]}))
                continue
            line = {
                "B": r["num_envs"],
                "device": r["device"],
                "env_steps_per_s": round(r["env_steps_per_s"], 1),
                "wall_time_s": round(r["wall_time_s"], 3),
            }
            if "profile" in r:
                p = r["profile"]
                line["frac"] = {
                    "apply": round(p["frac_apply"], 3),
                    "ik": round(p["frac_ik"], 3),
                    "control": round(p["frac_control"], 3),
                    "physics": round(p["frac_physics"], 3),
                    "sync": round(p["frac_sync"], 3),
                }
            print(json.dumps(line))
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="ascii") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote {out}")
        return

    if args.actions_npz:
        actions = load_actions_npz(args.actions_npz)
    else:
        root = Path(args.root) if args.root else None
        actions = load_episode_actions(args.repo_id, args.episode, root=root)
    print(f"Loaded {len(actions)} frames from {src}")

    init_genesis(backend=args.backend, performance_mode=True, logging_level="warning")
    result = run_benchmark(
        actions,
        num_envs=args.num_envs,
        fps=args.fps,
        settle_frames=args.settle_frames,
        cube_xy_noise=args.cube_xy_noise,
        cube_yaw_noise=args.cube_yaw_noise,
        disk_xy_noise=args.disk_xy_noise,
        seed=args.seed,
        warmup=not args.no_warmup,
        profile=args.profile,
        jitter_num_envs=args.jitter_num_envs,
        jitter_env_id=args.jitter_env_id,
    )
    result["source"] = str(src)
    result["episode"] = int(args.episode)

    print(json.dumps({k: v for k, v in result.items() if not k.startswith("pose_")}, indent=2))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="ascii") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
