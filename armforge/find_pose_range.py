"""Find min/max initial cube/cylinder poses that still succeed under multi-env replay.

Uses ``BatchedKeyActionScene`` with ``-B`` parallel envs to:

1. Sweep each parameter independently (others fixed at nominal) on a linspace
   over a search window, and take the largest contiguous successful interval
   containing the nominal value.
2. Verify all ``2^5 = 32`` corners of the resulting 5D box in one batch.
3. If any corner fails, shrink all ranges toward nominal by a common scale
   (binary search) until every corner succeeds.

Parameters (absolute meters / radians; offsets reported relative to nominal)::

  x_cube, y_cube, yaw_cube, x_cylinder, y_cylinder

Example
-------
  python armforge/find_pose_range.py --backend cpu -B 64 \\
      --actions-npz datasets/so101_cube_disk/optimized/episode_000_pose_range.npz
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
import time
from dataclasses import dataclass

import numpy as np

from backend import add_backend_arg, init_genesis
from key_action_replay import BatchedKeyActionScene, load_actions_npz, load_episode_actions, task_geometry

# (name, unit, which array, axis index or None for yaw)
PARAM_SPECS: tuple[tuple[str, str, str, int | None], ...] = (
    ("x_cube", "m", "cube_xy", 0),
    ("y_cube", "m", "cube_xy", 1),
    ("yaw_cube", "rad", "cube_yaw", None),
    ("x_cylinder", "m", "disk_xy", 0),
    ("y_cylinder", "m", "disk_xy", 1),
)


@dataclass
class ParamRange:
    name: str
    unit: str
    nominal: float
    min: float
    max: float

    @property
    def min_offset(self) -> float:
        return self.min - self.nominal

    @property
    def max_offset(self) -> float:
        return self.max - self.nominal


def _nominal_poses(geom: dict) -> tuple[np.ndarray, float, np.ndarray]:
    cube_xy0 = np.asarray(geom["cube_xy"], dtype=np.float64)
    disk_xy0 = np.asarray(geom["disk_xy"], dtype=np.float64)
    return cube_xy0, 0.0, disk_xy0


def _nominal_value(name: str, cube_xy0: np.ndarray, yaw0: float, disk_xy0: np.ndarray) -> float:
    for n, _unit, kind, axis in PARAM_SPECS:
        if n != name:
            continue
        if kind == "cube_xy":
            return float(cube_xy0[int(axis)])
        if kind == "disk_xy":
            return float(disk_xy0[int(axis)])
        return float(yaw0)
    raise KeyError(name)


def build_pose_batch(
    num_envs: int,
    *,
    cube_xy0: np.ndarray,
    yaw0: float,
    disk_xy0: np.ndarray,
    overrides: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Broadcast nominal poses; apply per-env 1D overrides keyed by param name."""
    cube_xy = np.broadcast_to(cube_xy0, (num_envs, 2)).copy()
    cube_yaw = np.full(num_envs, float(yaw0), dtype=np.float64)
    disk_xy = np.broadcast_to(disk_xy0, (num_envs, 2)).copy()
    if not overrides:
        return cube_xy, cube_yaw, disk_xy
    for name, values in overrides.items():
        vals = np.asarray(values, dtype=np.float64).reshape(-1)
        if vals.shape[0] != num_envs:
            raise ValueError(f"{name}: expected {num_envs} values, got {vals.shape[0]}")
        for n, _unit, kind, axis in PARAM_SPECS:
            if n != name:
                continue
            if kind == "cube_xy":
                cube_xy[:, int(axis)] = vals
            elif kind == "disk_xy":
                disk_xy[:, int(axis)] = vals
            else:
                cube_yaw[:] = vals
            break
        else:
            raise KeyError(name)
    return cube_xy, cube_yaw, disk_xy


def replay_once(
    scene: BatchedKeyActionScene,
    actions: np.ndarray,
    *,
    cube_xy: np.ndarray,
    cube_yaw: np.ndarray,
    disk_xy: np.ndarray,
    settle_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reset all envs to the given poses, replay, return (success, physics_failed)."""
    B = scene.num_envs
    T = int(actions.shape[0])
    settle = max(0, int(settle_frames))
    zero = np.zeros((B, actions.shape[1]), dtype=np.float32)
    scene.reset_all(cube_xy=cube_xy, cube_yaw=cube_yaw, disk_xy=disk_xy)
    for t in range(T + settle):
        batch = zero.copy()
        if t < T:
            batch[:] = actions[t]
        scene.step(batch)
    return scene.success_mask().copy(), scene.physics_failed_mask().copy()


def contiguous_success_around_nominal(
    values: np.ndarray,
    success: np.ndarray,
    nominal: float,
) -> tuple[float, float] | None:
    """Largest contiguous successful interval containing ``nominal`` (or None)."""
    order = np.argsort(values)
    v = values[order]
    ok = success[order]
    # Index of sample closest to nominal.
    i0 = int(np.argmin(np.abs(v - nominal)))
    if not ok[i0]:
        # Nominal sample itself failed — try exact nominal not on grid.
        return None
    lo_i = i0
    while lo_i > 0 and ok[lo_i - 1]:
        lo_i -= 1
    hi_i = i0
    while hi_i + 1 < len(ok) and ok[hi_i + 1]:
        hi_i += 1
    return float(v[lo_i]), float(v[hi_i])


def scan_axis(
    scene: BatchedKeyActionScene,
    actions: np.ndarray,
    *,
    name: str,
    nominal: float,
    search_lo: float,
    search_hi: float,
    cube_xy0: np.ndarray,
    yaw0: float,
    disk_xy0: np.ndarray,
    settle_frames: int,
) -> dict:
    """Linspace ``name`` over [search_lo, search_hi]; return contiguous success range."""
    B = scene.num_envs
    # Ensure nominal is represented exactly (replace closest grid point).
    grid = np.linspace(search_lo, search_hi, B, dtype=np.float64)
    grid[int(np.argmin(np.abs(grid - nominal)))] = nominal
    cube_xy, cube_yaw, disk_xy = build_pose_batch(
        B,
        cube_xy0=cube_xy0,
        yaw0=yaw0,
        disk_xy0=disk_xy0,
        overrides={name: grid},
    )
    t0 = time.perf_counter()
    success, phys_fail = replay_once(
        scene,
        actions,
        cube_xy=cube_xy,
        cube_yaw=cube_yaw,
        disk_xy=disk_xy,
        settle_frames=settle_frames,
    )
    elapsed = time.perf_counter() - t0
    span = contiguous_success_around_nominal(grid, success & ~phys_fail, nominal)
    if span is None:
        vmin = vmax = nominal
        ok = False
    else:
        vmin, vmax = span
        ok = True
    return {
        "name": name,
        "search_lo": float(search_lo),
        "search_hi": float(search_hi),
        "nominal": float(nominal),
        "min": float(vmin),
        "max": float(vmax),
        "ok": ok,
        "success_count": int(success.sum()),
        "physics_fail_count": int(phys_fail.sum()),
        "wall_time_s": elapsed,
        "grid": grid.tolist(),
        "success": success.astype(bool).tolist(),
        "physics_failed": phys_fail.astype(bool).tolist(),
    }


def corner_overrides(ranges: dict[str, ParamRange], num_envs: int) -> dict[str, np.ndarray]:
    """Fill envs with all 2^K corners of the range box (repeat if B > 2^K)."""
    names = [n for n, *_ in PARAM_SPECS]
    lows = np.array([ranges[n].min for n in names], dtype=np.float64)
    highs = np.array([ranges[n].max for n in names], dtype=np.float64)
    K = len(names)
    n_corners = 1 << K
    corners = np.zeros((n_corners, K), dtype=np.float64)
    for i in range(n_corners):
        for k in range(K):
            corners[i, k] = highs[k] if (i >> k) & 1 else lows[k]
    # Tile / trim to B.
    idx = np.arange(num_envs) % n_corners
    picked = corners[idx]
    return {names[k]: picked[:, k].copy() for k in range(K)}


def verify_corners(
    scene: BatchedKeyActionScene,
    actions: np.ndarray,
    ranges: dict[str, ParamRange],
    *,
    cube_xy0: np.ndarray,
    yaw0: float,
    disk_xy0: np.ndarray,
    settle_frames: int,
) -> dict:
    B = scene.num_envs
    overrides = corner_overrides(ranges, B)
    cube_xy, cube_yaw, disk_xy = build_pose_batch(
        B,
        cube_xy0=cube_xy0,
        yaw0=yaw0,
        disk_xy0=disk_xy0,
        overrides=overrides,
    )
    t0 = time.perf_counter()
    success, phys_fail = replay_once(
        scene,
        actions,
        cube_xy=cube_xy,
        cube_yaw=cube_yaw,
        disk_xy=disk_xy,
        settle_frames=settle_frames,
    )
    elapsed = time.perf_counter() - t0
    # Only the unique corner slots matter (first 32 if B>=32).
    n_corners = 1 << len(PARAM_SPECS)
    n_check = min(B, n_corners)
    corner_ok = bool(np.all(success[:n_check] & ~phys_fail[:n_check]))
    return {
        "all_corners_ok": corner_ok,
        "n_corners_checked": n_check,
        "success_count": int(success[:n_check].sum()),
        "physics_fail_count": int(phys_fail[:n_check].sum()),
        "wall_time_s": elapsed,
        "failed_corner_ids": np.nonzero(~(success[:n_check] & ~phys_fail[:n_check]))[0].tolist(),
    }


def scale_ranges(ranges: dict[str, ParamRange], scale: float) -> dict[str, ParamRange]:
    out: dict[str, ParamRange] = {}
    for name, r in ranges.items():
        out[name] = ParamRange(
            name=r.name,
            unit=r.unit,
            nominal=r.nominal,
            min=r.nominal + scale * (r.min - r.nominal),
            max=r.nominal + scale * (r.max - r.nominal),
        )
    return out


def shrink_until_corners_ok(
    scene: BatchedKeyActionScene,
    actions: np.ndarray,
    ranges: dict[str, ParamRange],
    *,
    cube_xy0: np.ndarray,
    yaw0: float,
    disk_xy0: np.ndarray,
    settle_frames: int,
    tol: float = 1e-3,
    max_iters: int = 12,
) -> tuple[dict[str, ParamRange], list[dict]]:
    """Binary-search common scale in (0, 1] so all box corners succeed."""
    history: list[dict] = []
    base = verify_corners(
        scene,
        actions,
        ranges,
        cube_xy0=cube_xy0,
        yaw0=yaw0,
        disk_xy0=disk_xy0,
        settle_frames=settle_frames,
    )
    history.append({"scale": 1.0, **base})
    if base["all_corners_ok"]:
        return ranges, history

    lo, hi = 0.0, 1.0
    best_scale = 0.0
    best_ranges = scale_ranges(ranges, 0.0)
    for _ in range(max_iters):
        mid = 0.5 * (lo + hi)
        cand = scale_ranges(ranges, mid)
        res = verify_corners(
            scene,
            actions,
            cand,
            cube_xy0=cube_xy0,
            yaw0=yaw0,
            disk_xy0=disk_xy0,
            settle_frames=settle_frames,
        )
        history.append({"scale": mid, **res})
        if res["all_corners_ok"]:
            best_scale = mid
            best_ranges = cand
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    print(f"[corners] shrunk to scale={best_scale:.4f}", flush=True)
    return best_ranges, history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--actions-npz",
        type=str,
        default="datasets/so101_cube_disk/optimized/episode_000_pose_range.npz",
        help="NPZ with key 'action' (T, 14). Default: crafted place demo that succeeds on CPU.",
    )
    parser.add_argument("--repo-id", type=str, default="local/so101_cube_disk")
    parser.add_argument("--root", type=str, default="datasets/so101_cube_disk")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument(
        "--from-dataset",
        action="store_true",
        help="Load actions from LeRobot --root/--episode instead of --actions-npz",
    )
    parser.add_argument("-B", "--num_envs", type=int, default=64)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--settle-frames", type=int, default=0)
    parser.add_argument(
        "--xy-search",
        type=float,
        default=0.05,
        help="Half-width (m) of independent XY linspace around nominal",
    )
    parser.add_argument(
        "--yaw-search",
        type=float,
        default=0.5,
        help="Half-width (rad) of independent yaw linspace around nominal",
    )
    parser.add_argument(
        "--no-corner-shrink",
        action="store_true",
        help="Skip joint corner verification / shrink",
    )
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--out", type=str, default="logs/pose_range.json")
    add_backend_arg(parser)
    args = parser.parse_args()

    if args.num_envs < 32:
        print(f"WARNING: B={args.num_envs} < 32; corner check covers only {args.num_envs} of 32 corners")

    if args.from_dataset:
        root = Path(args.root) if args.root else None
        actions = load_episode_actions(args.repo_id, args.episode, root=root)
        src = str(root if root is not None else args.repo_id)
    else:
        actions = load_actions_npz(args.actions_npz)
        src = args.actions_npz
    print(f"Loaded {len(actions)} frames from {src}", flush=True)

    init_genesis(backend=args.backend, performance_mode=True, logging_level="warning")
    geom = task_geometry()
    cube_xy0, yaw0, disk_xy0 = _nominal_poses(geom)
    scene = BatchedKeyActionScene(num_envs=args.num_envs, fps=args.fps, show_viewer=False)

    if not args.no_warmup:
        print("[warmup] …", flush=True)
        cube_xy, cube_yaw, disk_xy = build_pose_batch(
            args.num_envs, cube_xy0=cube_xy0, yaw0=yaw0, disk_xy0=disk_xy0
        )
        scene.reset_all(cube_xy=cube_xy, cube_yaw=cube_yaw, disk_xy=disk_xy)
        warm_n = min(30, len(actions))
        zero = np.zeros((args.num_envs, actions.shape[1]), dtype=np.float32)
        for t in range(warm_n):
            batch = zero.copy()
            batch[:] = actions[t]
            scene.step(batch)

    # Fail fast if the nominal pose itself does not succeed.
    print("[baseline] nominal pose …", flush=True)
    cube_xy, cube_yaw, disk_xy = build_pose_batch(
        args.num_envs, cube_xy0=cube_xy0, yaw0=yaw0, disk_xy0=disk_xy0
    )
    base_ok, base_fail = replay_once(
        scene,
        actions,
        cube_xy=cube_xy,
        cube_yaw=cube_yaw,
        disk_xy=disk_xy,
        settle_frames=args.settle_frames,
    )
    print(
        f"  baseline success={int(base_ok.sum())}/{args.num_envs} "
        f"physics_fail={int(base_fail.sum())}",
        flush=True,
    )
    if not bool(base_ok.all()) or bool(base_fail.any()):
        raise SystemExit(
            "Nominal pose is not 100% successful — fix the demo trajectory before range search."
        )

    axis_results: list[dict] = []
    ranges: dict[str, ParamRange] = {}
    for name, unit, _kind, _axis in PARAM_SPECS:
        nom = _nominal_value(name, cube_xy0, yaw0, disk_xy0)
        half = float(args.yaw_search if name == "yaw_cube" else args.xy_search)
        print(f"[scan] {name} in [{nom - half:.4g}, {nom + half:.4g}] …", flush=True)
        res = scan_axis(
            scene,
            actions,
            name=name,
            nominal=nom,
            search_lo=nom - half,
            search_hi=nom + half,
            cube_xy0=cube_xy0,
            yaw0=yaw0,
            disk_xy0=disk_xy0,
            settle_frames=args.settle_frames,
        )
        axis_results.append(res)
        ranges[name] = ParamRange(
            name=name,
            unit=unit,
            nominal=nom,
            min=res["min"],
            max=res["max"],
        )
        print(
            f"  -> [{res['min']:.6g}, {res['max']:.6g}] "
            f"(offsets [{res['min'] - nom:+.4g}, {res['max'] - nom:+.4g}]) "
            f"success={res['success_count']}/{args.num_envs} "
            f"in {res['wall_time_s']:.1f}s",
            flush=True,
        )

    corner_history: list[dict] = []
    if not args.no_corner_shrink:
        print("[corners] verifying 5D extremes …", flush=True)
        ranges, corner_history = shrink_until_corners_ok(
            scene,
            actions,
            ranges,
            cube_xy0=cube_xy0,
            yaw0=yaw0,
            disk_xy0=disk_xy0,
            settle_frames=args.settle_frames,
        )

    summary_ranges = {}
    for name, *_ in PARAM_SPECS:
        r = ranges[name]
        summary_ranges[name] = {
            "unit": r.unit,
            "nominal": r.nominal,
            "min": r.min,
            "max": r.max,
            "min_offset": r.min_offset,
            "max_offset": r.max_offset,
        }

    result = {
        "source": str(src),
        "episode": int(args.episode),
        "num_envs": args.num_envs,
        "fps": args.fps,
        "settle_frames": args.settle_frames,
        "xy_search_m": args.xy_search,
        "yaw_search_rad": args.yaw_search,
        "nominal": {
            "x_cube": float(cube_xy0[0]),
            "y_cube": float(cube_xy0[1]),
            "yaw_cube": float(yaw0),
            "x_cylinder": float(disk_xy0[0]),
            "y_cylinder": float(disk_xy0[1]),
        },
        "ranges": summary_ranges,
        "axis_scans": [{k: v for k, v in r.items() if k not in ("grid", "success", "physics_failed")} for r in axis_results],
        "axis_scans_detail": axis_results,
        "corner_history": corner_history,
        "corners_ok": bool(corner_history[-1]["all_corners_ok"]) if corner_history else None,
        "corner_scale": float(corner_history[-1]["scale"]) if corner_history else None,
    }

    print("\n=== Robust pose ranges (extremes succeed jointly) ===", flush=True)
    for name, *_ in PARAM_SPECS:
        r = summary_ranges[name]
        print(
            f"  {name:12s}  [{r['min']:.6g}, {r['max']:.6g}] {r['unit']}  "
            f"(nominal {r['nominal']:.6g}, offsets [{r['min_offset']:+.4g}, {r['max_offset']:+.4g}])",
            flush=True,
        )
    if result["corners_ok"] is not None:
        print(
            f"  corners_ok={result['corners_ok']}  scale={result['corner_scale']}",
            flush=True,
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="ascii") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
