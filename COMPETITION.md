# ArmForge: Teaching SO-ARM-101 Policies on AMD Radeon with Genesis

## Abstract

We present **ArmForge**, an open ROCm-native pipeline for SO-ARM-101 teleop and dataset collection in cube-disk pick-and-place scenes. Using Genesis on AMD Radeon GPUs, humans teleoperate the SO-101 with key-action control; episodes export LeRobot-shaped datasets so the same stack targets real SO-101 hardware. We demonstrate simulation, recording, and replay on ROCm, contribute SO-101 manipulation tooling as an upstream Genesis example, and release AMD throughput benchmarks for accessible embodied AI.

## Judging criteria mapping

| Criterion (pts) | Evidence |
|-----------------|----------|
| Robot capability (30) | Teleop success on `cube_disk`; demo videos from `record_lerobot.py` / replay |
| AMD Radeon + ROCm (20) | `--backend amdgpu`, `docker/Dockerfile`, `benchmark_key_action.py` JSON benches |
| Innovation (20) | SO-101 + Genesis + ROCm teleop→LeRobot loop; key-action optimize + multi-env bench |
| Real-world value (20) | Affordable SO-101 + LeRobot-shaped obs/actions and dataset meta |
| Upstream OSS (10) | `examples/armforge/` contribution; LeRobot export path |

## Demo script for judges

1. Show key-action teleop teaching cube→disk (`record_lerobot.py`).
2. Show parallel key-action throughput on Radeon (`benchmark_key_action.py --backend amdgpu`).
3. Show optimized episode replay (`optimize_lerobot.py` then `replay_lerobot.py`).
4. Point to `meta/info.json` LeRobot compatibility and Genesis example PR.

## Reproducibility

```bash
cd examples/armforge
python record_lerobot.py --root datasets/so101_cube_disk
python optimize_lerobot.py --root datasets/so101_cube_disk --episode 0 --backend cpu
python replay_lerobot.py --root datasets/so101_cube_disk --episode 0
python benchmark_key_action.py --backend amdgpu -B 8
```

Hardware: AMD Radeon GPU with ROCm 6.x + PyTorch HIP build (see repo `docker/Dockerfile`).

## Scope notes

- Madrona is CUDA-only; ROCm uses Genesis rasterizer cameras.
- Full ACT / Diffusion / Pi stacks and real-robot fine-tuning are future work; ArmForge ships the sim→dataset path on AMD.
