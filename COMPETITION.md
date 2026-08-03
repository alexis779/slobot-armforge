# ArmForge: Teaching SO-ARM-101 Kitchen Policies on AMD Radeon with Genesis

## Abstract

We present **ArmForge**, an open ROCm-native pipeline for SO-ARM-101 policy learning in household kitchen scenes. Using Genesis on AMD Radeon GPUs, humans teleoperate the SO-101 with Cartesian end-effector control; episodes train a vision policy via parallel RL distillation and behavior cloning, with a visual success classifier for automatic evaluation. Datasets follow LeRobot conventions so the same stack targets real SO-101 hardware. We demonstrate simulation, training, and inference on ROCm, contribute SO-101 manipulation tooling as an upstream Genesis example, and release AMD throughput benchmarks for accessible embodied AI.

## Judging criteria mapping

| Criterion (pts) | Evidence |
|-----------------|----------|
| Robot capability (30) | `eval.py` success rates on `cube_disk`; demo videos from `--record` |
| AMD Radeon + ROCm (20) | `--backend amdgpu`, `docker/Dockerfile.amdgpu`, `benchmark_fps.py` JSON benches |
| Innovation (20) | SO-101 + Genesis + ROCm kitchen teach-to-policy loop; visual success CNN; human+teacher BC mix |
| Real-world value (20) | Affordable SO-101 + LeRobot-shaped obs/actions and dataset meta |
| Upstream OSS (10) | `examples/armforge/` contribution; LeRobot export path |

## Demo script for judges

1. Show Cartesian teleop teaching cube→disk (`teleop_record.py`).
2. Show parallel sim FPS on Radeon (`benchmark_fps.py --backend amdgpu`).
3. Show RL teacher then BC student rollouts (`eval.py --stage rl|bc --record`).
4. Show visual classifier accuracy printout.
5. Point to `meta/info.json` LeRobot compatibility and Genesis example PR.

## Reproducibility

```bash
cd examples/armforge
python benchmark_fps.py --backend amdgpu -B 128 --steps 200 --out ../../logs/armforge_bench.json
python train.py --stage rl --backend amdgpu -B 512 --max_iterations 300
python train.py --stage bc --backend amdgpu --max_iterations 200
python eval.py --stage bc --episodes 20
python visual_classifier.py --backend amdgpu --steps 200
```

Hardware: AMD Radeon GPU with ROCm 6.x + PyTorch HIP build (see repo `docker/Dockerfile.amdgpu`).

## Scope notes

- Madrona is CUDA-only; ROCm uses Genesis rasterizer cameras.
- Full ACT / Diffusion / Pi stacks and real-robot fine-tuning are future work; ArmForge ships the sim→dataset→policy path on AMD.
