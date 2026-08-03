# slobot-armforge

Teach an **SO-ARM-101** to place a cube on a disk in [Genesis](https://github.com/Genesis-Embodied-AI/Genesis),
train a vision policy on **AMD Radeon / ROCm**, and export LeRobot-shaped datasets.

Layout and ROCm Docker follow the spirit of
[`wangxunx/franka_fruit_pick_demo`](https://github.com/wangxunx/franka_fruit_pick_demo)
(cloned next to this repo for reference).

## Pipeline

| Stage | Command |
|-------|---------|
| Benchmark FPS | `python armforge/benchmark_fps.py --backend amdgpu -B 64` |
| Teleop + record | `python armforge/teleop_record.py --out datasets/cube_disk` |
| RL teacher | `python armforge/train.py --stage rl --backend amdgpu -B 256` |
| Vision BC | `python armforge/train.py --stage bc --backend amdgpu --human_demo_dir datasets/cube_disk/data` |
| Eval | `python armforge/eval.py --stage rl --backend amdgpu --record` |
| Success CNN | `python armforge/visual_classifier.py --backend amdgpu` |

**Cameras:** one high-res **episode** camera (1280×960). Training downscales to 256×256.

**Actions:** Cartesian EE delta + gripper (7-D) via DLS IK.

## Quickstart (local / cloud VM)

```bash
# Python 3.12 recommended (matches ROCm wheels)
uv sync
# Install ROCm torch — see docs/AMD_DEVELOPER_CLOUD.md or use Docker below

python armforge/benchmark_fps.py --backend auto -B 16 --steps 50
```

### AMD ROCm Docker (recommended)

```bash
docker build -f docker/Dockerfile -t slobot-armforge:rocm7.2.1 .
bash scripts/cloud_bootstrap.sh          # interactive shell on GPU
bash scripts/cloud_bootstrap.sh train    # PPO on amdgpu
```

## AMD Developer Cloud

See **[docs/AMD_DEVELOPER_CLOUD.md](docs/AMD_DEVELOPER_CLOUD.md)** for claiming credits,
launching an Instinct/Radeon instance, and running the bootstrap script over SSH.

## Competition

AMD AI DevMaster Track 3 notes: [COMPETITION.md](COMPETITION.md).

## License

Apache-2.0 for this code. SO-101 MJCF is fetched from HuggingFace `Genesis-Intelligence/assets`
(Robot Studio / LeRobot lineage). Genesis is upstream Apache-2.0.
