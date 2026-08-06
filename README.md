# slobot-armforge

Teach an **SO-ARM-101** to place a cube on a disk in [Genesis](https://github.com/Genesis-Embodied-AI/Genesis),
record teleop on **AMD Radeon / ROCm**, and export LeRobot-shaped datasets.

Layout and ROCm Docker follow the spirit of
[`wangxunx/franka_fruit_pick_demo`](https://github.com/wangxunx/franka_fruit_pick_demo)
(cloned next to this repo for reference).

## Pipeline

| Stage | Command |
|-------|---------|
| Teleop + record (NPZ) | `python armforge/teleop_record.py --out datasets/cube_disk` |
| Teleop + LeRobot dataset | `python armforge/record_lerobot.py --root datasets/so101_cube_disk` |
| Replay LeRobot episode | `python armforge/replay_lerobot.py --root datasets/so101_cube_disk --episode 0` |
| Drop idle teleop frames | `python armforge/optimize_lerobot.py --root datasets/so101_cube_disk --episode 0 --backend cpu` |
| Key-action throughput | `python armforge/benchmark_key_action.py --backend amdgpu -B 8` |

**Cameras:** one high-res **episode** camera (1280×960).

**Actions:** key-action teleop / replay use a 14-D multi-hot layout (`record_lerobot.py`).
`optimize_lerobot.py` drops all-zero idle frames and verifies success in a single env.

## Quickstart (local / cloud VM)

```bash
# Python 3.12 recommended (matches ROCm wheels)
uv sync
# Install ROCm torch — see docs/AMD_DEVELOPER_CLOUD.md or use Docker below

python armforge/benchmark_key_action.py --backend auto -B 8
```

### AMD ROCm Docker (recommended)

```bash
docker build -f docker/Dockerfile -t slobot-armforge:rocm7.2.1 .
bash scripts/cloud_bootstrap.sh          # interactive shell on GPU
bash scripts/cloud_bootstrap.sh bench    # key-action throughput on amdgpu
```

## AMD Developer Cloud

See **[docs/AMD_DEVELOPER_CLOUD.md](docs/AMD_DEVELOPER_CLOUD.md)** for claiming credits,
launching an Instinct/Radeon instance, and running the bootstrap script over SSH.

## Competition

AMD AI DevMaster Track 3 notes: [COMPETITION.md](COMPETITION.md).

## License

Apache-2.0 for this code. SO-101 MJCF is fetched from HuggingFace `Genesis-Intelligence/assets`
(Robot Studio / LeRobot lineage). Genesis is upstream Apache-2.0.
