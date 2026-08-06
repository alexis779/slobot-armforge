#!/usr/bin/env bash
# Bootstrap slobot-armforge on an AMD Developer Cloud / ROCm VM.
# Usage (on the remote VM after cloning the repo):
#   bash scripts/cloud_bootstrap.sh
#   bash scripts/cloud_bootstrap.sh bench   # key-action throughput on amdgpu
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "[armforge] repo: $ROOT"
echo "[armforge] building ROCm Docker image..."
docker build -f docker/Dockerfile -t slobot-armforge:rocm7.2.1 .

RUN_CMD=${1:-shell}

COMMON=(
  docker run --rm -it
  --network=host
  --device=/dev/kfd --device=/dev/dri
  --group-add video
  --security-opt seccomp=unconfined
  --cap-add=SYS_PTRACE
  --ipc=host --shm-size=8g
  -v "$ROOT":/workspace/slobot-armforge
  -w /workspace/slobot-armforge
  slobot-armforge:rocm7.2.1
)

case "$RUN_CMD" in
  shell)
    "${COMMON[@]}"
    ;;
  bench)
    "${COMMON[@]}" bash -lc 'python armforge/benchmark_key_action.py --backend amdgpu -B 8 --out logs/bench.json'
    ;;
  *)
    echo "Unknown command: $RUN_CMD (shell|bench)"
    exit 1
    ;;
esac
