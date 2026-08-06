#!/usr/bin/env bash
# Launch multi-env key-action throughput bench on Hugging Face Jobs (NVIDIA GPU).
#
# Uploads a local code+actions snapshot to the Hub, then runs the same
# benchmark_key_action.py command used locally (with --backend cuda).
#
# Prerequisites:
#   export HF_TOKEN=...   # or hf auth login
#
# Usage:
#   bash scripts/hf_jobs_bench.sh
#   FLAVOR=a10g-large N_ENVS=32 bash scripts/hf_jobs_bench.sh
#   SWEEP=1-12 TIMEOUT=2h bash scripts/hf_jobs_bench.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Prefer `hf` on PATH; fall back to `uv run hf` from the project venv.
if [[ -n "${HF_BIN:-}" ]]; then
  HF_CMD=("$HF_BIN")
elif command -v hf >/dev/null 2>&1; then
  HF_CMD=(hf)
else
  HF_CMD=(uv run hf)
fi
FLAVOR="${FLAVOR:-a10g-small}"
N_ENVS="${N_ENVS:-8}"
SWEEP="${SWEEP:-}"
TIMEOUT="${TIMEOUT:-1h}"
IMAGE="${IMAGE:-pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime}"
HUB_REPO="${HUB_REPO:-armforge-key-action-bench}"
ACTIONS_NPZ="${ACTIONS_NPZ:-datasets/so101_cube_disk/optimized/episode_000_opt.npz}"
CUBE_XY_NOISE="${CUBE_XY_NOISE:-0.01}"
CUBE_YAW_NOISE="${CUBE_YAW_NOISE:-0.1}"
DISK_XY_NOISE="${DISK_XY_NOISE:-0.01}"
SEED="${SEED:-0}"
JITTER_NUM_ENVS="${JITTER_NUM_ENVS:-}"
JITTER_ENV_ID="${JITTER_ENV_ID:-}"
JITTER_FLAGS=""
if [[ -n "$JITTER_NUM_ENVS" ]]; then
  JITTER_FLAGS+=" --jitter-num-envs $JITTER_NUM_ENVS"
fi
if [[ -n "$JITTER_ENV_ID" ]]; then
  JITTER_FLAGS+=" --jitter-env-id $JITTER_ENV_ID"
fi
PROFILE_FLAG=""
if [[ "${PROFILE:-0}" == "1" ]]; then
  PROFILE_FLAG="--profile"
fi

cd "$ROOT"

USER_NAME="$(uv run python -c 'from huggingface_hub import whoami; print(whoami()["name"])')"
REPO_ID="${USER_NAME}/${HUB_REPO}"
SNAP_DIR="$(mktemp -d /tmp/armforge-bench-XXXXXX)"
trap 'rm -rf "$SNAP_DIR"' EXIT

mkdir -p "$SNAP_DIR/armforge" "$SNAP_DIR/datasets"
cp -a armforge/*.py "$SNAP_DIR/armforge/"
cp -a "$ACTIONS_NPZ" "$SNAP_DIR/datasets/episode_actions.npz"

echo "[hf-jobs-bench] uploading snapshot → $REPO_ID"
uv run python - <<PY
from pathlib import Path
from huggingface_hub import HfApi
api = HfApi()
repo_id = "$REPO_ID"
api.create_repo(repo_id, repo_type="dataset", exist_ok=True, private=True)
api.upload_folder(
    folder_path="$SNAP_DIR",
    repo_id=repo_id,
    repo_type="dataset",
    path_in_repo="snap",
)
print("uploaded", repo_id)
PY

if [[ -n "$SWEEP" ]]; then
  SWEEP_TAG="${SWEEP//-/_}"
  SWEEP_TAG="${SWEEP_TAG//,/_}"
  OUT_NAME="${SWEEP_OUT:-key_action_sweep_hf_${SWEEP_TAG}.json}"
  BENCH_CMD="xvfb-run -a python armforge/benchmark_key_action.py \
  --backend cuda \
  --sweep $SWEEP \
  --actions-npz datasets/episode_actions.npz \
  --cube-xy-noise $CUBE_XY_NOISE \
  --cube-yaw-noise $CUBE_YAW_NOISE \
  --disk-xy-noise $DISK_XY_NOISE \
  --seed $SEED \
  $PROFILE_FLAG \
  --out logs/$OUT_NAME"
  echo "[hf-jobs-bench] flavor=$FLAVOR sweep=$SWEEP timeout=$TIMEOUT"
else
  OUT_NAME="${OUT_NAME:-key_action_bench_hf_B${N_ENVS}.json}"
  BENCH_CMD="xvfb-run -a python armforge/benchmark_key_action.py \
  --backend cuda \
  -B $N_ENVS \
  --actions-npz datasets/episode_actions.npz \
  --cube-xy-noise $CUBE_XY_NOISE \
  --cube-yaw-noise $CUBE_YAW_NOISE \
  --disk-xy-noise $DISK_XY_NOISE \
  --seed $SEED \
  $JITTER_FLAGS \
  $PROFILE_FLAG \
  --out logs/$OUT_NAME"
  echo "[hf-jobs-bench] flavor=$FLAVOR n_envs=$N_ENVS timeout=$TIMEOUT jitter=${JITTER_FLAGS:-none}"
fi
echo "[hf-jobs-bench] pose noise: cube_xy=±${CUBE_XY_NOISE}m yaw=±${CUBE_YAW_NOISE}rad disk_xy=±${DISK_XY_NOISE}m"

# Detach; poll with: hf jobs logs <id> / hf jobs wait <id>
"${HF_CMD[@]}" jobs run \
  --flavor "$FLAVOR" \
  --timeout "$TIMEOUT" \
  -e HF_HUB_ENABLE_HF_TRANSFER=1 \
  -e PYOPENGL_PLATFORM=glx \
  -e REPO_ID="$REPO_ID" \
  -s HF_TOKEN \
  -d \
  "$IMAGE" \
  -- bash -c "
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq \
  git wget ca-certificates xvfb \
  libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libx11-6 \
  libxi6 libxrandr2 libxcursor1 libxinerama1 libxfixes3 \
  libegl1 libgles2 libosmesa6 >/dev/null
pip install -q --upgrade pip
pip install -q --upgrade --force-reinstall 'torch==2.6.0' 'torchvision==0.21.0' --index-url https://download.pytorch.org/whl/cu124
pip install -q 'genesis-world>=1.1.2' huggingface_hub hf_transfer Pillow numpy
mkdir -p /workspace && cd /workspace
python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(repo_id='$REPO_ID', repo_type='dataset', local_dir='/workspace/snap_dl')
PY
cp -a /workspace/snap_dl/snap /workspace/bench
cd /workspace/bench
export PYTHONPATH=/workspace/bench/armforge:/workspace/bench
mkdir -p logs
$BENCH_CMD
python - <<PY
from pathlib import Path
from huggingface_hub import HfApi
api = HfApi()
repo_id = '$REPO_ID'
path = Path('logs/$OUT_NAME')
api.upload_file(path_or_fileobj=str(path), path_in_repo=f'results/{path.name}', repo_id=repo_id, repo_type='dataset')
print(path.read_text())
print('Uploaded results ->', repo_id)
PY
"
