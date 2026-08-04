#!/usr/bin/env bash
# Launch ArmForge PPO training on Hugging Face Jobs (NVIDIA GPU flavors).
#
# Prerequisites:
#   export HF_TOKEN=...   # or hf auth login
#   pip/uv: huggingface_hub[cli]
#
# Usage:
#   bash scripts/hf_jobs_train.sh
#   FLAVOR=a10g-large ITERS=500 bash scripts/hf_jobs_train.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HF="${HF_BIN:-hf}"
FLAVOR="${FLAVOR:-a10g-small}"
ITERS="${ITERS:-300}"
N_ENVS="${N_ENVS:-256}"
TIMEOUT="${TIMEOUT:-3h}"
IMAGE="${IMAGE:-pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime}"
REPO_URL="${REPO_URL:-https://github.com/alexis779/slobot-armforge.git}"
EXP_NAME="${EXP_NAME:-armforge_so101_hf_joint_v2}"
HUB_REPO="${HUB_REPO:-armforge-so101-rl-joint}"

echo "[hf-jobs] flavor=$FLAVOR iters=$ITERS n_envs=$N_ENVS timeout=$TIMEOUT"
echo "[hf-jobs] image=$IMAGE exp=$EXP_NAME hub=$HUB_REPO"

# Detach so we can poll logs; token from env is picked up by hf CLI.
# Use `--` so hf CLI does not steal bash flags (-l is --label on jobs run).
"$HF" jobs run \
  --flavor "$FLAVOR" \
  --timeout "$TIMEOUT" \
  -e HF_HUB_ENABLE_HF_TRANSFER=1 \
  -e PYOPENGL_PLATFORM=glx \
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
# Keep the image CUDA torch; genesis/rsl must not pull a CPU/newer torch.
pip install -q --upgrade --force-reinstall 'torch==2.6.0' 'torchvision==0.21.0' --index-url https://download.pytorch.org/whl/cu124
pip install -q 'genesis-world>=1.1.2' 'rsl-rl-lib>=5.0.0' tensordict huggingface_hub Pillow
git clone --depth 1 '$REPO_URL' /workspace/slobot-armforge
cd /workspace/slobot-armforge
xvfb-run -a python armforge/train.py --stage rl --backend cuda -B $N_ENVS --max_iterations $ITERS -e $EXP_NAME
# Upload checkpoints to the Hub under the authenticated user
python - <<'PY'
from pathlib import Path
from huggingface_hub import HfApi, whoami
api = HfApi()
user = whoami()['name']
repo_id = f'{user}/$HUB_REPO'
api.create_repo(repo_id, repo_type='model', exist_ok=True)
log = Path('logs/${EXP_NAME}_rl')
api.upload_folder(folder_path=str(log), repo_id=repo_id, repo_type='model')
print('Uploaded', log, '->', repo_id)
PY
"
