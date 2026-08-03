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

echo "[hf-jobs] flavor=$FLAVOR iters=$ITERS n_envs=$N_ENVS timeout=$TIMEOUT"
echo "[hf-jobs] image=$IMAGE"

# Detach so we can poll logs; token from env is picked up by hf CLI.
"$HF" jobs run \
  --flavor "$FLAVOR" \
  --timeout "$TIMEOUT" \
  -e HF_HUB_ENABLE_HF_TRANSFER=1 \
  -s HF_TOKEN \
  -d \
  "$IMAGE" \
  bash -lc "
set -euo pipefail
apt-get update -qq
apt-get install -y -qq git wget ca-certificates libgl1 libglib2.0-0 >/dev/null
pip install -q --upgrade pip
pip install -q 'genesis-world>=1.1.2' 'rsl-rl-lib>=5.0.0' tensordict huggingface_hub Pillow
git clone --depth 1 '$REPO_URL' /workspace/slobot-armforge
cd /workspace/slobot-armforge
python armforge/train.py --stage rl --backend cuda -B $N_ENVS --max_iterations $ITERS -e armforge_so101_hf
# Upload checkpoints to the Hub under the authenticated user
python - <<'PY'
from pathlib import Path
from huggingface_hub import HfApi, whoami
api = HfApi()
user = whoami()['name']
repo_id = f'{user}/armforge-so101-rl'
api.create_repo(repo_id, repo_type='model', exist_ok=True)
log = Path('logs/armforge_so101_hf_rl')
api.upload_folder(folder_path=str(log), repo_id=repo_id, repo_type='model')
print('Uploaded', log, '->', repo_id)
PY
"
