#!/usr/bin/env bash
# Launch ArmForge hybrid-Cartesian PPO training on Hugging Face Jobs (NVIDIA).
#
# Prerequisites:
#   export HF_TOKEN=...   # or hf auth login
#   pip/uv: huggingface_hub[cli]
#
# Usage:
#   bash scripts/hf_jobs_train.sh
#   FLAVOR=a10g-large ITERS=2700 N_ENVS=4096 bash scripts/hf_jobs_train.sh
#
# Defaults target ~1h train wall from prior joint-PPO logs at N_ENVS=4096
# (steady ~1.33 s/iter → ITERS≈2700). No on-job probe (cold start ~15 min).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HF="${HF_BIN:-hf}"
FLAVOR="${FLAVOR:-a10g-small}"
ITERS="${ITERS:-2700}"
N_ENVS="${N_ENVS:-4096}"
TIMEOUT="${TIMEOUT:-2h}"
IMAGE="${IMAGE:-pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime}"
REPO_URL="${REPO_URL:-https://github.com/alexis779/slobot-armforge.git}"
REPO_BRANCH="${REPO_BRANCH:-hybrid-mdp-ppo}"
EXP_NAME="${EXP_NAME:-armforge_so101_hf_hybrid}"
HUB_REPO="${HUB_REPO:-armforge-so101-rl-hybrid}"
# Timing provenance for budget.json (from HF job 6a740701… joint PPO @ 4096).
SEC_PER_ITER="${SEC_PER_ITER:-1.33}"
TRAIN_BUDGET_S="${TRAIN_BUDGET_S:-3600}"
SEC_PER_ITER_SOURCE="${SEC_PER_ITER_SOURCE:-hf_job_6a740701_joint_ppo_n4096}"

echo "[hf-jobs] flavor=$FLAVOR iters=$ITERS n_envs=$N_ENVS timeout=$TIMEOUT"
echo "[hf-jobs] branch=$REPO_BRANCH image=$IMAGE exp=$EXP_NAME hub=$HUB_REPO"
echo "[hf-jobs] budget: sec_per_iter=$SEC_PER_ITER source=$SEC_PER_ITER_SOURCE train_budget_s=$TRAIN_BUDGET_S"

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
git clone --depth 1 --branch '$REPO_BRANCH' '$REPO_URL' /workspace/slobot-armforge
cd /workspace/slobot-armforge
mkdir -p logs/${EXP_NAME}_rl
python - <<'PY'
import json
from pathlib import Path
Path('logs/${EXP_NAME}_rl/budget.json').write_text(json.dumps({
    'sec_per_iter': float('$SEC_PER_ITER'),
    'sec_per_iter_source': '$SEC_PER_ITER_SOURCE',
    'inferred_iters': int('$ITERS'),
    'train_budget_s': int('$TRAIN_BUDGET_S'),
    'n_envs': int('$N_ENVS'),
    'control_mode': 'hybrid_cartesian',
}, indent=2) + '\n')
print('Wrote budget.json')
PY
xvfb-run -a python armforge/train.py --stage rl --algo ppo --backend cuda -B $N_ENVS --max_iterations $ITERS -e $EXP_NAME
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
