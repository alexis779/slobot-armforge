#!/usr/bin/env bash
# Ablation: root-cause B=2048 Genesis INVALID_FORCE_NAN crash.
# Runs three configs on HF A10G and uploads JSON diagnostics.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -n "${HF_BIN:-}" ]]; then HF_CMD=("$HF_BIN")
elif command -v hf >/dev/null 2>&1; then HF_CMD=(hf)
else HF_CMD=(uv run hf); fi

FLAVOR="${FLAVOR:-a10g-small}"
TIMEOUT="${TIMEOUT:-2h}"
IMAGE="${IMAGE:-pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime}"
HUB_REPO="${HUB_REPO:-armforge-key-action-bench}"
ACTIONS_NPZ="${ACTIONS_NPZ:-datasets/so101_cube_disk/optimized/episode_000_opt.npz}"

USER_NAME="$(uv run python -c 'from huggingface_hub import whoami; print(whoami()["name"])')"
REPO_ID="${USER_NAME}/${HUB_REPO}"
SNAP_DIR="$(mktemp -d /tmp/armforge-bench-XXXXXX)"
trap 'rm -rf "$SNAP_DIR"' EXIT
mkdir -p "$SNAP_DIR/armforge" "$SNAP_DIR/datasets"
cp -a armforge/*.py "$SNAP_DIR/armforge/"
cp -a "$ACTIONS_NPZ" "$SNAP_DIR/datasets/episode_actions.npz"

uv run python - <<PY
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("$REPO_ID", repo_type="dataset", exist_ok=True, private=True)
api.upload_folder(folder_path="$SNAP_DIR", repo_id="$REPO_ID", repo_type="dataset", path_in_repo="snap")
print("uploaded snap")
PY

# Three ablations in one job (cheaper than 3x setup):
# 1) B=2048 default noise (repro)
# 2) B=2048 zero noise (isolates pose jitter)
# 3) B=2048 noise + 2x substeps via env note — actually substeps is in scene ctor;
#    for (3) re-run with smaller noise only if needed. Keep two for now + B=1536.
OUT_NAME="key_action_nan_ablation_B2048.json"

"${HF_CMD[@]}" jobs run \
  --flavor "$FLAVOR" \
  --timeout "$TIMEOUT" \
  --name "nan-ablation-B2048" \
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
apt-get install -y -qq git wget ca-certificates xvfb libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libx11-6 libxi6 libxrandr2 libxcursor1 libxinerama1 libxfixes3 libegl1 libgles2 libosmesa6 >/dev/null
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

python - <<'PY'
import json, subprocess, sys
from pathlib import Path

cfgs = [
    dict(tag='B2048_noise', B=2048, cube_xy=0.01, cube_yaw=0.1, disk_xy=0.01),
    dict(tag='B2048_nonoise', B=2048, cube_xy=0.0, cube_yaw=0.0, disk_xy=0.0),
    dict(tag='B1536_noise', B=1536, cube_xy=0.01, cube_yaw=0.1, disk_xy=0.01),
    dict(tag='B2048_noise_seed1', B=2048, cube_xy=0.01, cube_yaw=0.1, disk_xy=0.01, seed=1),
]
rows = []
for c in cfgs:
    out = Path(f\"logs/_ablate_{c['tag']}.json\")
    cmd = [
        'xvfb-run', '-a', 'python', 'armforge/benchmark_key_action.py',
        '--backend', 'cuda', '-B', str(c['B']),
        '--actions-npz', 'datasets/episode_actions.npz',
        '--cube-xy-noise', str(c['cube_xy']),
        '--cube-yaw-noise', str(c['cube_yaw']),
        '--disk-xy-noise', str(c['disk_xy']),
        '--seed', str(c.get('seed', 0)),
        '--out', str(out),
    ]
    print('[ablate]', c['tag'], flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    row = dict(c)
    row['returncode'] = proc.returncode
    # Prefer structured DIAG_JSON from our catch wrapper
    err_json = None
    blob = (proc.stderr or '') + '\n' + (proc.stdout or '')
    for line in blob.splitlines():
        if 'DIAG_JSON:' in line:
            try:
                err_json = json.loads(line.split('DIAG_JSON:', 1)[1].strip())
            except Exception:
                pass
    row['error'] = err_json
    row['stderr_tail'] = (proc.stderr or '')[-2500:]
    row['stdout_tail'] = (proc.stdout or '')[-1500:]
    if 'GenesisException' in blob:
        row['genesis_exception'] = True
    if out.is_file() and proc.returncode == 0:
        row['result'] = json.loads(out.read_text())
        row['ok'] = True
    else:
        row['ok'] = False
    rows.append(row)
    print(json.dumps({k: row[k] for k in ('tag','B','ok','returncode','error')}, indent=2), flush=True)

summary = {'ablation': 'B2048_nan_rootcause', 'rows': rows}
Path('logs/$OUT_NAME').write_text(json.dumps(summary, indent=2))
print('Wrote logs/$OUT_NAME', flush=True)
PY

python - <<PY
from pathlib import Path
from huggingface_hub import HfApi
api = HfApi()
path = Path('logs/$OUT_NAME')
api.upload_file(path_or_fileobj=str(path), path_in_repo=f'results/{path.name}', repo_id='$REPO_ID', repo_type='dataset')
print(path.read_text()[:4000])
print('Uploaded', path)
PY
"
