# Deploying ArmForge on AMD Developer Cloud

ArmForge targets **AMD Developer Cloud** (Instinct MI300X / ROCm) and local Radeon GPUs.
Cloud access is provisioned through the [AMD AI Developer Program](https://www.amd.com/en/developer/resources/cloud-access/amd-developer-cloud.html)
(pay-as-you-go via DigitalOcean / Vultr, or complimentary credits).

This repo cannot create your AMD account or claim credits for you. Once you have an SSH-accessible
ROCm VM, the steps below deploy and run the cube-disk task.

## 1. Get a GPU VM

1. Join the [AMD AI Developer Program](https://www.amd.com/en/developer/resources/cloud-access/amd-developer-cloud.html).
2. Claim credits (optional): [How to claim AMD cloud credits](https://www.amd.com/en/developer/resources/technical-articles/2026/how-to-claim-amd-cloud-credits.html).
3. Launch an AMD GPU instance (DigitalOcean or Vultr from the AMD portal).
4. Prefer an image with Docker + ROCm, or a vanilla Ubuntu 24.04 and use our Dockerfile.

## 2. Clone and bootstrap

```bash
ssh <user>@<amd-cloud-host>
git clone https://github.com/alexis779/slobot-armforge.git
cd slobot-armforge
bash scripts/cloud_bootstrap.sh          # build image + open shell
# or
bash scripts/cloud_bootstrap.sh train    # PPO teacher on amdgpu
bash scripts/cloud_bootstrap.sh bench    # FPS report -> logs/bench.json
bash scripts/cloud_bootstrap.sh eval     # rollout + video
```

Inside the container always use:

```bash
python armforge/train.py --backend amdgpu ...
# equivalent to gs.init(backend=gs.amdgpu)
```

## 3. Headless rendering

Cloud VMs are headless. The Docker image includes `xvfb` and GL/Vulkan libs. If Genesis
fails to open a GL context:

```bash
xvfb-run -a python armforge/benchmark_fps.py --backend amdgpu -B 64 --steps 50
```

## 4. Hackathon GPU pool

For the AMD AI DevMaster Hackathon, eligible participants may receive temporary Radeon GPU
access via the event channels. Point that VM at this repo with the same bootstrap script.

## 5. Sanity checks

```bash
python -c "import torch; print(torch.__version__, torch.version.hip); print(torch.cuda.is_available())"
python -c "import genesis as gs; gs.init(backend=gs.amdgpu); print(gs.backend, gs.device)"
```

`torch.version.hip` must be non-`None` and Genesis must report `gs.amdgpu`.
