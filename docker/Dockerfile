# =============================================================================
# slobot-armforge — AMD ROCm development / cloud environment
# Inspired by wangxunx/franka_fruit_pick_demo docker/Dockerfile
#
#   docker build -f docker/Dockerfile -t slobot-armforge:rocm7.2.1 .
#   docker run --rm -it --network=host \
#       --device=/dev/kfd --device=/dev/dri \
#       --group-add video --group-add render \
#       --security-opt seccomp=unconfined \
#       --cap-add=SYS_PTRACE \
#       --ipc=host --shm-size=8g \
#       -v "$PWD":/workspace/slobot-armforge \
#       -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
#       slobot-armforge:rocm7.2.1
# =============================================================================
FROM rocm/dev-ubuntu-24.04:7.2.1-complete

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl wget git \
        libgl1 libglu1-mesa libegl1 libgles2 libglvnd0 libosmesa6 \
        libglib2.0-0 libsm6 libxext6 libxrender1 libx11-6 \
        libxi6 libxrandr2 libxcursor1 libxinerama1 libxfixes3 \
        libvulkan1 vulkan-tools mesa-vulkan-drivers \
        xvfb ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -f render && usermod -aG render,video root

COPY --from=ghcr.io/astral-sh/uv:0.11.23 /uv /usr/local/bin/uv

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON=python3.12 \
    UV_LINK_MODE=copy \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

WORKDIR /workspace/slobot-armforge

COPY pyproject.toml ./
RUN uv sync --no-install-project || uv pip install -e ".[learn]"

ARG TORCH_WHL=torch-2.9.1+rocm7.2.1.lw.gitff65f5bc-cp312-cp312-linux_x86_64.whl
ARG VISION_WHL=torchvision-0.24.0+rocm7.2.1.gitb919bd0c-cp312-cp312-linux_x86_64.whl
ARG AUDIO_WHL=torchaudio-2.9.0+rocm7.2.1.gite3c6ee2b-cp312-cp312-linux_x86_64.whl
ARG TRITON_WHL=triton-3.5.1+rocm7.2.1.gita272dfa8-cp312-cp312-linux_x86_64.whl
ARG ROCM_WHL_BASE=https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1

RUN set -eux; \
    mkdir -p /tmp/wheels && cd /tmp/wheels; \
    for w in "$TORCH_WHL" "$VISION_WHL" "$AUDIO_WHL" "$TRITON_WHL"; do \
        url="${ROCM_WHL_BASE}/$(python3 -c "import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1]))" "$w")"; \
        wget -q -O "$w" "$url"; \
    done; \
    uv pip uninstall torch torchvision torchaudio triton pytorch-triton pytorch-triton-rocm || true; \
    uv pip install "$TORCH_WHL" "$VISION_WHL" "$AUDIO_WHL" "$TRITON_WHL"; \
    cd / && rm -rf /tmp/wheels

ENV PYOPENGL_PLATFORM=glx \
    HSA_ENABLE_SDMA=0

RUN python -c "import torch; print('torch', torch.__version__, 'hip', torch.version.hip); \
assert torch.version.hip is not None, 'torch is not a ROCm build!'" \
 && python -c "import genesis; print('genesis', genesis.__version__)"

CMD ["/bin/bash"]
