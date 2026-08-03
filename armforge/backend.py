"""Genesis backend selection for ArmForge (AMD ROCm preferred)."""

from __future__ import annotations

import argparse

import torch

import genesis as gs


def add_backend_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "amdgpu", "gpu", "cuda", "cpu", "metal"],
        help="Genesis compute backend. 'auto' prefers ROCm/amdgpu when HIP is available.",
    )


def resolve_backend(name: str):
    if name == "auto":
        if getattr(torch.version, "hip", None):
            return gs.amdgpu
        return gs.gpu
    return {
        "amdgpu": gs.amdgpu,
        "gpu": gs.gpu,
        "cuda": gs.cuda,
        "cpu": gs.cpu,
        "metal": gs.metal,
    }[name]


def init_genesis(
    backend: str = "auto",
    *,
    seed: int = 1,
    performance_mode: bool = True,
    logging_level: str = "warning",
) -> None:
    gs.init(
        backend=resolve_backend(backend),
        precision="32",
        logging_level=logging_level,
        seed=seed,
        performance_mode=performance_mode,
    )
    print(f"[ArmForge] Genesis backend: {gs.backend}")
