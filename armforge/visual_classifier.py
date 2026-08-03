"""Visual success classifier for ArmForge kitchen tasks.

Trains a small CNN on episode-camera RGB to predict task success / failure.
Can label frames online from privileged env success, or from recorded NPZ
episodes that include a `success` flag (0/1 per frame or per episode).
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from backend import add_backend_arg, init_genesis
from so101_env import SO101KitchenEnv
from configs import get_task_cfgs


class SuccessClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(64, 1)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        x = self.encoder(rgb).flatten(1)
        return self.head(x).squeeze(-1)


def collect_labeled_frames(env: SO101KitchenEnv, n_steps: int, random_actions: bool = True):
    rgbs, labels = [], []
    obs = env.reset()
    for _ in range(n_steps):
        if random_actions:
            actions = torch.randn(env.num_envs, env.num_actions, device=env.device) * 0.3
        else:
            actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        obs, _r, _d, infos = env.step(actions)
        rgb = env.get_rgb_images(normalize=True)
        success = infos["success"]
        rgbs.append(rgb.detach().cpu())
        labels.append(success.detach().cpu())
    return torch.cat(rgbs, dim=0), torch.cat(labels, dim=0)


def load_demo_labels(demo_dir: Path):
    """Load NPZ demos; label last 10% of each episode as success if `success` missing."""
    rgbs, labels = [], []
    for path in sorted(demo_dir.glob("*.npz")):
        data = np.load(path)
        rgb = torch.as_tensor(data["rgb"], dtype=torch.float32)
        if "success" in data:
            lab = torch.as_tensor(data["success"], dtype=torch.float32)
        else:
            lab = torch.zeros(rgb.shape[0], dtype=torch.float32)
            lab[int(0.9 * rgb.shape[0]) :] = 1.0
        rgbs.append(rgb)
        labels.append(lab)
    if not rgbs:
        raise FileNotFoundError(f"No NPZ episodes in {demo_dir}")
    return torch.cat(rgbs, dim=0), torch.cat(labels, dim=0)


def main():
    parser = argparse.ArgumentParser(description="Train visual success classifier")
    parser.add_argument("--out", type=str, default="logs/armforge_success_clf.pt")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--task", type=str, default="cube_disk", choices=["cube_disk"])
    parser.add_argument("--demo_dir", type=str, default=None, help="Optional NPZ demo directory")
    parser.add_argument("-B", "--num_envs", type=int, default=8)
    add_backend_arg(parser)
    args = parser.parse_args()

    init_genesis(backend=args.backend, performance_mode=True)

    if args.demo_dir:
        rgb, labels = load_demo_labels(Path(args.demo_dir))
    else:
        env_cfg, reward_cfg, robot_cfg = get_task_cfgs(args.task)
        env_cfg["num_envs"] = args.num_envs
        env = SO101KitchenEnv(env_cfg=env_cfg, reward_cfg=reward_cfg, robot_cfg=robot_cfg, show_viewer=False)
        rgb, labels = collect_labeled_frames(env, n_steps=args.steps)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SuccessClassifier().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    rgb = rgb.to(device)
    labels = labels.to(device)
    # Balance if possible
    pos = labels > 0.5
    if pos.any() and (~pos).any():
        n = min(int(pos.sum()), int((~pos).sum()))
        idx_pos = torch.where(pos)[0][:n]
        idx_neg = torch.where(~pos)[0][:n]
        idx = torch.cat([idx_pos, idx_neg])
        rgb, labels = rgb[idx], labels[idx]

    model.train()
    batch = 64
    for epoch in range(args.epochs):
        perm = torch.randperm(rgb.shape[0], device=device)
        total_loss = 0.0
        n_batches = 0
        for i in range(0, rgb.shape[0], batch):
            b = perm[i : i + batch]
            logits = model(rgb[b])
            loss = F.binary_cross_entropy_with_logits(logits, labels[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            n_batches += 1
        with torch.no_grad():
            pred = (torch.sigmoid(model(rgb)) > 0.5).float()
            acc = (pred == labels).float().mean().item()
        print(f"epoch {epoch + 1:03d} loss={total_loss / max(n_batches, 1):.4f} acc={acc:.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "task": args.task}, out)
    print(f"Saved classifier to {out}")


if __name__ == "__main__":
    main()
