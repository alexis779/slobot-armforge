# Upstream contribution package (ArmForge)

This document is ready to paste into a Genesis pull request. Opening the PR requires pushing a branch (not done automatically).

## Suggested PR title

`[FEATURE] Add ArmForge SO-101 kitchen manipulation example with ROCm path.`

## Summary

- Add `examples/armforge/`: SO-ARM-101 Cartesian teleop → LeRobot-style dataset → privileged PPO → vision BC on kitchen cube-disk tasks.
- Prefer `gs.amdgpu` via `--backend amdgpu|auto` and document Docker ROCm workflow; vision uses rasterizer cameras on AMD (Madrona remains CUDA-only).
- Include FPS benchmark script, visual success classifier, and competition writeup (`COMPETITION.md`).

## Test plan

- [ ] `cd examples/armforge && python benchmark_fps.py --backend cpu -B 16 --steps 50`
- [ ] `python train.py --stage rl -B 8 --max_iterations 5 --backend cpu`
- [ ] `python train.py --stage bc --max_iterations 3 --backend cpu` (after RL)
- [ ] `python visual_classifier.py --backend cpu -B 4 --steps 40 --epochs 3`
- [ ] On AMD hardware / DevMaster GPU: repeat with `--backend amdgpu`

## Files

See `examples/armforge/` (README.md for run commands).

## LeRobot note

Episode NPZ + `meta/info.json` follow a LeRobot-shaped schema. Full parquet/mp4 Hub export is documented in `lerobot_export.py` / `LEROBOT_EXPORT.md` after teleop.
