# Upstream contribution package (ArmForge)

This document is ready to paste into a Genesis pull request. Opening the PR requires pushing a branch (not done automatically).

## Suggested PR title

`[FEATURE] Add ArmForge SO-101 cube-disk teleop example with ROCm path.`

## Summary

- Add `examples/armforge/`: SO-ARM-101 key-action teleop → LeRobot-style dataset on cube-disk tasks.
- Prefer `gs.amdgpu` via `--backend amdgpu|auto` and document Docker ROCm workflow; vision uses rasterizer cameras on AMD (Madrona remains CUDA-only).
- Include key-action throughput benchmark and competition writeup (`COMPETITION.md`).

## Test plan

- [ ] `cd examples/armforge && python record_lerobot.py --root datasets/so101_cube_disk` (short episode)
- [ ] `python optimize_lerobot.py --root datasets/so101_cube_disk --episode 0 --backend cpu`
- [ ] `python replay_lerobot.py --root datasets/so101_cube_disk --episode 0`
- [ ] `python benchmark_key_action.py --backend cpu -B 4`
- [ ] On AMD hardware / DevMaster GPU: repeat with `--backend amdgpu`

## Files

See `examples/armforge/` (README.md for run commands).

## LeRobot note

Episode NPZ + `meta/info.json` follow a LeRobot-shaped schema. Full parquet/mp4 Hub export is documented in `lerobot_export.py` / `LEROBOT_EXPORT.md` after teleop.
