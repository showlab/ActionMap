# Real Robot Training (Panda)

Training Pi0.5 on real Panda robot data collected locally and stored in [LeRobot v2.0 format](https://github.com/huggingface/lerobot).

## Dataset

Data is expected under `data/lerobot/` (relative to the repository root) with one subdirectory per task:

| Task | Path | Episodes | Task instruction |
|------|------|----------|-----------------|
| `build_block` | `data/lerobot/build_block/5grids` | 125 | "Place the orange block on top of the gray block" |
| `cup` | `data/lerobot/cup/5grids` | 92 | "Place the cup on the green coaster" |
| `sweep` | `data/lerobot/sweep/5grids` | 125 | "Sweep the green block into the dustpan" |

Each dataset directory has the structure:
```
<task>/5grids/
  meta/
    info.json       # fps=15, features definitions
    tasks.jsonl     # task instruction string
    episodes.jsonl  # per-episode metadata
  data/
    chunk-000/
      episode_000000.parquet
      …
```

### Features

| Key | Type | Shape | Description |
|-----|------|-------|-------------|
| `image` | float32 | (3,224,224) | Third-person / base camera (CHW, [0,1]) |
| `wrist_image` | float32 | (3,224,224) | Right wrist camera |
| `left_image` | float32 | (3,224,224) | Left wrist camera |
| `state` | float32 | (8,) | [x, y, z, roll, pitch, yaw, gripper_open, gripper_close] |
| `actions` | float32 | (T, 7) | [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper] — already delta |

## Available Configs

Nine pre-defined configs are registered in `src/openpi/training/config.py` (3 tasks × 3 modes):

| Config name | Task | Mode |
|-------------|------|------|
| `pi05_flow_matching_build_block` | build_block | flow matching (baseline) |
| `pi05_heatmap_build_block` | build_block | heatmap single-pass |
| `pi05_flow_heatmap_build_block` | build_block | flow matching + heatmap |
| `pi05_flow_matching_cup` | cup | flow matching |
| `pi05_heatmap_cup` | cup | heatmap single-pass |
| `pi05_flow_heatmap_cup` | cup | flow matching + heatmap |
| `pi05_flow_matching_sweep` | sweep | flow matching |
| `pi05_heatmap_sweep` | sweep | heatmap single-pass |
| `pi05_flow_heatmap_sweep` | sweep | flow matching + heatmap |

All configs share these hyperparameters:
- **Model**: Pi0.5 (`action_horizon=10`, `action_dim=32`)
- **Batch size**: 64
- **Steps**: 20 000
- **LR schedule**: cosine decay, warmup 1 k steps → peak 5e-5 → decay to 3e-5
- **Weight init**: `PartialCheckpointWeightLoader` from `gs://openpi-assets/checkpoints/pi05_base/params`
- **Norm**: quantile normalization (automatically computed on first run)

## Training

### Quick start

```bash
# Flow-matching on build_block (4 GPUs)
bash scripts/run_real_robot_train.sh --task build_block --mode flow_matching --gpus 0,1,2,3

# Heatmap on cup
bash scripts/run_real_robot_train.sh --task cup --mode heatmap --gpus 0,1,2,3

# Flow-heatmap on sweep, 50% of data
bash scripts/run_real_robot_train.sh --task sweep --mode flow_heatmap --data-ratio 0.5 --gpus 0,1
```

### Script options

| Flag | Default | Description |
|------|---------|-------------|
| `--task` | `build_block` | Task name: `build_block`, `cup`, `sweep` |
| `--mode` | `flow_matching` | Mode: `flow_matching`, `heatmap`, `flow_heatmap` |
| `--config` | _(derived)_ | Override full config name, e.g. `pi05_heatmap_cup` |
| `--batch-size` | `64` | Total batch size (divided evenly across GPUs) |
| `--steps` | `20000` | Training steps |
| `--gpus` | all | Comma-separated CUDA device IDs, e.g. `0,1,2,3` |
| `--data-ratio` | `1.0` | Fraction of dataset to use (deterministic subset) |
| `--save-interval` | `5000` | Steps between checkpoint saves |

### Norm stats

On first run the script automatically calls `compute_norm_stats.py` and caches the result in
`assets/real_robot/<task>/norm_stats.json` (one file shared by all three modes of the same task,
since they use the same data pipeline).

To compute manually:
```bash
uv run scripts/compute_norm_stats.py --config-name pi05_flow_matching_build_block
```

The output is written to `./assets/real_robot/build_block/norm_stats.json` and loaded automatically
by the training configs.

### Manual training (without the script)

```bash
uv run scripts/train.py pi05_flow_heatmap_build_block \
    --exp-name=my_experiment \
    --batch-size=64 \
    --num-train-steps=20000 \
    --save-interval=5000
```

## Adding a new dataset

1. Place your data at `<root>/data/` and `<root>/meta/` (LeRobot v2.0 format).
2. Add a `LeRobotRealRobotDataConfig` entry in `src/openpi/training/config.py`:

```python
TrainConfig(
    name="pi05_flow_matching_my_task",
    model=pi0_heatmap_config.Pi0HeatmapConfig(action_head_mode="flow_matching", action_horizon=10),
    data=LeRobotRealRobotDataConfig(
        repo_id="real_robot/my_task",          # used as norm-stats key
        local_dir="/path/to/my_task/5grids",   # actual data on disk
        base_config=DataConfig(prompt_from_task=True),
    ),
    ...
)
```

3. Run `bash scripts/run_real_robot_train.sh --config pi05_flow_matching_my_task`.

## Key source files

| File | Purpose |
|------|---------|
| `src/openpi/policies/real_robot_policy.py` | `RealRobotInputs` / `RealRobotOutputs` transforms |
| `src/openpi/training/config.py` | `LeRobotRealRobotDataConfig` class + the 9 training configs |
| `src/openpi/training/data_loader.py` | Threads `local_dir` → `root` in `LeRobotDataset` |
| `scripts/run_real_robot_train.sh` | Convenience training script |
