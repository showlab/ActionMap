# ActionMap-PI

Heatmap ("action map") action heads for [Pi0.5 / openpi](https://github.com/Physical-Intelligence/openpi) vision-language-action models, with experiments on the LIBERO benchmark and on a real Franka Panda robot.

This repository is a fork of [openpi](https://github.com/Physical-Intelligence/openpi) that adds:

- **Heatmap action head** (`src/openpi/models/pi0_heatmap.py`): predicts end-effector actions as spatial heatmaps over discretized translation/rotation grids, with `single_pass` and `flow_matching + heatmap` modes (`src/openpi/models/pi0_heatmap_config.py`).
- **LIBERO heatmap experiments**: training and evaluation pipelines comparing the heatmap head against the Pi0.5 flow-matching baseline ([docs/libero_heatmap_experiments.md](docs/libero_heatmap_experiments.md)).
- **Real-robot (Franka Panda) support**: training on LeRobot-format teleop data ([docs/real_robot_training.md](docs/real_robot_training.md)) and real-time inference ([examples/franka_pi05/](examples/franka_pi05/)).

The OpenVLA-OFT-based implementation of ActionMap is available in [`../actionmap_oft`](../actionmap_oft).

## Installation

Clone with submodules:

```bash
git clone --recurse-submodules git@github.com:showlab/ActionMap.git
cd ActionMap/actionmap_pi
```

We use [uv](https://docs.astral.sh/uv/) to manage the Python environment:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
source .venv/bin/activate
```

For the LIBERO evaluation environment (separate Python 3.8 env) see [docs/libero_heatmap_experiments.md](docs/libero_heatmap_experiments.md).

## Quick start: LIBERO heatmap experiments

Train (from the repository root):

```bash
bash scripts/run_libero_heatmap_train.sh \
  --config pi05_flow_heatmap_libero \
  --gpus 0,1,2,3 \
  --batch-size 256 \
  --steps 30000 \
  --save-interval 5000 \
  --data-ratio 1.0
```

Main configs (registered in `src/openpi/training/config.py`):

| Config | Description |
|--------|-------------|
| `pi05_flow_heatmap_libero` | flow matching + heatmap head (ours) |
| `pi05_heatmap_libero` | single-pass heatmap head |
| `pi05_flow_matching_libero` | Pi0.5 flow-matching baseline |

Evaluate on LIBERO (server + client; see the doc above for the two-terminal workflow):

```bash
bash scripts/run_libero_heatmap_eval.sh \
  pi05_flow_heatmap_libero \
  checkpoints/pi05_flow_heatmap_libero/<date>/<time>/pi05_flow_heatmap_libero/<step> \
  --trials 50 --port 8811 --gpu 0
```

## Real-robot (Franka Panda)

- Data format, configs, and training: [docs/real_robot_training.md](docs/real_robot_training.md) and `scripts/run_real_robot_train.sh`.
- Real-time inference on the robot: [examples/franka_pi05/](examples/franka_pi05/).

Real-robot configs follow the pattern `pi05_{flow_matching,heatmap,flow_heatmap}_{build_block,cup,sweep}` and expect LeRobot v2.0 datasets under `data/lerobot/<task>/`.

## Documentation

- [docs/libero_heatmap_experiments.md](docs/libero_heatmap_experiments.md) — LIBERO training/evaluation guide
- [docs/real_robot_training.md](docs/real_robot_training.md) — real-robot data format and training
- [docs/norm_stats.md](docs/norm_stats.md) — normalization statistics
- [docs/remote_inference.md](docs/remote_inference.md) — running policies remotely
- [docs/docker.md](docs/docker.md) — Docker setup

## Acknowledgments

This codebase is built on [openpi](https://github.com/Physical-Intelligence/openpi) by Physical Intelligence (Apache 2.0). LIBERO evaluation uses [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO).

## License

Apache 2.0 (see [LICENSE](LICENSE)). Gemma model weights are covered by [LICENSE_GEMMA.txt](LICENSE_GEMMA.txt).
