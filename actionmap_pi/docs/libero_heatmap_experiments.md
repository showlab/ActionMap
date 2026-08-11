# LIBERO Heatmap Experiments Quick Guide

This guide is for running Pi0.5 heatmap experiments in this repository.

- Training environment: Python 3.11 project environment
- LIBERO evaluation environment: separate Python 3.8 environment
- Main scripts:
  - scripts/run_libero_heatmap_train.sh
  - scripts/run_libero_heatmap_eval.sh

## 1) Environment Setup

### 1.1 Project environment (Python 3.11)

Run from repository root:

    GIT_LFS_SKIP_SMUDGE=1 uv sync
    GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

Activate when running training/server:

    source .venv/bin/activate

### 1.2 LIBERO evaluation environment (Python 3.8)

Create and install:

    uv venv --python 3.8 examples/libero/.venv
    source examples/libero/.venv/bin/activate
    uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
    uv pip install -e packages/openpi-client -e third_party/libero

## 2) Training

### 2.1 Recommended configs

- pi05_flow_heatmap_libero: flow matching input plus heatmap x0 head
- pi05_heatmap_libero: single-pass heatmap baseline
- pi05_flow_matching_libero: Pi0.5 flow baseline

### 2.2 Recommended first run

From repository root:

    source .venv/bin/activate
    bash scripts/run_libero_heatmap_train.sh \
      --config pi05_flow_heatmap_libero \
      --gpus 0,1,2,3 \
      --batch-size 256 \
      --steps 30000 \
      --save-interval 5000 \
      --data-ratio 1.0

Notes:
- Use data-ratio 1.0 for final experiments.
- The script auto-computes norm stats if missing.
- Checkpoints are saved under checkpoints/<config>/<date>/<time>/<config>/<step>.

### 2.3 Key hyperparameters

- config: model/head mode
- gpus: visible GPUs
- batch-size: total global batch size
- steps: total train steps
- save-interval: checkpoint frequency
- data-ratio: fraction of training set used

## 3) Evaluation on LIBERO

### 3.1 Two-terminal workflow (recommended)

Terminal A (server, Python 3.11 env):

    source .venv/bin/activate
    bash scripts/run_libero_heatmap_eval.sh \
      --mode server \
      pi05_flow_heatmap_libero \
      checkpoints/pi05_flow_heatmap_libero/<date>/<time>/pi05_flow_heatmap_libero/<step> \
      --gpu 0 --port 8811

Terminal B (client, Python 3.8 env):

    source examples/libero/.venv/bin/activate
    bash scripts/run_libero_heatmap_eval.sh \
      --mode client \
      --suite libero_spatial \
      --trials 50 \
      --port 8811

### 3.2 All-in-one workflow

    source .venv/bin/activate
    bash scripts/run_libero_heatmap_eval.sh \
      pi05_flow_heatmap_libero \
      checkpoints/pi05_flow_heatmap_libero/<date>/<time>/pi05_flow_heatmap_libero/<step> \
      --trials 50 --port 8811 --gpu 0

### 3.3 Evaluate all suites

Set suite to empty string or omit it in client mode:

    bash scripts/run_libero_heatmap_eval.sh --mode client --suite "" --trials 50 --port 8811

## 4) Inference Only (Policy Server)

Run policy server directly:

    source .venv/bin/activate
    uv run scripts/serve_policy.py \
      --env LIBERO \
      --port 8811 \
      policy:checkpoint \
      --policy.config pi05_flow_heatmap_libero \
      --policy.dir checkpoints/pi05_flow_heatmap_libero/<date>/<time>/pi05_flow_heatmap_libero/<step>

## 5) Practical Experiment Rules

- Always pass config explicitly in training and eval commands.
- Prefer later checkpoints for evaluation (for example 30000 over 10000).
- Keep train/eval ports consistent.
- Confirm checkpoint directory contains params and assets.
- Compare against pi05_flow_matching_libero as baseline.

## 6) Quick Troubleshooting

- 0% success, random actions:
  - verify config and checkpoint path match
  - verify you are not evaluating an early checkpoint only
  - verify training used sufficient data-ratio and steps
- Server exits immediately:
  - check Python environment and dependency activation
- Client cannot connect:
  - check port, firewall, and server logs
- LIBERO import/runtime errors:
  - re-activate examples/libero/.venv and re-sync requirements
