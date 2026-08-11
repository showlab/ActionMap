#!/bin/bash
# Training script for Pi0.5 on real robot (Panda) data.
#
# Tasks:
#   build_block   "Place the orange block on top of the gray block"  (125 episodes)
#   cup           "Place the cup on the green coaster"               (92 episodes)
#   sweep         "Sweep the green block into the dustpan"           (125 episodes)
#
# Action head modes (config suffixes):
#   pi05_flow_matching_<task>   — vanilla Pi0.5 flow matching (baseline)
#   pi05_heatmap_<task>         — heatmap action head, single-pass
#   pi05_flow_heatmap_<task>    — flow matching + heatmap x0-prediction
#
# Options (all optional, order-independent):
#   --task TASK          One of: build_block, cup, sweep (default: build_block)
#   --mode MODE          One of: flow_matching, heatmap, flow_heatmap (default: flow_matching)
#   --config NAME        Override the full config name (ignores --task / --mode)
#   --batch-size N       Total batch size across all GPUs (default: 64)
#   --steps N            Number of training steps (default: 20000)
#   --gpus DEVICES       Comma-separated GPU ids, e.g. "0,1" (default: "", all visible)
#   --data-dir PATH      Absolute path to the local dataset directory (overrides config default)
#   --data-ratio FLOAT   Fraction of dataset to use, in (0,1] (default: 1.0)
#   --save-interval N    Checkpoint save interval (default: 5000)
#
# Examples:
#   # Train flow-matching on build_block with 4 GPUs (use default path from config)
#   bash scripts/run_real_robot_train.sh --task build_block --mode flow_matching --gpus 0,1,2,3
#
#   # Override the data path explicitly
#   bash scripts/run_real_robot_train.sh --task build_block --mode flow_matching \
#       --data-dir data/lerobot/build_block/5grids --gpus 0,1,2,3
#
#   # Train heatmap on cup with a custom data directory
#   bash scripts/run_real_robot_train.sh --task cup --mode heatmap \
#       --data-dir data/lerobot/cup/5grids --gpus 0,1,2,3
#
#   # Train flow-heatmap on sweep, use only 50% of data
#   bash scripts/run_real_robot_train.sh --task sweep --mode flow_heatmap --data-ratio 0.5 --gpus 0,1,2,3
#
#   # Specify a full config name directly
#   bash scripts/run_real_robot_train.sh --config pi05_flow_heatmap_build_block --gpus 0,1,2,3

set -e

# ================== Defaults ================== #
TASK="build_block"
MODE="flow_matching"
CONFIG_NAME=""           # if set, overrides TASK + MODE
TOTAL_BATCH_SIZE=64
NUM_TRAIN_STEPS=20000
SAVE_INTERVAL=5000
DATA_RATIO="1.0"
DATA_DIR=""
GPUS=""
# ============================================= #

# ---- Parse named args ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)          TASK="$2";              shift 2 ;;
        --mode)          MODE="$2";              shift 2 ;;
        --config)        CONFIG_NAME="$2";       shift 2 ;;
        --batch-size)    TOTAL_BATCH_SIZE="$2";  shift 2 ;;
        --steps)         NUM_TRAIN_STEPS="$2";   shift 2 ;;
        --gpus)          GPUS="$2";              shift 2 ;;
        --data-dir)      DATA_DIR="$2";          shift 2 ;;
        --data-ratio)    DATA_RATIO="$2";        shift 2 ;;
        --save-interval) SAVE_INTERVAL="$2";     shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---- Resolve config name ----
if [[ -z "${CONFIG_NAME}" ]]; then
    CONFIG_NAME="pi05_${MODE}_${TASK}"
fi

# Validate task and mode
VALID_TASKS="build_block cup sweep"
VALID_MODES="flow_matching heatmap flow_heatmap"
if [[ -z "$(echo ${VALID_TASKS} | grep -w "${TASK}")" ]]; then
    echo "Error: --task must be one of: ${VALID_TASKS}"
    exit 1
fi
if [[ -z "$(echo ${VALID_MODES} | grep -w "${MODE}")" ]]; then
    echo "Error: --mode must be one of: ${VALID_MODES}"
    exit 1
fi

# ---- GPU setup ----
if [[ -n "${GPUS}" ]]; then
    export CUDA_VISIBLE_DEVICES="${GPUS}"
fi
NUM_GPUS=$(python3 -c "
import os, subprocess
cvd = os.environ.get('CUDA_VISIBLE_DEVICES', '')
if cvd:
    print(len(cvd.split(',')))
else:
    out = subprocess.check_output(['nvidia-smi', '-L'], text=True)
    print(len([l for l in out.strip().splitlines() if l.startswith('GPU')]))
")

now_date=$(date +%Y%m%d)
now_seconds=$(date +%H%M%S)

OPENPI_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=========================================="
echo " Config:          ${CONFIG_NAME}"
echo " Task:            ${TASK}"
echo " Mode:            ${MODE}"
echo " Total batchsize: ${TOTAL_BATCH_SIZE}  (${NUM_GPUS} GPUs)"
echo " Steps:           ${NUM_TRAIN_STEPS}"
echo " Data ratio:      ${DATA_RATIO}"
echo " Save interval:   ${SAVE_INTERVAL}"
if [[ -n "${DATA_DIR}" ]]; then
echo " Data dir:        ${DATA_DIR}"
fi
if [[ -n "${GPUS}" ]]; then
echo " GPUs:            ${GPUS}"
fi
echo " Exp name:        ${now_date}/${now_seconds}/${CONFIG_NAME}"
echo "=========================================="

# ---- Compute norm stats if missing ----
NORM_STATS_DIR="${OPENPI_DIR}/assets/${CONFIG_NAME}"
if find "${NORM_STATS_DIR}" -type f -name "norm_stats.json" 2>/dev/null | grep -q .; then
    echo "[INFO] norm_stats.json already exists in ${NORM_STATS_DIR} — skipping compute_norm_stats."
else
    echo "[INFO] Computing norm stats for ${CONFIG_NAME} ..."
    DATA_DIR_ARG_NORM=""
    if [[ -n "${DATA_DIR}" ]]; then
        DATA_DIR_ARG_NORM="--data.local-dir=${DATA_DIR}"
    fi
    CUDA_VISIBLE_DEVICES=0 uv run scripts/compute_norm_stats.py --config-name "${CONFIG_NAME}" ${DATA_DIR_ARG_NORM}
    sleep 3
fi

# ---- Build optional overrides ----
EXTRA_ARGS=""
if [[ -n "${DATA_DIR}" ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} --data.local-dir=${DATA_DIR}"
fi
if [[ "${DATA_RATIO}" != "1.0" ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} --data.data-ratio=${DATA_RATIO}"
fi

# ---- Launch training ----
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py "${CONFIG_NAME}" \
    --exp-name="${now_date}/${now_seconds}/${CONFIG_NAME}" \
    --batch-size="${TOTAL_BATCH_SIZE}" \
    --num-train-steps="${NUM_TRAIN_STEPS}" \
    --save-interval="${SAVE_INTERVAL}" \
    ${EXTRA_ARGS}

echo ""
echo "Training complete: ${CONFIG_NAME}"
