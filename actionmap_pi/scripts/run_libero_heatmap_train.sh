#!/bin/bash
# Training script for Pi0.5 with configurable action head on LIBERO.
#
# Supports three action head modes:
#   - pi05_heatmap_libero         (single-pass heatmap)
#   - pi05_flow_heatmap_libero    (flow matching + heatmap x0-prediction)
#   - pi05_flow_matching_libero   (vanilla Pi0.5 flow matching - baseline)
#
# Options (all optional, order-independent):
#   --config NAME          Config name (default: pi05_heatmap_libero)
#   --batch-size N         Total batch size across all GPUs (default: 256)
#   --steps N              Number of training steps (default: 20000)
#   --gpus DEVICES         Comma-separated GPU ids, e.g. "0,1" (default: "", all visible)
#   --data-ratio FLOAT     Fraction of dataset to use, in (0,1] (default: 1.0)
#   --save-interval N      Checkpoint save interval (default: 5000)
#
# Examples:
#   bash scripts/run_libero_heatmap_train.sh
#   bash scripts/run_libero_heatmap_train.sh --config pi05_flow_heatmap_libero --gpus 0,1 --data-ratio 0.1
set -e

# ================== Important ================== #
# CONFIG_NAME="pi05_flow_heatmap_libero"
# CONFIG_NAME="pi05_flow_matching_libero"
CONFIG_NAME="pi05_heatmap_libero"

TOTAL_BATCH_SIZE=256
NUM_TRAIN_STEPS=20000
SAVE_INTERVAL=5000
DATA_RATIO="0.1"

GPUS="0,1,2,3"
# ================== Important ================== #


# ---- Parse named args ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)        CONFIG_NAME="$2";      shift 2 ;;
        --batch-size)    TOTAL_BATCH_SIZE="$2";  shift 2 ;;
        --steps)         NUM_TRAIN_STEPS="$2";   shift 2 ;;
        --gpus)          GPUS="$2";              shift 2 ;;
        --data-ratio)    DATA_RATIO="$2";        shift 2 ;;
        --save-interval) SAVE_INTERVAL="$2";     shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---- GPU setup ----
if [[ -n "${GPUS}" ]]; then
    export CUDA_VISIBLE_DEVICES="${GPUS}"
fi
# Count visible GPUs
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

echo "=========================================="
echo " Config:          ${CONFIG_NAME}"
echo " Total batchsize :${TOTAL_BATCH_SIZE}  (${NUM_GPUS} GPUs)"
echo " Steps:           ${NUM_TRAIN_STEPS}"
echo " Data ratio:      ${DATA_RATIO}"
echo " Save interval:   ${SAVE_INTERVAL}"
echo " Exp name:        ${now_date}/${now_seconds}/${CONFIG_NAME}"
if [[ -n "${GPUS}" ]]; then
echo " GPUs:            ${GPUS}"
fi
echo "=========================================="

DATA_RATIO_ARG=""
if [[ "${DATA_RATIO}" != "1.0" ]]; then
    DATA_RATIO_ARG="--data.data-ratio=${DATA_RATIO}"
fi

# Use local HF cache to avoid API rate limits (dataset must be downloaded once beforehand).
export HF_HUB_OFFLINE=1

OPENPI_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# ---- Compute norm stats if missing ----
if find "$OPENPI_DIR/assets/$CONFIG_NAME" -type f -name "norm_stats.json" 2>/dev/null | grep -q .; then
    echo "[INFO] norm_stats.json already exists — skipping compute_norm_stats."
else
    echo "[INFO] Computing norm stats for ${CONFIG_NAME} ..."
    CUDA_VISIBLE_DEVICES=0 uv run scripts/compute_norm_stats.py --config-name "${CONFIG_NAME}"
    sleep 5
fi

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py "${CONFIG_NAME}" \
    --exp-name="${now_date}/${now_seconds}/${CONFIG_NAME}" \
    --batch-size="${TOTAL_BATCH_SIZE}" \
    --num-train-steps="${NUM_TRAIN_STEPS}" \
    --save-interval="${SAVE_INTERVAL}" \
    ${DATA_RATIO_ARG}

echo ""
echo "Training complete."
