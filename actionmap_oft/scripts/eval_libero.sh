#!/bin/bash
# Evaluate an ActionMap policy in a LIBERO task suite.
#
# Usage:
#   bash scripts/eval_libero.sh <task_suite> <checkpoint_path>
#
# Example:
#   bash scripts/eval_libero.sh libero_spatial ./runs/<run_name>--10000_chkpt
#
# The heatmap grid and sigma arguments must match the ones used during training.
set -euo pipefail

TASK_SUITE=${1:?"usage: bash scripts/eval_libero.sh <task_suite> <checkpoint_path>"}
CHECKPOINT=${2:?"usage: bash scripts/eval_libero.sh <task_suite> <checkpoint_path>"}

# Headless rendering for MuJoCo. Drop these when running with a display attached.
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-0}

python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint "${CHECKPOINT}" \
    --task_suite_name "${TASK_SUITE}" \
    --use_l1_regression False \
    --use_diffusion False \
    --use_heatmap True \
    --heatmap_trans_grid "48,48,24" \
    --heatmap_rot_grid "24,24,24" \
    --heatmap_trans_sigma 0.10 \
    --heatmap_rot_sigma 0.10 \
    --heatmap_decode_mode topk \
    --heatmap_decode_top_k 10 \
    --use_film False \
    --num_images_in_input 2 \
    --use_proprio True \
    --lora_rank 32 \
    --center_crop True \
    --num_trials_per_task 50 \
    --seed 7
