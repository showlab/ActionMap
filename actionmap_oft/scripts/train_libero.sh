#!/bin/bash
# Finetune OpenVLA with the ActionMap voxel action heatmap head on a LIBERO task suite.
#
# Usage:
#   bash scripts/train_libero.sh <task_suite> <data_root_dir> [num_gpus]
#
# Example:
#   bash scripts/train_libero.sh libero_spatial /path/to/modified_libero_rlds 2
#
# <task_suite> is one of: libero_spatial, libero_object, libero_goal, libero_10
# <data_root_dir> is the directory holding the RLDS datasets (see LIBERO.md).
set -euo pipefail

TASK_SUITE=${1:?"usage: bash scripts/train_libero.sh <task_suite> <data_root_dir> [num_gpus]"}
DATA_ROOT_DIR=${2:?"usage: bash scripts/train_libero.sh <task_suite> <data_root_dir> [num_gpus]"}
NUM_GPUS=${3:-2}

# The RLDS datasets carry a "_no_noops" suffix.
DATASET_NAME="${TASK_SUITE}_no_noops"

# Effective batch size is fixed at 64 = BATCH_SIZE * GRAD_ACCUMULATION_STEPS * NUM_GPUS.
BATCH_SIZE=${BATCH_SIZE:-8}
GRAD_ACCUMULATION_STEPS=${GRAD_ACCUMULATION_STEPS:-$((64 / BATCH_SIZE / NUM_GPUS))}

# Required for multi-GPU DDP on machines without peer-to-peer GPU access.
export NCCL_P2P_DISABLE=1

# Invoked through the active interpreter rather than the `torchrun` console script, so that the
# launcher always matches the Python that has PyTorch installed.
python -m torch.distributed.run --standalone --nnodes 1 --nproc-per-node "${NUM_GPUS}" \
    vla-scripts/finetune.py \
    --vla_path openvla/openvla-7b \
    --data_root_dir "${DATA_ROOT_DIR}" \
    --dataset_name "${DATASET_NAME}" \
    --run_root_dir ./runs \
    --use_l1_regression False \
    --use_diffusion False \
    --use_heatmap True \
    --heatmap_trans_grid "48,48,24" \
    --heatmap_rot_grid "24,24,24" \
    --heatmap_trans_sigma 0.10 \
    --heatmap_rot_sigma 0.10 \
    --heatmap_loss_type ce \
    --use_film False \
    --num_images_in_input 2 \
    --use_proprio True \
    --batch_size "${BATCH_SIZE}" \
    --grad_accumulation_steps "${GRAD_ACCUMULATION_STEPS}" \
    --learning_rate 5e-4 \
    --num_steps_before_decay 10000 \
    --max_steps 10001 \
    --save_freq 2000 \
    --save_latest_checkpoint_only False \
    --max_checkpoints 2 \
    --merge_lora_during_training False \
    --image_aug True \
    --use_lora True \
    --lora_rank 32 \
    --run_id_note "actionmap--${TASK_SUITE}"

# To log metrics to Weights and Biases, append the following to the command above:
#     --use_wandb True --wandb_entity <your-entity> --wandb_project <your-project>
