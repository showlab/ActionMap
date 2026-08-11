#!/bin/bash
# Inference script for Pi0.5 on real Franka robot.
#
# Tasks and their prompts:
#   build_block  "Place the orange block on top of the gray block"
#   cup          "Place the cup on the green coaster"
#   sweep        "Sweep the green block into the dustpan"
#
# Options:
#   --task TASK        One of: build_block, cup, sweep (default: build_block)
#   --mode MODE        One of: flow_matching, heatmap, flow_heatmap (default: flow_matching)
#   --config NAME      Override full config name (ignores --task / --mode)
#   --step N           Checkpoint step to load (default: 20000)
#   --exp-name NAME    Experiment subdirectory under checkpoints/<config>/ (if not set,
#                      the script will pick the latest one automatically)
#   --ckpt-path PATH   Override the full checkpoint path (ignores --step / --exp-name)
#   --nuc-ip IP        NUC IP address (default: 192.168.1.112)
#   --instruction STR  Override the task instruction prompt
#   --timesteps N      Max rollout steps (default: 600)
#   --no-video         Disable video saving
#
# Examples:
#   # Basic: run flow_matching policy for build_block
#   bash run_real_infer.sh --task build_block --mode flow_matching
#
#   # Run heatmap policy for cup task, explicit step
#   bash run_real_infer.sh --task cup --mode heatmap --step 15000
#
#   # Full override
#   bash run_real_infer.sh --config pi05_flow_heatmap_sweep \
#       --ckpt-path ./checkpoints/pi05_flow_heatmap_sweep/<date>/<time>/pi05_flow_heatmap_sweep/20000

set -e

# ---- Camera serials (update if cameras change) ----
EXTERNAL_CAM="327122079691"
WRIST_CAM="218622273043"
LEFT_CAM="317222075319"

# ---- Defaults ----
TASK="build_block"
MODE="flow_matching"
CONFIG_NAME=""
STEP=20000
EXP_NAME=""
CKPT_PATH_OVERRIDE=""
NUC_IP="192.168.1.112"
INSTRUCTION=""
MAX_TIMESTEPS=600
SAVE_VIDEO="--save-video"

# ---- Per-task instructions ----
declare -A TASK_INSTRUCTIONS
TASK_INSTRUCTIONS["build_block"]="Place the orange block on top of the gray block"
TASK_INSTRUCTIONS["cup"]="Place the cup on the green coaster"
TASK_INSTRUCTIONS["sweep"]="Sweep the green block into the dustpan"

# ---- Parse args ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)        TASK="$2";                shift 2 ;;
        --mode)        MODE="$2";                shift 2 ;;
        --config)      CONFIG_NAME="$2";         shift 2 ;;
        --step)        STEP="$2";                shift 2 ;;
        --exp-name)    EXP_NAME="$2";            shift 2 ;;
        --ckpt-path)   CKPT_PATH_OVERRIDE="$2";  shift 2 ;;
        --nuc-ip)      NUC_IP="$2";              shift 2 ;;
        --instruction) INSTRUCTION="$2";         shift 2 ;;
        --timesteps)   MAX_TIMESTEPS="$2";       shift 2 ;;
        --no-video)    SAVE_VIDEO="";            shift   ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---- Resolve config name ----
if [[ -z "${CONFIG_NAME}" ]]; then
    CONFIG_NAME="pi05_${MODE}_${TASK}"
fi

# ---- Resolve instruction ----
if [[ -z "${INSTRUCTION}" ]]; then
    if [[ -n "${TASK_INSTRUCTIONS[$TASK]+_}" ]]; then
        INSTRUCTION="${TASK_INSTRUCTIONS[$TASK]}"
    else
        echo "Warning: no default instruction for task '${TASK}'. Use --instruction."
        INSTRUCTION="do the task"
    fi
fi

# ---- Resolve checkpoint path ----
OPENPI_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
CKPT_BASE="${OPENPI_DIR}/checkpoints/${CONFIG_NAME}"

if [[ -n "${CKPT_PATH_OVERRIDE}" ]]; then
    CKPT_PATH="${CKPT_PATH_OVERRIDE}"
elif [[ -n "${EXP_NAME}" ]]; then
    CKPT_PATH="${CKPT_BASE}/${EXP_NAME}/${STEP}"
else
    # Pick the latest experiment directory automatically
    if [[ ! -d "${CKPT_BASE}" ]]; then
        echo "Error: checkpoint base dir not found: ${CKPT_BASE}"
        echo "  Run training first, or use --ckpt-path to specify the path manually."
        exit 1
    fi
    # Latest directory = highest-sorted subdirectory (date/time format sorts correctly)
    LATEST_EXP=$(find "${CKPT_BASE}" -mindepth 2 -maxdepth 2 -type d | sort | tail -1)
    if [[ -z "${LATEST_EXP}" ]]; then
        echo "Error: no experiment subdirectory found under ${CKPT_BASE}"
        exit 1
    fi
    CKPT_PATH="${LATEST_EXP}/${STEP}"
fi

echo "=========================================="
echo " Config:      ${CONFIG_NAME}"
echo " Checkpoint:  ${CKPT_PATH}"
echo " Instruction: ${INSTRUCTION}"
echo " NUC IP:      ${NUC_IP}"
echo " Cameras:     ext=${EXTERNAL_CAM}  wrist=${WRIST_CAM}  left=${LEFT_CAM}"
echo "=========================================="

cd "${OPENPI_DIR}/examples/franka_pi05"

uv run --project "${OPENPI_DIR}" python main_pi05.py \
    --checkpoint-name "${CONFIG_NAME}" \
    --checkpoint-path "${CKPT_PATH}" \
    --instruction "${INSTRUCTION}" \
    --nuc-ip "${NUC_IP}" \
    --external-camera "${EXTERNAL_CAM}" \
    --wrist-camera "${WRIST_CAM}" \
    --left-camera "${LEFT_CAM}" \
    --max-timesteps "${MAX_TIMESTEPS}" \
    ${SAVE_VIDEO}
