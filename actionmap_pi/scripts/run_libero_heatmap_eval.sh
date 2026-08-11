#!/bin/bash
# Evaluation for Pi0.5 heatmap on LIBERO.
#
# Requires TWO terminal windows because the LIBERO simulation needs Python 3.8
# while the policy server runs in the project venv (Python 3.11 + JAX).
#
# This script handles BOTH roles via the --mode flag:
#   --mode server   → Launch the policy server (main project venv)
#   --mode client   → Run LIBERO simulation client (Python 3.8 venv)
#   --mode both     → Launch server in background, then run client (default)
#
# Usage:
#   source examples/libero/.venv/bin/activate
#
#   # Option A: Single script (launches server in background, runs client)
#   bash scripts/run_libero_heatmap_eval.sh <CONFIG_NAME> <CHECKPOINT_DIR> [options]
#
#   # Option B: Two-terminal (recommended for debugging)
#   # Terminal 1 (server):
#   bash scripts/run_libero_heatmap_eval.sh --mode server <CONFIG_NAME> <CHECKPOINT_DIR>
#   # Terminal 2 (client):
#   bash scripts/run_libero_heatmap_eval.sh --mode client
#
# Arguments:
#   CONFIG_NAME     Config name (required for server/both modes)
#   CHECKPOINT_DIR  Path to checkpoint dir (required for server/both modes)
#
# Options:
#   --mode MODE         server | client | both (default: both)
#   --port PORT         Server port (default: 8000)
#   --suite SUITE       Single suite to eval (default: all 4)
#   --trials N          Trials per task (default: 50)
#   --gpu GPU_ID        GPU id for server (default: 0)
#   --mujoco-gl MODE    EGL or glx (default: egl)
#
# Examples:
#   # All-in-one:
#   bash scripts/run_libero_heatmap_eval.sh pi05_flow_heatmap_libero ./checkpoints/pi05_flow_heatmap_libero/<date>/<time>/pi05_flow_heatmap_libero/10000 --trials 50
#
#   Two terminals
#   Server:
#   [Ours]:
#   bash scripts/run_libero_heatmap_eval.sh --mode server pi05_flow_heatmap_libero ./checkpoints/pi05_flow_heatmap_libero/<date>/<time>/pi05_flow_heatmap_libero/10000 --gpu 0
#   bash scripts/run_libero_heatmap_eval.sh --mode server pi05_flow_heatmap_libero ./checkpoints/pi05_heatmap_libero/<date>/<time>/pi05_heatmap_libero/10000 --gpu 0
#   [PI0.5 flow matching baseline]:
#   bash scripts/run_libero_heatmap_eval.sh --mode server pi05_flow_matching_libero ./checkpoints/pi05_flow_matching_libero/<date>/<time>/pi05_flow_matching_libero/10000 --gpu 1
#
#   Client (same for both):
#   bash scripts/run_libero_heatmap_eval.sh --mode client --suite "" --trials 50


set -e

# ---- Defaults ----
MODE="both"
PORT=8811
SUITE=""
TRIALS=50
GPU_ID=0
MUJOCO_GL="egl"
CONFIG_NAME=""
CHECKPOINT_DIR=""

# ---- Parse args ----
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)       MODE="$2";       shift 2 ;;
        --port)       PORT="$2";       shift 2 ;;
        --suite)      SUITE="$2";      shift 2 ;;
        --trials)     TRIALS="$2";     shift 2 ;;
        --gpu)        GPU_ID="$2";     shift 2 ;;
        --mujoco-gl)  MUJOCO_GL="$2";  shift 2 ;;
        -*)           echo "Unknown option: $1"; exit 1 ;;
        *)            POSITIONAL+=("$1"); shift ;;
    esac
done

# Positional args: CONFIG_NAME CHECKPOINT_DIR
if [[ ${#POSITIONAL[@]} -ge 1 ]]; then CONFIG_NAME="${POSITIONAL[0]}"; fi
if [[ ${#POSITIONAL[@]} -ge 2 ]]; then CHECKPOINT_DIR="${POSITIONAL[1]}"; fi

# ---- Determine suites ----
if [[ -n "${SUITE}" ]]; then
    SUITES=("${SUITE}")
else
    SUITES=("libero_spatial" "libero_object" "libero_goal" "libero_10")
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LIBERO_VENV="${PROJECT_DIR}/examples/libero/.venv"

# ============================================================
# SERVER MODE
# ============================================================
start_server() {
    if [[ -z "${CONFIG_NAME}" || -z "${CHECKPOINT_DIR}" ]]; then
        echo "ERROR: CONFIG_NAME and CHECKPOINT_DIR are required for server mode."
        echo "Usage: $0 --mode server <CONFIG_NAME> <CHECKPOINT_DIR>"
        exit 1
    fi
    echo "=========================================="
    echo " Policy Server"
    echo " Config:     ${CONFIG_NAME}"
    echo " Checkpoint: ${CHECKPOINT_DIR}"
    echo " Port:       ${PORT}"
    echo " GPU:        ${GPU_ID}"
    echo "=========================================="
    cd "${PROJECT_DIR}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    uv run scripts/serve_policy.py \
        --env LIBERO \
        --port "${PORT}" \
        policy:checkpoint \
        --policy.config "${CONFIG_NAME}" \
        --policy.dir "${CHECKPOINT_DIR}"
}

# ============================================================
# CLIENT MODE
# ============================================================
run_client() {
    # Check LIBERO venv exists
    if [[ ! -f "${LIBERO_VENV}/bin/python" ]]; then
        echo "ERROR: LIBERO venv not found at ${LIBERO_VENV}"
        echo "Set it up first:"
        echo "  uv venv --python 3.8 examples/libero/.venv"
        echo "  source examples/libero/.venv/bin/activate"
        echo "  uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt \\"
        echo "    --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match"
        echo "  uv pip install -e packages/openpi-client -e third_party/libero"
        exit 1
    fi

    echo "=========================================="
    echo " LIBERO Simulation Client"
    echo " Port:    ${PORT}"
    echo " Suites:  ${SUITES[*]}"
    echo " Trials:  ${TRIALS}"
    echo " GL:      ${MUJOCO_GL}"
    echo "=========================================="

    cd "${PROJECT_DIR}"
    export PYTHONPATH="${PYTHONPATH}:${PROJECT_DIR}/third_party/libero"
    export MUJOCO_GL="${MUJOCO_GL}"

    # Associative array to collect per-suite results
    declare -A SUITE_RESULTS

    for SUITE_NAME in "${SUITES[@]}"; do
        echo ""
        echo "---------- Evaluating: ${SUITE_NAME} ----------"
        LOG_FILE=$(mktemp)
        "${LIBERO_VENV}/bin/python" examples/libero/main.py \
            --args.port "${PORT}" \
            --args.task-suite-name "${SUITE_NAME}" \
            --args.num-trials-per-task "${TRIALS}" \
            --args.seed 7 \
            --args.video-out-path "data/libero_heatmap/videos/${SUITE_NAME}" \
            2>&1 | tee "${LOG_FILE}"

        # Extract "Total success rate: <float>" from the log
        RATE=$(grep -oP 'Total success rate: \K[0-9.]+' "${LOG_FILE}" | tail -1)
        if [[ -z "${RATE}" ]]; then
            SUITE_RESULTS["${SUITE_NAME}"]="N/A"
        else
            # Convert to percentage
            PCT=$(python3 -c "print(f'{float(${RATE})*100:.1f}')")
            SUITE_RESULTS["${SUITE_NAME}"]="${PCT}"
        fi
        rm -f "${LOG_FILE}"
        echo "---------- Done: ${SUITE_NAME} ----------"
    done

    # ---- Print and save results table ----
    RESULTS_DIR="${PROJECT_DIR}/data/libero_heatmap"
    mkdir -p "${RESULTS_DIR}"
    RESULTS_FILE="${RESULTS_DIR}/eval_results_$(date +%Y%m%d_%H%M%S).txt"

    # Use a function to print to both stdout and file
    _print_table() {
        echo ""
        echo "=========================================="
        echo "          LIBERO Evaluation Results"
        echo "=========================================="
        printf "%-20s %10s\n" "Suite" "Accuracy"
        printf "%-20s %10s\n" "--------------------" "----------"

        SUM=0
        COUNT=0
        for SUITE_NAME in "${SUITES[@]}"; do
            PCT="${SUITE_RESULTS[${SUITE_NAME}]}"
            printf "%-20s %9s%%\n" "${SUITE_NAME}" "${PCT}"
            if [[ "${PCT}" != "N/A" ]]; then
                SUM=$(python3 -c "print(${SUM} + ${PCT})")
                COUNT=$((COUNT + 1))
            fi
        done

        printf "%-20s %10s\n" "--------------------" "----------"
        if [[ ${COUNT} -gt 0 ]]; then
            AVG=$(python3 -c "print(f'{${SUM} / ${COUNT}:.1f}')")
            printf "%-20s %9s%%\n" "Average" "${AVG}"
        else
            printf "%-20s %10s\n" "Average" "N/A"
        fi
        echo "=========================================="
    }

    _print_table
    _print_table > "${RESULTS_FILE}"
    echo ""
    echo "Results saved to: ${RESULTS_FILE}"
}

# ============================================================
# BOTH MODE (server in background + client)
# ============================================================
run_both() {
    if [[ -z "${CONFIG_NAME}" || -z "${CHECKPOINT_DIR}" ]]; then
        echo "ERROR: CONFIG_NAME and CHECKPOINT_DIR are required."
        echo "Usage: $0 <CONFIG_NAME> <CHECKPOINT_DIR> [options]"
        exit 1
    fi

    echo "[1/2] Starting policy server in background..."
    cd "${PROJECT_DIR}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    uv run scripts/serve_policy.py \
        --env LIBERO \
        --port "${PORT}" \
        policy:checkpoint \
        --policy.config "${CONFIG_NAME}" \
        --policy.dir "${CHECKPOINT_DIR}" &
    SERVER_PID=$!
    trap "echo 'Killing server (PID ${SERVER_PID})...'; kill ${SERVER_PID} 2>/dev/null || true" EXIT

    # Wait for server to be ready (poll the port)
    echo "Waiting for server on port ${PORT}..."
    for i in $(seq 1 120); do
        if python3 -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('127.0.0.1',${PORT})); s.close()" 2>/dev/null; then
            echo "Server is ready (took ~${i}s)."
            break
        fi
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "ERROR: Server process died. Check logs above."
            exit 1
        fi
        sleep 1
    done

    echo "[2/2] Running LIBERO evaluations..."
    run_client
}

# ============================================================
# DISPATCH
# ============================================================
case "${MODE}" in
    server) start_server ;;
    client) run_client ;;
    both)   run_both ;;
    *)      echo "Unknown mode: ${MODE}. Use: server, client, or both."; exit 1 ;;
esac
