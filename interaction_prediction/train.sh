#!/usr/bin/env bash
set -Eeuo pipefail

#
# Cloud-friendly launcher for distributed training with logging.
# - Configure via environment variables or simple arguments.
# - Captures stdout/stderr separately to designated directories.
#

# Number of GPUs on this node (positional arg 1 or env GPUS)
GPUS="${GPUS:-${1:-1}}"
GPUS_PER_NODE=$(( GPUS < 8 ? GPUS : 8 ))

# Multi-node settings (defaults work for single node)
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-28596}"

# Training hyperparams and dataset paths (override via env vars)
BATCH_SIZE="${BATCH_SIZE:-16}"
EPOCHS="${EPOCHS:-30}"
WORKERS="${WORKERS:-8}"
TRAIN_SET="${TRAIN_SET:-path_to_trainset}"
VALID_SET="${VALID_SET:-path_to_valset}"
RUN_NAME="${RUN_NAME:-exp_log_name}"

# Log directories for stdout/stderr (external to Python logging)
LOG_ROOT="${LOG_ROOT:-/workspace/logs}"
OUT_LOG_DIR="${OUT_LOG_DIR:-${LOG_ROOT}/out}"
ERR_LOG_DIR="${ERR_LOG_DIR:-${LOG_ROOT}/errors}"
mkdir -p "${OUT_LOG_DIR}" "${ERR_LOG_DIR}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_PREFIX="${RUN_NAME}_${TS}_node${NODE_RANK}"
OUT_LOG="${OUT_LOG_DIR}/${LOG_PREFIX}.out"
ERR_LOG="${ERR_LOG_DIR}/${LOG_PREFIX}.err"

# Mirror output to console and files; preserve exit codes
exec > >(tee -a "${OUT_LOG}") 2> >(tee -a "${ERR_LOG}" >&2)

echo "[INFO] Launching training: ${RUN_NAME}"
echo "[INFO] Logs: stdout -> ${OUT_LOG} ; stderr -> ${ERR_LOG}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Prefer torchrun; fallback to legacy launcher if unavailable
if command -v torchrun >/dev/null 2>&1; then
  LAUNCHER=(torchrun \
    --nproc_per_node="${GPUS_PER_NODE}" \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}")
else
  echo "[WARN] 'torchrun' not found. Falling back to python -m torch.distributed.launch"
  LAUNCHER=(python3 -m torch.distributed.launch \
    --nproc_per_node="${GPUS_PER_NODE}" \
    --master_port="${MASTER_PORT}")
fi

"${LAUNCHER[@]}" \
  "${SCRIPT_DIR}/train.py" \
  --batch_size="${BATCH_SIZE}" \
  --training_epochs="${EPOCHS}" \
  --workers="${WORKERS}" \
  --train_set="${TRAIN_SET}" \
  --valid_set="${VALID_SET}" \
  --name="${RUN_NAME}" \
  "$@"