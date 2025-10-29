#!/usr/bin/env bash
# Usage: bash train.sh /path/to/log_dir
# Accept only one argument: log directory
set -euo pipefail

LOG_DIR=${1:-./training_log}

# Auto-detect GPU count; fallback to 1 if not available
if command -v nvidia-smi >/dev/null 2>&1; then
  NUM_GPUS=$(nvidia-smi -L | wc -l)
else
  NUM_GPUS=1
fi

# Respect CUDA_VISIBLE_DEVICES if set
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  IFS=',' read -r -a CUDA_DEVICES_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
  NUM_GPUS=${#CUDA_DEVICES_ARRAY[@]}
fi

GPUS_PER_NODE=$(( NUM_GPUS>0 ? (NUM_GPUS<8?NUM_GPUS:8) : 1 ))

MASTER_PORT=${MASTER_PORT:-28596}

python3 -m torch.distributed.launch \
  --nproc_per_node=$GPUS_PER_NODE \
  --master_port=$MASTER_PORT \
  train.py \
  --log_dir="${LOG_DIR}" \
  --no_name_subdir \
  # specify your own args (override as needed):
  --batch_size=16 \
  --train_set=path_to_trainset \
  --valid_set=path_to_valset \
  --workers=8 \