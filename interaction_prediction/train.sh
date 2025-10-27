#!/usr/bin/env bash
# -------------------------------------------------- #
GPUS=$1                                              #
# -------------------------------------------------- #
GPUS_PER_NODE=$(($GPUS<8?$GPUS:8))

MASTER_PORT=${MASTER_PORT:-28596}

if command -v torchrun >/dev/null 2>&1; then
  LAUNCHER=(torchrun --nproc_per_node="$GPUS_PER_NODE" --master_port="$MASTER_PORT")
else
  LAUNCHER=(python3 -m torch.distributed.launch --nproc_per_node="$GPUS_PER_NODE" --master_port="$MASTER_PORT")
fi

# Resolve dataset directories from env vars or exit early
TRAIN_SET_DIR=${TRAIN_SET:-path_to_trainset}
VALID_SET_DIR=${VALID_SET:-path_to_valset}

if [[ "$TRAIN_SET_DIR" == path_to_trainset || ! -d "$TRAIN_SET_DIR" ]]; then
  echo "[ERROR] TRAIN_SET directory is not set or does not exist: $TRAIN_SET_DIR" >&2
  echo "        Set TRAIN_SET=/abs/path/to/train_data_dir before running, or pass --train_set." >&2
  exit 2
fi

if [[ "$VALID_SET_DIR" == path_to_valset || ! -d "$VALID_SET_DIR" ]]; then
  echo "[ERROR] VALID_SET directory is not set or does not exist: $VALID_SET_DIR" >&2
  echo "        Set VALID_SET=/abs/path/to/val_data_dir before running, or pass --valid_set." >&2
  exit 2
fi

"${LAUNCHER[@]}" \
  train.py \
  --batch_size=16 \
  --train_set="$TRAIN_SET_DIR" \
  --valid_set="$VALID_SET_DIR" \
  --name=exp_log_name \
  --workers=8 \
  "$@"