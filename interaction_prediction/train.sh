#!/usr/bin/env bash
# -------------------------------------------------- #
GPUS=${1:-1}                                           # number of GPUs
TRAIN_SET=${2:-path_to_trainset}                       # processed train dir
VALID_SET=${3:-path_to_valset}                         # processed valid dir
NAME=${4:-exp_log_name}                                # experiment name

# -------------------------------------------------- #
GPUS_PER_NODE=$(($GPUS<8?$GPUS:8))
MASTER_PORT=${MASTER_PORT:-28596}

if [[ ! -d "$TRAIN_SET" ]]; then
  echo "Error: TRAIN_SET directory not found: $TRAIN_SET" 1>&2
  exit 1
fi

if [[ ! -d "$VALID_SET" ]]; then
  echo "Warning: VALID_SET directory not found: $VALID_SET. Validation will be skipped." 1>&2
fi

python3 -m torch.distributed.launch \
  --nproc_per_node=$GPUS_PER_NODE \
  --master_port=$MASTER_PORT \
  train.py \
  --batch_size=16 \
  --train_set="$TRAIN_SET" \
  --valid_set="$VALID_SET" \
  --name="$NAME" \
  --workers=8 \