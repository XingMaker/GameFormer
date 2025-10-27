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

"${LAUNCHER[@]}" \
  train.py \
  --batch_size=16 \
  --train_set=path_to_trainset \
  --valid_set=path_to_valset \
  --name=exp_log_name \
  --workers=8 \