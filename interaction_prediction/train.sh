#!/usr/bin/env bash
# -------------------------------------------------- #
GPUS=$1                                              #    
# Optional: custom log directory to override --name
LOG_DIR=${2:-}
# -------------------------------------------------- #
GPUS_PER_NODE=$(($GPUS<8?$GPUS:8))

MASTER_PORT=${MASTER_PORT:-28596}

python3 -m torch.distributed.launch \
        --nproc_per_node=$GPUS_PER_NODE \
        --master_port=$MASTER_PORT \
        train.py \
        # specify your own args:
        --log_dir="$LOG_DIR" \
        --batch_size=16 \
        --train_set=path_to_trainset \
        --valid_set=path_to_valset \
        --name=exp_log_name \
        --workers=8 \