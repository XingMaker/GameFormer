#!/usr/bin/env bash
# -------------------------------------------------- #
GPUS=$1                                              #    
# Optional: specify log directory as 2nd argument
LOG_DIR=${2:-./training_log}
# -------------------------------------------------- #
GPUS_PER_NODE=$(($GPUS<8?$GPUS:8))

MASTER_PORT=${MASTER_PORT:-28596}

python3 -m torch.distributed.launch \
        --nproc_per_node=$GPUS_PER_NODE \
        --master_port=$MASTER_PORT \
        train.py \
        --log_dir="${LOG_DIR}" \
        # specify your own args:
        --batch_size=16 \
        --train_set=path_to_trainset \
        --valid_set=path_to_valset \
        --name=exp_log_name \
        --workers=8 \