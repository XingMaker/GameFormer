#!/usr/bin/env bash
# -------------------------------------------------- #
GPUS=$1                                              #    
# Optional: specify log directory as 2nd argument
LOG_DIR=${2:-./training_log}
# Optional: specify experiment name as 3rd argument (omit to skip)
EXP_NAME=${3:-}
# Optional: pass any non-empty 4th arg to disable name subdir
NO_NAME_SUBDIR_FLAG=${4:-}

# Build optional args
NAME_ARG=""
if [ -n "$EXP_NAME" ]; then
    NAME_ARG="--name=\"$EXP_NAME\""
fi
NO_NAME_ARG=""
if [ -n "$NO_NAME_SUBDIR_FLAG" ]; then
    NO_NAME_ARG="--no_name_subdir"
fi
# -------------------------------------------------- #
GPUS_PER_NODE=$(($GPUS<8?$GPUS:8))

MASTER_PORT=${MASTER_PORT:-28596}

python3 -m torch.distributed.launch \
        --nproc_per_node=$GPUS_PER_NODE \
        --master_port=$MASTER_PORT \
        train.py \
        --log_dir="${LOG_DIR}" \
        ${NAME_ARG} ${NO_NAME_ARG} \
        # specify your own args (override as needed):
        --batch_size=16 \
        --train_set=path_to_trainset \
        --valid_set=path_to_valset \
        --workers=8 \