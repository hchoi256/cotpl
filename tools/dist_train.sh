#!/usr/bin/env bash
# CUBLAS_WORKSPACE_CONFIG=:4096:8 PORT=39320 bash tools/dist_train.sh configs/baron/ov_coco/baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2.py 2 6,7 1194806617 --work-dir work_dirs/

CONFIG=$1
GPUS=$2
NODES=$3
SEED=$4
NODE_RANK=${NODE_RANK:-0}
NNODES=${NNODES:-1}
PORT=${PORT:-39500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=$NODES python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --cfg-options randomness.seed=$SEED \
    randomness.diff_rank_seed=True \
    randomness.deterministic=True \
    --launcher pytorch ${@:5}