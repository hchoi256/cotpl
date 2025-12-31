#!/usr/bin/env bash
# bash tools/dist_test.sh configs/baron/ov_coco/baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2.py checkpoints/iter_90000.pth 2 6,7

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
GNODES=$4
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-39001}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=$GNODES python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/test.py \
    $CONFIG \
    $CHECKPOINT \
    --launcher pytorch \
    ${@:5}