#!/bin/bash

EXP_NAME='YOUR_EXP_NAME'
mkdir -p logs/${EXP_NAME}

python3 -m torch.distributed.launch --nproc_per_node 1 \
        --use_env main.py \
        --checkpoints YOUR_CHECKPOINTS.pth \
        --delete_category_weight \
        --coco_path YOUR_DATASET \
        --position_embedding learned \
        --backbone resnet18 \
        --enc_layers 4 \
        --dec_layers 4 \
        --dim_feedforward 512 \
        --hidden_dim 128 \
        --nheads 4 \
        --num_queries 20 \
        --batch_size 8 \
        --epochs 300 \
        --num_classes 2 \
        --output_dir ./weights/${EXP_NAME} \
        --num_workers 8\
        | tee weights/${EXP_NAME}/bash_logs.txt 

