#!/bin/bash

EXP_NAME='20230316_afl_taisei_cats_wooden_face_plastic_face_100epochs'
mkdir -p weights/${EXP_NAME}

python3 -m torch.distributed.launch --nproc_per_node 1 \
        --use_env main.py \
        --checkpoints weights/taisei_posem_learned/checkpoint0299.pth \
        --delete_category_weight \
        --coco_path data/train_flaptter_taisei_wooden_face_plastic_face \
        --position_embedding learned \
        --backbone resnet18 \
        --enc_layers 4 \
        --dec_layers 4 \
        --dim_feedforward 512 \
        --hidden_dim 128 \
        --nheads 4 \
        --num_queries 20 \
        --batch_size 8 \
        --epochs 100 \
        --lr_drop 50 \
        --lr 0.0001 \
        --lr_backbone 0.00001 \
        --weight_decay 0.0001 \
        --num_classes 2 \
        --output_dir ./weights/${EXP_NAME} \
        --num_workers 8\
        | tee weights/${EXP_NAME}/bash_logs.txt 

