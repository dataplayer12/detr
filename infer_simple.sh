python3 infer_simple.py YOUR_VIDEO.mp4  \
        --position_embedding learned \
        --loadpath YOUR_WEIGHTS.pth \
        --num_classes 2 \
        --backbone resnet18 --enc_layers 4 --dec_layers 4 --dim_feedforward 512 --hidden_dim 128 --nheads 4 --num_queries 20