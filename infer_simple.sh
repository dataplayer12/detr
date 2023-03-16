python3 infer_simple.py data/test/flaptter/20230315_oakd_rgb_hirano_wooden_and_plastic_pallets/flaptter_demo_hirano_60fps_bright/color.mp4  \
        --position_embedding learned \
        --loadpath weights/20230308_flaptter_posem_learned_300_epochs_2_cats_pretrained_taisei_posem_learned/checkpoint0099.pth \
        --num_classes 2 \
        --process_fps 2 \
        --prob_thresh 0.5 \
        --backbone resnet18 --enc_layers 4 --dec_layers 4 --dim_feedforward 512 --hidden_dim 128 --nheads 4 --num_queries 20
ffmpeg -r 2 -i infer/frame_%05d.png infer/result.mp4
rm infer/*.png