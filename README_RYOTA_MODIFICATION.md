# The summary of modification to Jaiyam's fork detr

[The original repository of Jaiyam's forked detr](https://github.com/dataplayer12/detr)
## Motivation
Thanks to Jaiyam works, qualitative evaluation was done for "Taisei" according to this [report](https://docs.google.com/document/d/1q2g1J2ThPAw6XMinEkaqzviytJDCgFqKORxx3m8BvtY/edit).

Now, I am going to modify this repository slightly for training detr and qualitative evaluation for "flapters".
The purposes of this README_RYTOA_MODIFICATION.md are

- Notes modifications so that you can find them easily
- Short instruction of how to run codes.


## Modification
### ADDED files and directories
- infer_simple.py
  
- data: dataset for training and evaluation
  - trainval
  - test

- weights: The directory saving weights confirmed.

- logs: The directory saving all logs and untested weights.

- video2png.py: Converter from video to png, not used.

### MODIFIED files and directories
- main.py
  - added some options to enable using pretained model of checkpoint and deleting the weights of categories.

## How to run
### Docker
Dockerfile did not work by 2022/03/08

### Inference
*infer_simply.py* can receive a directory having images or a video.
It runs inference over images in the input directory or the video.
It outputs the results under *infer/*.

Usage is as follows:
~~~
python3 infer_simple.py YOUR_IMAGE_DIRECTORY_PATH_OR_VIDEO_PATH  \
        --position_embedding learned \
        --loadpath YOUR_TRAINED_MODEL_PATH \
        --num_classes 2 \
        --backbone resnet18 --enc_layers 4 --dec_layers 4 --dim_feedforward 512 --hidden_dim 128 --nheads 4 --num_queries 20

~~~
e.g.,
~~~
python3 infer_simple.py data/test/flaptter/eval_video_flaptter_20220307.mp4  \
        --position_embedding learned \
        --loadpath logs/flaptter_posem_learned_50_epochs_2_cats/eval/latest.pth \
        --num_classes 2 \
        --backbone resnet18 --enc_layers 4 --dec_layers 4 --dim_feedforward 512 --hidden_dim 128 --nheads 4 --num_queries 20
~~~
The options from *--backbone* to the end are fixed to the model you used.


### Training
1. Please prepare datasets following [Data-Install](#Data-install)
2. Please check parameters and set the name of experiment in train.sh.
3. Run the following script
   ~~~
   sh train.sh
   ~~~

### Data-install
1. Collect data from cvat server using [rr_cvat](https://github.com/rapyuta-robotics/rr_cvat)
   1. Remark 1: The datasets for flaptter are 
   ~~~
   [844,808,684,677,676,675,674,673,672,671,670,669,668,666,665,664,663,661,654,653,652,650,647,646,625,624,623,622,621,]
   ~~~
   2. Remark 2: Categories are only *plastic_pallet_face* and .
2. Merge the dataset by [rr_ml_pipeline/data_utils](https://github.com/rapyuta-robotics/rr_ml_pipeline.git)
3. 



### Deploying
#### Pth->Onnx
The usage is as follows
~~~
python3 pth2onnx.py  \
  --loadpath YOUR_CHECKPOINT_PTH \
  --output_onnx YOUR_ONNX_FILE \
  --num_classes 2 \
  --position_embedding learned \
  --backbone resnet18 --enc_layers 4 --dec_layers 4 --dim_feedforward 512 --hidden_dim 128 --nheads 4 --num_queries 20
~~~

The options from *--backbone* to the end are fixed to the model you used.
The size of image is 3x 768 x 480 (c x w x h).


## TODD
- [x] Add simple demo.py
- [] Check data instller 1 hour
- [] Check train.sh 1 hour
- 
