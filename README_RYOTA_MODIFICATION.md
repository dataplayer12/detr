# The summary of modification to Jaiyam's fork detr


## Motivation

This wiki aims for logging my modification and for reproducing my results.

## Modification
### ADDED files and directories
- infer_simple.py
- infer_simple.sh
- README_RYOTA_MODIFICATION.md

### MODIFIED files and directories
- main.py
  - added some options to enable using pretained model of checkpoint and deleting the weights of categories.
- train.sh
## How to run
### Docker
- Build
~~~
docker build -f Dockerfile ./ -t pytorch/pytorch:1.5-cuda10.1-cudnn7-runtime-detr
~~~
- Run
~~~
docker run -it --name detr \
      --gpus all --net host --ipc host --privileged \
      -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY \
      -v $PWD:/workspace/detr/ \
      pytorch/pytorch:1.5-cuda10.1-cudnn7-runtime-detr bash
~~~

### Inference
*infer_simply.py* can receive a directory having images or a video.
It runs inference over images in the input directory or the video.
It outputs the results under *infer/*.

*infer_simple.py* needs a lot of options, so please modify infer_simple.sh and use it instead.

~~~
sh infer_simple.sh
~~~


### Training
1. Please check prepare your dataset.The dataset is supposed to be
~~~
YOUR_DATASET
├── annotations
  ├── instances_train2017.json
  └── instances_val2017.json
├── train2017
  ├── train_image_1.png 
  ├── ...
└── val2017
  ├── val_image_1.png 
  ├── ...
~~~
2. Please the parameters and set the name of experiment in train.sh.
3. Run the following script
   ~~~
   sh train.sh
   ~~~

