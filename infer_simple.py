import argparse
import glob
import os

import cv2
import matplotlib
import seaborn as sns
import torch
import torchvision.transforms as T
from models import build_model
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch import nn

# torch.set_grad_enabled(False)
# no gradients will be computed


def get_args_parser():
    parser = argparse.ArgumentParser("Set transformer detector", add_help=False)
    # Added options for inference
    parser.add_argument("input_image_dir_or_video_file_path", type=str)
    parser.add_argument("--prob_thresh", default=0.3, type=float)
    parser.add_argument("--process_fps", default=2, type=float, help='It is valid only if processing a video.')

    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--lr_backbone", default=1e-5, type=float)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--lr_drop", default=200, type=int)
    parser.add_argument(
        "--clip_max_norm", default=0.1, type=float, help="gradient clipping max norm"
    )

    # Model parameters
    parser.add_argument(
        "--frozen_weights",
        type=str,
        default=None,
        help="Path to the pretrained model. If set, only the mask head will be trained",
    )
    # * Backbone
    parser.add_argument(
        "--backbone",
        default="resnet50",
        type=str,
        help="Name of the convolutional backbone to use",
    )
    parser.add_argument(
        "--dilation",
        action="store_true",
        help="If true, we replace stride with dilation in the last convolutional block (DC5)",
    )
    parser.add_argument(
        "--position_embedding",
        default="sine",
        type=str,
        choices=("sine", "learned"),
        help="Type of positional embedding to use on top of the image features",
    )

    # * Transformer
    parser.add_argument(
        "--enc_layers",
        default=6,
        type=int,
        help="Number of encoding layers in the transformer",
    )
    parser.add_argument(
        "--dec_layers",
        default=6,
        type=int,
        help="Number of decoding layers in the transformer",
    )
    parser.add_argument(
        "--dim_feedforward",
        default=2048,
        type=int,
        help="Intermediate size of the feedforward layers in the transformer blocks",
    )
    parser.add_argument(
        "--hidden_dim",
        default=256,
        type=int,
        help="Size of the embeddings (dimension of the transformer)",
    )
    parser.add_argument(
        "--dropout", default=0.1, type=float, help="Dropout applied in the transformer"
    )
    parser.add_argument(
        "--nheads",
        default=8,
        type=int,
        help="Number of attention heads inside the transformer's attentions",
    )
    parser.add_argument(
        "--num_queries", default=100, type=int, help="Number of query slots"
    )
    parser.add_argument("--pre_norm", action="store_true")

    parser.add_argument("--export_onnx", default=False, help="exports onnx if true")

    # * Segmentation
    parser.add_argument(
        "--masks",
        action="store_true",
        help="Train segmentation head if the flag is provided",
    )

    # Loss
    parser.add_argument(
        "--no_aux_loss",
        dest="aux_loss",
        action="store_false",
        help="Disables auxiliary decoding losses (loss at each layer)",
    )
    # * Matcher
    parser.add_argument(
        "--set_cost_class",
        default=1,
        type=float,
        help="Class coefficient in the matching cost",
    )
    parser.add_argument(
        "--set_cost_bbox",
        default=5,
        type=float,
        help="L1 box coefficient in the matching cost",
    )
    parser.add_argument(
        "--set_cost_giou",
        default=2,
        type=float,
        help="giou box coefficient in the matching cost",
    )
    # * Loss coefficients
    parser.add_argument("--mask_loss_coef", default=1, type=float)
    parser.add_argument("--dice_loss_coef", default=1, type=float)
    parser.add_argument("--bbox_loss_coef", default=5, type=float)
    parser.add_argument("--giou_loss_coef", default=2, type=float)
    parser.add_argument(
        "--eos_coef",
        default=0.1,
        type=float,
        help="Relative classification weight of the no-object class",
    )

    # dataset parameters
    parser.add_argument("--dataset_file", default="coco")
    parser.add_argument(
        "--num_classes",
        type=int,
        default=90,
        help="the number of classes in the dataset",
    )
    parser.add_argument("--coco_path", type=str)
    parser.add_argument("--coco_panoptic_path", type=str)
    parser.add_argument("--remove_difficult", action="store_true")

    parser.add_argument(
        "--output_dir", default="", help="path where to save, empty for no saving"
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--loadpath", default="", help="load from checkpoint", required=True
    )
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--num_workers", default=2, type=int)

    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )

    return parser



transform = T.Compose(
    [
        T.Resize([480, 768]),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
color_map = sns.color_palette(n_colors=100)


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def detect(im, model, transform, prob_thresh=0.5):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)

    # demo model only support by default images with aspect ratio between 0.5 and 2
    # if you want to use images with an aspect ratio outside this range
    # rescale your image so that the maximum size is at most 1333 for best results
    assert (
        img.shape[-2] <= 1600 and img.shape[-1] <= 1600
    ), "demo model only supports images up to 1600 pixels on each side"

    # propagate through the model
    outputs = model(img)

    # keep only predictions with 0.7+ confidence
    probas = outputs["pred_logits"].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values >= prob_thresh

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs["pred_boxes"][0, keep], im.size)
    return probas[keep], bboxes_scaled


def draw_bounding_boxes(pil_img, prob, boxes, imname):
    plt.figure(figsize=(16, 9))
    plt.imshow(pil_img)
    ax = plt.gca()
    for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):
        cl = p.argmax()
        ax.add_patch(
            plt.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=color_map[cl], linewidth=3
            )
        )
        text = f"object_{cl}: {p[cl]:0.2f}"
        ax.text(xmin, ymin, text, fontsize=10, bbox=dict(facecolor="yellow", alpha=0.2))
    plt.axis("off")
    plt.savefig(f"./infer/{imname}")
    print(f"Done {imname}")


def exportonnx(model, path):
    model.eval()
    # print(model)
    # pdb.set_trace()
    x = Image.open("testinfer.png").convert("RGB")  # np.random.randn(1080, 1920,3)

    intensor = transform(x).unsqueeze(0)
    print(intensor.shape)
    torch.onnx.export(
        model,
        intensor,
        path,
        do_constant_folding=True,
        opset_version=14,
        input_names=["image"],
        output_names=["logits", "boxes"],
    )

    print(f"Onnx model exported to {path}")


def main(args):
    os.makedirs("./infer", exist_ok=True)
    model, criterion, postprocessors = build_model(args)
    model.to("cpu")
    # we infer on cpu because training is running in parallel and we dont want to hog GPU resoures
    model.load_state_dict(torch.load(args.loadpath, map_location="cpu")["model"])

    if args.export_onnx:
        exportonnx(model, "detrmodel.onnx")
    else:
        if os.path.isdir(args.input_image_dir_or_video_file_path) is True:
            input_image_dir = args.input_image_dir_or_video_file_path
            img_paths = sorted(glob.glob(os.path.join(input_image_dir, "*.png")))
            with torch.no_grad():
                for idx, img_path in enumerate(img_paths):
                    img = Image.open(img_path)
                    scores, boxes = detect(img, model, transform, args.prob_thresh)
                    draw_bounding_boxes(
                        img, scores, boxes, "frame_" + str(idx).zfill(5) + ".png"
                    )

        elif os.path.isfile(args.input_image_dir_or_video_file_path) is True:
            video_file_path = args.input_image_dir_or_video_file_path
            cam = cv2.VideoCapture(video_file_path)
            read_fps= cam.get(cv2.CAP_PROP_FPS)
            process_fps = args.process_fps
            thresh = read_fps / process_fps
            idx = 0
            frame_counter = 0

            while True:
                ret, frame = cam.read()
                if ret:
                    frame_counter += 1
                    if frame_counter >= thresh:
                        img_cv2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        im_pil = Image.fromarray(img_cv2)
                        scores, boxes = detect(im_pil, model, transform, args.prob_thresh)
                        draw_bounding_boxes(
                            im_pil, scores, boxes, "frame_" + str(idx).zfill(5) + ".png"
                        )
                        idx += 1
                        frame_counter = 0
                    
                else:
                    break

            # Release all space and windows once done
            cam.release()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
