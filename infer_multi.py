import subprocess
import os


tasks = [
    "912",
    "913",
    "914",
    "915",
    "916",
]
EXP_NAME="yolo_single_stacked_with_datagen_epoch_274_inference"
DATA_ROOT="/home/ryota/datasets/20230425_hirano_testset"
subprocess.call(["mkdir", os.path.join(DATA_ROOT, EXP_NAME)])
for task in tasks:
    subprocess.call(["rm", "-rf", "runs/detect"])
    det_cmd = [
        "python3",
        "detect.py",
        "--weights",
        "runs/train/yolov7_2/weights/epoch_274.pt",
        "--device",
        "cpu",
        "--img-size",
        "640",
        "--conf",
        "0.3",
        "--source",
        os.path.join(DATA_ROOT, task, "images"),
        "--save-txt",
    ]
    subprocess.call(det_cmd)
    subprocess.call(
        [
            "cp",
            "runs/detect/exp/instances_default.json",
            os.path.join(DATA_ROOT, EXP_NAME, task+".json"),
        ]
    )


# directory_paths = [
#     "/home/ryota/datasets/loco_dataset/subset-1/split_images",
#     "/home/ryota/datasets/loco_dataset/subset-2/split_images",
#     "/home/ryota/datasets/loco_dataset/subset-3/split_images",
#     "/home/ryota/datasets/loco_dataset/subset-4/split_images",
#     "/home/ryota/datasets/loco_dataset/subset-5/split_images",
# ]
# for directory_path in directory_paths:
#     subdirectories = []
#     for item in os.listdir(directory_path):
#         item_path = os.path.join(directory_path, item)
#         if os.path.isdir(item_path):
#             subdirectories.append(item)

#     subdirectories = sorted(subdirectories)
#     for sub_dir in subdirectories:
#         print(sub_dir)
#         subprocess.call(["rm", "-rf", "infer"])
#         subprocess.call(
#             [
#                 "python3",
#                 "infer_simple.py",
#                 os.path.join(directory_path, sub_dir, "images"),
#                 "N/A",
#                 "single_pallet",
#                 "stacked_pallets",
#                 "--position_embedding",
#                 "learned",
#                 "--loadpath",
#                 "weights/" + EXP_NAME + "/checkpoint.pth",
#                 "--num_classes",
#                 "2",
#                 "--prob_thresh",
#                 "0.3",
#                 "--device",
#                 "cpu",
#                 "--backbone",
#                 "resnet18",
#                 "--enc_layers",
#                 "4",
#                 "--dec_layers",
#                 "4",
#                 "--dim_feedforward",
#                 "512",
#                 "--hidden_dim",
#                 "128",
#                 "--nheads",
#                 "4",
#                 "--num_queries",
#                 "20",
#             ]
#         )
#         subprocess.call(
#             [
#                 "cp",
#                 "infer/instances_default.json",
#                 os.path.join(directory_path, sub_dir, "annotations/instances_default.json"),
#             ]
#         )

