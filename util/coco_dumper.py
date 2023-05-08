import collections as cl
import json
import os
import shutil
import cv2


class COCODumper:
    def __init__(self, input_image_dir, out_path, category_name_list, format="gt", dump_image=False):
        self.out_path = out_path
        out_dir = os.path.dirname(out_path)
        self.output_image_dir = os.path.join(out_dir, "images")
        self.category_name_list = category_name_list
        self.input_image_dir = input_image_dir
        self.js = cl.OrderedDict()
        self.format = format
        self.image_list = []
        self.annotaion_list = []
        self.object_id = 1
        self.image_id = 1
        self.dump_image = dump_image
        if self.dump_image:
            os.makedirs(self.output_image_dir, exist_ok=True)

    def create_info(self):
        tmp = cl.OrderedDict()
        tmp["contributor"] = ""
        tmp["date_created"] = ""
        tmp["description"] = ""
        tmp["url"] = ""
        tmp["version"] = ""
        tmp["year"] = ""
        self.js["info"] = tmp

    def create_license(self):
        tmp = cl.OrderedDict()
        tmp["name"] = 1
        tmp["id"] = 0
        tmp["url"] = ""
        self.js["license"] = tmp
        return

    def create_categories(self):
        tmps = []
        for i, cat_name in enumerate(self.category_name_list):
            tmp = cl.OrderedDict()
            tmp["id"] = i + 1
            tmp["name"] = cat_name
            tmps.append(tmp)
        self.js["categories"] = tmps

    def create_images(self):
        self.js["images"] = self.image_list

    def create_annotations(self):
        if self.format == "gt":
            self.js["annotations"] = self.annotaion_list
        elif self.format == "dt":
            self.js = self.annotaion_list
        else:
            raise ValueError("The format is only gt or dt.")

    def add_one_image(self, image_name, img_w=None, img_h=None):
        tmp = cl.OrderedDict()

        if img_w is None or img_h is None:
            img = cv2.imread(os.path.join(self.input_image_dir, image_name))
            img_h, img_w, _ = img.shape

        tmp = cl.OrderedDict()
        tmp["license"] = 1
        tmp["id"] = self.image_id
        tmp["file_name"] = os.path.basename(image_name)
        tmp["width"] = img_w
        tmp["height"] = img_h
        tmp["date_captured"] = ""
        tmp["coco_url"] = ""
        tmp["flickr_url"] = ""
        self.image_list.append(tmp)

        if self.dump_image:
            input_img_path = os.path.join(self.input_image_dir, image_name)
            dump_img_path = os.path.join(self.output_image_dir, image_name)
            shutil.copyfile(input_img_path, dump_img_path)

    def increment_image_id(self):
        self.image_id += 1

    def add_one_annotation(self, image_id, bbox, cat_id, score=None):
        tmp = cl.OrderedDict()

        # tmp["segmentation"] = [seg_polygon]
        tmp["id"] = self.object_id
        tmp["image_id"] = image_id
        tmp["category_id"] = int(cat_id)
        tmp["area"] = bbox[2] * bbox[3]
        tmp["iscrowd"] = 0
        tmp["bbox"] = bbox
        if self.format == "dt":
            tmp["score"] = score
        self.object_id += 1
        self.annotaion_list.append(tmp)

    def add_one_image_and_add_annotations_per_image(self, image_name, img_w, img_h, bboxes, cat_ids, scores=None):
        self.add_one_image(image_name, img_w, img_h)

        if self.format == "dt":
            assert len(bboxes) == len(scores)
            for bbox, cat_id, score in zip(bboxes, cat_ids, scores):
                self.add_one_annotation(self.image_id, bbox, cat_id, score)
        else:
            for bbox in bboxes:
                self.add_one_annotation(self.image_id, cat_id, bbox)
        self.increment_image_id()

    def dump_json(self):
        if self.format == "gt":
            self.create_info()
            self.create_license()
            self.create_categories()
            self.create_images()

        self.create_annotations()

        fw = open(self.out_path, "w")
        json.dump(self.js, fw, indent=2)
