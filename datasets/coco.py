import os

import albumentations as A
import cv2
import numpy as np
import torchvision

from transforms import v2 as T
from transforms.convert_coco_polys_to_mask import ConvertCocoPolysToMask
from util import datapoints
from util.misc import deepcopy


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(
        self,
        img_folder,
        ann_file,
        transforms=None,
        train=False,
    ):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.prepare = ConvertCocoPolysToMask()
        self._transforms = transforms
        self._transforms = self.update_dataset(self._transforms)
        self.train = train

        if train:
            self._coco_remove_images_without_annotations()

    def update_dataset(self, transform):
        if isinstance(transform, (T.Compose, A.Compose)):
            processed_transforms = []
            for trans in transform.transforms:
                trans = self.update_dataset(trans)
                processed_transforms.append(trans)
            return type(transform)(processed_transforms)
        if hasattr(transform, "update_dataset"):
            transform.update_dataset(self)
        return transform

    def load_image(self, image_name):
        # after comparing the speed of PIL, torchvision and cv2,
        # cv2 is chosen as the default backend to load images,
        # uncomment the following code to switch among them.

        # image = Image.open(os.path.join(self.root, path)).convert('RGB')
        # image = torchvision.io.read_image(os.path.join(self.root, path))

        # To avoid deadlock between DataLoader and OpenCV
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        # image = cv2.imread(os.path.join(self.root, image_name))
        image = cv2.imdecode(np.fromfile(os.path.join(self.root, image_name), dtype=np.uint8), -1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        return image

    def get_image_id(self, item: int):
        if hasattr(self, "indices"):
            item = self.indices[item]
        image_id = self.ids[item]
        return image_id

    def load_image_and_target(self, item: int):
        image_id = self.get_image_id(item)
        # load images and annotations
        image_name = self.coco.loadImgs([image_id])[0]["file_name"]
        image = self.load_image(image_name)
        target = self.coco.loadAnns(self.coco.getAnnIds([image_id]))
        target = dict(image_id=image_id, annotations=target)
        image, target = self.prepare((image, target))
        return image, target

    def data_augmentation(self, image, target):
        # preprocess
        image = datapoints.Image(image)
        bounding_boxes = datapoints.BoundingBox(
            target["boxes"],
            format=datapoints.BoundingBoxFormat.XYXY,
            spatial_size=image.shape[-2:],
        )
        labels = target["labels"]
        if self._transforms is not None:
            image, bounding_boxes, labels = self._transforms(image, bounding_boxes, labels)

        return image.data, bounding_boxes.data, labels

    def __getitem__(self, item):
        image, target = self.load_image_and_target(item)
        image, target["boxes"], target["labels"] = self.data_augmentation(image, target)

        return deepcopy(image), deepcopy(target)

    def __len__(self):
        return len(self.indices) if hasattr(self, "indices") else len(self.ids)

    def _coco_remove_images_without_annotations(self, cat_list=None):
        def _has_only_empty_bbox(anno):
            return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

        def _count_visible_keypoints(anno):
            return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)

        min_keypoints_per_image = 10

        def _has_valid_annotation(anno):
            # if it's empty, there is no annotation
            if len(anno) == 0:
                return False
            # if all boxes have close to zero area, there is no annotation
            if _has_only_empty_bbox(anno):
                return False
            # keypoints task have a slight different critera for considering
            # if an annotation is valid
            if "keypoints" not in anno[0]:
                return True
            # for keypoint detection tasks, only consider valid images those
            # containing at least min_keypoints_per_image
            if _count_visible_keypoints(anno) >= min_keypoints_per_image:
                return True
            return False

        ids = []
        for ds_idx, img_id in enumerate(self.ids):
            ann_ids = self.coco.getAnnIds(imgIds=[img_id], iscrowd=None)
            anno = self.coco.loadAnns(ann_ids)
            if cat_list:
                anno = [obj for obj in anno if obj["category_id"] in cat_list]
            if _has_valid_annotation(anno):
                ids.append(ds_idx)

        self.indices = ids


class Object365Detection(CocoDetection):
    def load_image_and_target(self, item: int):
        image_id = self.get_image_id(item)
        # load images and annotations
        image_name = self.coco.loadImgs([image_id])[0]["file_name"]
        # NOTE: Only for object 365
        image_name = os.path.join(*image_name.split(os.sep)[-2:])
        if self.train:
            image_name = os.path.join("images/train", image_name)
        else:
            image_name = os.path.join("images/val", image_name)
        image = self.load_image(image_name)
        target = self.coco.loadAnns(self.coco.getAnnIds([image_id]))
        target = dict(image_id=image_id, annotations=target)
        image, target = self.prepare((image, target))
        return image, target

    def __getitem__(self, item):
        try:
            image, target = self.load_image_and_target(item)
        except:
            item += 1
            image, target = self.load_image_and_target(item)
        image, target["boxes"], target["labels"] = self.data_augmentation(image, target)

        return deepcopy(image), deepcopy(target)
