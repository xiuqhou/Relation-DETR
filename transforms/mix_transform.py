import copy
import random
from typing import Any, List, Tuple

import albumentations as A
import numpy as np
import torch
from torch import Tensor, nn

from util import datapoints
from transforms import v2 as T
from util.misc import image_list_from_tensors


class BaseMixTransform(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.dataset = None
        self._original_transform = None
        self._pre_transform = None
        self.p = p

    def update_dataset(self, dataset):
        self.dataset = dataset

    @property
    def original_transform(self):
        if not self._original_transform:
            self._original_transform = copy.deepcopy(self.dataset._transforms)
        return self._original_transform

    @property
    def pre_transform(self):
        if not self._pre_transform:
            self._pre_transform = self.remove_post_transforms(self.original_transform)
        return self._pre_transform

    def remove_post_transforms(self, transform):
        if isinstance(transform, type(self)):
            return None
        if isinstance(transform, (T.Compose, A.Compose)):
            processed_transforms = []
            for trans in transform.transforms:
                trans = self.remove_post_transforms(trans)
                if not trans:
                    break
                processed_transforms.append(trans)
            return type(transform)(processed_transforms)
        return transform

    @staticmethod
    def get_images_boxes_labels_from_input(input: Any,):
        # get images, labels and boxes from input
        images = list(filter(lambda x: isinstance(x, datapoints.Image), input))
        boxes = list(filter(lambda x: isinstance(x, datapoints.BoundingBox), input))
        labels = list(
            filter(
                lambda x: not isinstance(x, (datapoints.Image, datapoints.BoundingBox)),
                input,
            )
        )

        if len(labels) != 1 or len(images) != 1 or len(boxes) != 1:
            raise ValueError(
                f"currently the input must be single datapoints.Image, datapoint.BoundingBox, labels"
            )

        return images[0].data, boxes[0].data, labels[0]


class MixUp(BaseMixTransform):
    def __init__(self, p=0.5):
        super().__init__(p=p)

    def forward(self, inputs: Any) -> Any:
        if random.uniform(0, 1) > self.p:
            return inputs
        # get images, labels and boxes from input
        images, boxes, labels = self.get_images_boxes_labels_from_input(inputs)

        # get single extra image
        index = random.randint(0, len(self.dataset) - 1)

        # a hack implementation for pre_transform
        self.dataset._transforms = self.pre_transform
        extra_images, extra_boxes, extra_labels = self.dataset.data_augmentation(
            *self.dataset.load_image_and_target(index)
        )
        self.dataset._transforms = self.original_transform

        images = [images, extra_images]
        boxes = [boxes, extra_boxes]
        labels = [labels, extra_labels]
        images, boxes, labels = self.mix_transform(images, boxes, labels)
        images = datapoints.Image(images)
        boxes = datapoints.BoundingBox(
            boxes,
            format=datapoints.BoundingBoxFormat.XYXY,
            spatial_size=images.shape[-2:],
        )
        return images, boxes, labels

    @staticmethod
    def mix_transform(images: List[Tensor], boxes: List[Tensor], labels: List[Tensor]):
        data_type = images[0].dtype
        images = image_list_from_tensors(images)
        ratios = torch.as_tensor(
            data=[np.random.beta(32.0, 32.0) for _ in range(len(images.tensors))],
            device=images.tensors.device,
            dtype=torch.float32,
        )
        ratios /= torch.sum(ratios)
        image_final = torch.sum(images.tensors * ratios[:, None, None, None], dim=0)
        box_final = torch.cat(boxes)
        label_final = torch.cat(labels)
        return image_final.to(data_type), box_final, label_final


class CachedMixUp(MixUp):
    def __init__(self, p=0.5, max_cached_images=40):
        super().__init__(p)
        self.results_cache = []
        self.max_cached_images = max_cached_images

    def clone_datapoints(self, datapoint):
        if isinstance(datapoint, (List, Tuple)):
            return type(datapoint)(self.clone_datapoints(data) for data in datapoint)
        if isinstance(datapoint, datapoints.Image):
            return datapoints.Image(datapoint.detach().clone().requires_grad_(datapoint.requires_grad))
        if isinstance(datapoint, datapoints.BoundingBox):
            return datapoints.BoundingBox.wrap_like(
                datapoint,
                datapoint.detach().clone().requires_grad_(datapoint.requires_grad),
            )
        if isinstance(datapoint, torch.Tensor):
            return datapoint.clone()

    def forward(self, inputs: Any) -> Any:
        self.results_cache.append(self.clone_datapoints(inputs))
        if len(self.results_cache) > self.max_cached_images:
            index = random.randint(0, len(self.results_cache) - 1)
            self.results_cache.pop(index)

        if len(self.results_cache) <= 4:
            return inputs

        if random.uniform(0, 1) > self.p:
            return inputs

        # get images, labels and boxes from input
        images, boxes, labels = self.get_images_boxes_labels_from_input(inputs)

        # get single extra image
        index = random.randint(0, len(self.results_cache) - 1)
        extra_images, extra_boxes, extra_labels = self.clone_datapoints(self.results_cache[index])

        images = [images, extra_images]
        boxes = [boxes, extra_boxes]
        labels = [labels, extra_labels]
        images, boxes, labels = self.mix_transform(images, boxes, labels)
        images = datapoints.Image(images)
        boxes = datapoints.BoundingBox(
            boxes,
            format=datapoints.BoundingBoxFormat.XYXY,
            spatial_size=images.shape[-2:],
        )
        return images, boxes, labels


class Mosaic(BaseMixTransform):
    def __init__(self, p=0.5, n=4):
        super().__init__(p=p)
        assert n == 4, "Currently only mosaic for n=4 is supported."
        self.n = n

    def forward(self, inputs: Any) -> Any:
        if random.uniform(0, 1) > self.p:
            return inputs
        # get images, labels and boxes from input
        images, boxes, labels = self.get_images_boxes_labels_from_input(inputs)

        # get extra images, boxes and labels
        self.dataset._transforms = self.pre_transform
        indices = self.get_indices()
        extra_data_metas = [
            self.dataset.data_augmentation(*self.dataset.load_image_and_target(index)) for index in indices
        ]
        extra_images, extra_boxes, extra_labels = list(zip(*extra_data_metas))
        self.dataset._transforms = self.original_transform

        # concat datas and extra datas to perform mosaic
        images = [images, *extra_images]
        boxes = [boxes, *extra_boxes]
        labels = [labels, *extra_labels]
        images, boxes, labels = getattr(self, f"_mosaic{self.n}")(images, boxes, labels)
        images = datapoints.Image(images)
        boxes = datapoints.BoundingBox(
            boxes,
            format=datapoints.BoundingBoxFormat.XYXY,
            spatial_size=images.shape[-2:],
        )
        return images, boxes, labels

    def get_indices(self):
        return [random.randint(0, len(self.dataset) - 1) for _ in range(self.n - 1)]

    @staticmethod
    def _mosaic4(images: List[Tensor], boxes: List[Tensor], labels: List[Tensor]):
        channel, height, width = images[0].shape
        # get average size of the max border as the output image size
        image_size = int(sum(max(image.shape[-2:]) for image in images) / len(images))
        center_y, center_x = (int(random.uniform(0.5 * image_size, 1.5 * image_size)) for _ in range(2))
        image_final = images[0].new_full((channel, image_size * 2, image_size * 2), 0)  #
        boxes_final = []
        labels_final = []
        for i in range(4):
            c, h, w = images[i].shape
            if i == 0:
                x1a, y1a, x2a, y2a = (
                    max(center_x - w, 0),
                    max(center_y - h, 0),
                    center_x,
                    center_y,
                )  # w.r.t small image
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # w.r.t. large image
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = (
                    center_x,
                    max(center_y - h, 0),
                    min(center_x + w, image_size * 2),
                    center_y,
                )
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = (
                    max(center_x - w, 0),
                    center_y,
                    center_x,
                    min(image_size * 2, center_y + h),
                )
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = (
                    center_x,
                    center_y,
                    min(center_x + w, image_size * 2),
                    min(image_size * 2, center_y + h),
                )
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            image_final[:, y1a:y2a, x1a:x2a] = images[i][:, y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]

            # update boxes and labels
            valid_flag = [b[0] >= x1b and b[1] >= y1b and b[2] < x2b and b[3] < y2b for b in boxes[i]]
            valid_flag = boxes[i].new_tensor(valid_flag, dtype=torch.bool)
            offset = boxes[i].new_tensor([x1a - x1b, y1a - y1b, x1a - x1b, y1a - y1b])
            boxes_final.append(boxes[i][valid_flag] + offset)
            labels_final.append(labels[i][valid_flag])

        image_final = datapoints.Image(image_final)
        boxes_final = datapoints.BoundingBox(
            torch.cat(boxes_final).reshape(-1, 4),  # in case of empty boxes after mosaic
            format=datapoints.BoundingBoxFormat.XYXY,
            spatial_size=image_final.shape[-2:],
        )
        labels_final = torch.cat(labels_final)

        image_final, boxes_final, labels_final = T.Resize(image_size)(image_final, boxes_final, labels_final)
        return image_final, boxes_final.data, labels_final


class CachedMosaic(Mosaic):
    def __init__(self, p=0.5, n=4, max_cached_images=40):
        super().__init__(p, n)
        self.results_cache = []
        self.max_cached_images = max_cached_images

    def get_indices(self):
        return [random.randint(0, len(self.results_cache) - 1) for _ in range(self.n - 1)]

    def clone_datapoints(self, datapoint):
        if isinstance(datapoint, (List, Tuple)):
            return type(datapoint)(self.clone_datapoints(data) for data in datapoint)
        if isinstance(datapoint, datapoints.Image):
            return datapoints.Image(datapoint.detach().clone().requires_grad_(datapoint.requires_grad))
        if isinstance(datapoint, datapoints.BoundingBox):
            return datapoints.BoundingBox.wrap_like(
                datapoint,
                datapoint.detach().clone().requires_grad_(datapoint.requires_grad),
            )
        if isinstance(datapoint, torch.Tensor):
            return datapoint.clone()

    def forward(self, inputs: Any) -> Any:
        self.results_cache.append(self.clone_datapoints(inputs))
        if len(self.results_cache) > self.max_cached_images:
            index = random.randint(0, len(self.results_cache) - 1)
            self.results_cache.pop(index)

        if len(self.results_cache) <= 4:
            return inputs

        if random.uniform(0, 1) > self.p:
            return inputs

        # get images, labels and boxes from input
        images, boxes, labels = self.get_images_boxes_labels_from_input(inputs)

        # get extra images, boxes and labels
        indices = self.get_indices()
        extra_results = [self.clone_datapoints(self.results_cache[index]) for index in indices]
        extra_images, extra_boxes, extra_labels = list(
            zip(*[self.get_images_boxes_labels_from_input(extra_inputs) for extra_inputs in extra_results])
        )

        images = [images, *extra_images]
        boxes = [boxes, *extra_boxes]
        labels = [labels, *extra_labels]
        images, boxes, labels = getattr(self, f"_mosaic{self.n}")(images, boxes, labels)
        images = datapoints.Image(images)
        boxes = datapoints.BoundingBox(
            boxes,
            format=datapoints.BoundingBoxFormat.XYXY,
            spatial_size=images.shape[-2:],
        )
        return images, boxes, labels
