import random
import copy
from typing import Dict, List

import cv2
import numpy as np
from albumentations import DualTransform
from albumentations.augmentations.crops import functional as FCrops
from albumentations.augmentations.geometric import functional as FGeometric
from albumentations.core.bbox_utils import denormalize_bboxes, normalize_bboxes


class RandomSizeCrop(DualTransform):
    def __init__(self, min_size, max_size, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.min_size = min_size
        self.max_size = max_size

    @property
    def targets_as_params(self) -> List[str]:
        return ["image"]

    def get_params(self):
        return {"h_start": random.random(), "w_start": random.random()}

    def get_params_dependent_on_targets(self, params):
        img_h, img_w = params["image"].shape[:2]
        crop_width = random.randint(self.min_size, min(img_w, self.max_size))
        crop_height = random.randint(self.min_size, min(img_h, self.max_size))
        return {"crop_height": crop_height, "crop_width": crop_width}

    def apply(self, img, crop_height=0, crop_width=0, h_start=0, w_start=0, **params):
        return FCrops.random_crop(img, crop_height, crop_width, h_start, w_start)

    def apply_to_bbox(self, bbox, **params):
        return FCrops.bbox_random_crop(bbox, **params)

    def get_transform_init_args_names(self):
        return ("min_size", "max_size")


class RandomShortestSize(DualTransform):
    def __init__(
        self, min_size, max_size, interpolation=cv2.INTER_LINEAR, always_apply=False, p=1.0
    ):
        super().__init__(always_apply, p)
        self.min_size = [min_size] if isinstance(min_size, int) else list(min_size)
        self.max_size = max_size
        self.interpolation = interpolation

    @property
    def targets_as_params(self) -> List[str]:
        return ["image"]

    def get_params_dependent_on_targets(self, params):
        img_h, img_w = params["image"].shape[:2]

        min_size = self.min_size[random.randint(0, len(self.min_size) - 1)]
        r = min_size / min(img_h, img_w)
        if self.max_size is not None:
            r = min(r, self.max_size / max(img_h, img_w))

        new_width = int(img_w * r)
        new_height = int(img_h * r)
        return {"height": new_height, "width": new_width}

    def apply(self, img, height=0, width=0, interpolation=cv2.INTER_LINEAR, **params):
        return FGeometric.resize(img, height=height, width=width, interpolation=interpolation)

    def apply_to_bbox(self, bbox, **params):
        # Bounding box coordinates are scale invariant
        return bbox

    def apply_to_keypoint(self, keypoint, **params):
        height = params["rows"]
        width = params["cols"]
        scale_x = params["width"] / width
        scale_y = params["height"] / height
        return FGeometric.keypoint_scale(keypoint, scale_x, scale_y)

    def get_transform_init_args_names(self):
        return ("min_size", "max_size", "interpolation")


class CachedMosaic(DualTransform):
    def __init__(self, n=4, max_cached_images=40, always_apply=False, p=1.0):
        super().__init__(True, p)
        assert n == 4, "currenly we only support mosaic_4"
        self.n = n
        # override always_apply to _always_apply
        # it is only used for mosaic_transform not in __call__
        # since each data should be applyied to update results_cache
        self._always_apply = always_apply
        self.results_cache = []
        self.max_cached_images = max_cached_images

    @property
    def targets_as_params(self) -> List[str]:
        return ["image", "bboxes"]

    def update_results_cache(self, inputs):
        self.results_cache.append(copy.deepcopy(inputs))
        if len(self.results_cache) > self.max_cached_images:
            index = random.randint(0, len(self.results_cache) - 1)
            self.results_cache.pop(index)

    def get_params_dependent_on_targets(self, params):
        self.update_results_cache(params)
        # judge whether to apply mosaic transform
        apply = (random.random() < self.p) or self._always_apply
        apply = apply and len(self.results_cache) > self.n
        if not apply:
            return {
                "image_sizes": None,
                "image_size": None,
                "coordinates": None,
                "extra_images": None,
                "extra_bboxes": None,
            }

        # get images and bboxes, and extra images and bboxes
        indices = [random.randint(0, len(self.results_cache) - 1) for _ in range(self.n - 1)]
        extra_params = [self.results_cache[i] for i in indices]
        params = [params] + extra_params
        images, bboxes = list(zip(*map(lambda x: (x["image"], x["bboxes"]), params)))

        # get other parameters
        image_sizes = [image.shape[:2] for image in images]
        image_size = int(sum(max(image_size) for image_size in image_sizes) / len(image_sizes))
        center_y, center_x = [
            int(random.uniform(0.5 * image_size, 1.5 * image_size)) for _ in range(2)
        ]
        # get transformed coordinates
        coordinates = []
        for i in range(self.n):
            h, w = image_sizes[i]
            if i == 0:
                # relative to large image
                x1a, y1a = max(center_x - w, 0), max(center_y - h, 0)
                x2a, y2a = center_x, center_y
                # relative to small image
                x1b, y1b = w - (x2a - x1a), h - (y2a - y1a)
                x2b, y2b = w, h
            if i == 1:
                x1a, y1a = center_x, max(center_y - h, 0)
                x2a, y2a = min(center_x + w, image_size * 2), center_y
                x1b, y1b = 0, h - (y2a - y1a)
                x2b, y2b = min(w, x2a - x1a), h
            if i == 2:
                x1a, y1a = max(center_x - w, 0), center_y
                x2a, y2a = center_x, min(image_size * 2, center_y + h)
                x1b, y1b = w - (x2a - x1a), 0
                x2b, y2b = w, min(y2a - y1a, h)
            if i == 3:
                x1a, y1a = center_x, center_y
                x2a, y2a = min(center_x + w, image_size * 2), min(image_size * 2, center_y + h)
                x1b, y1b = 0, 0
                x2b, y2b = min(w, x2a - x1a), min(y2a - y1a, h)
            coordinates.append([x1a, y1a, x2a, y2a, x1b, y1b, x2b, y2b])
        return {
            "image_sizes": image_sizes,
            "image_size": image_size,
            "coordinates": coordinates,
            "extra_images": images[1:],
            "extra_bboxes": bboxes[1:],
        }

    def apply(self, img, image_size=0, coordinates=None, extra_images=None, **params):
        if coordinates is None:
            return img
        image_final = np.zeros((image_size * 2, image_size * 2, img.shape[-1]), dtype=img.dtype)
        for (x1a, y1a, x2a, y2a, x1b, y1b, x2b, y2b), image in zip(
            coordinates, [img, *extra_images]
        ):
            image_final[y1a:y2a, x1a:x2a, :] = image[y1b:y2b, x1b:x2b, :]
        
        return FGeometric.resize(image_final, image_size, image_size)

    def apply_to_bboxes(
        self, bboxes, coordinates=None, extra_bboxes=None, image_size=0, image_sizes=None, **params
    ):
        if coordinates is None:
            return bboxes
        bboxes_final = []
        for (x1a, y1a, x2a, y2a, x1b, y1b, x2b, y2b), bboxes, (rows, cols) in zip(
            coordinates,
            [bboxes, *extra_bboxes],
            image_sizes,
        ):
            bboxes = denormalize_bboxes(bboxes, rows, cols)
            valid_flag = [b[0] >= x1b and b[1] >= y1b and b[2] < x2b and b[3] < y2b for b in bboxes]
            bboxes = [
                (b[0] + x1a - x1b, b[1] + y1a - y1b, b[2] + x1a - x1b, b[3] + y1a - y1b, *b[4:])
                for b, vf in zip(bboxes, valid_flag)
                if vf == True
            ]
            bboxes = normalize_bboxes(bboxes, image_size * 2, image_size * 2)
            bboxes_final.extend(bboxes)

        return bboxes_final


class CachedMixup(DualTransform):
    def __init__(self, max_cached_images=40, always_apply=False, p=1.0):
        super().__init__(True, p)
        # override always_apply to _always_apply
        # it is only used for mosaic_transform not in __call__
        # since each data should be applyied to update results_cache
        self._always_apply = always_apply
        self.results_cache = []
        self.max_cached_images = max_cached_images

    def get_params(self):
        ratios = [random.betavariate(32.0, 32.0) for _ in range(2)]
        ratios = [r / sum(ratios) for r in ratios]
        return {"ratios": ratios}

    @property
    def targets_as_params(self) -> List[str]:
        return ["image", "bboxes"]

    def update_results_cache(self, inputs):
        self.results_cache.append(copy.deepcopy(inputs))
        if len(self.results_cache) > self.max_cached_images:
            index = random.randint(0, len(self.results_cache) - 1)
            self.results_cache.pop(index)

    def get_params_dependent_on_targets(self, params):
        self.update_results_cache(params)
        # judge whether to apply mosaic transform
        apply = (random.random() < self.p) or self._always_apply
        apply = apply and len(self.results_cache) > 2
        if not apply:
            return {
                "extra_images": None,
                "extra_bboxes": None,
            }

        # get images and bboxes, and extra images and bboxes
        index = random.randint(0, len(self.results_cache) - 1)
        extra_params = self.results_cache[index]
        extra_images, extra_bboxes = extra_params["image"], extra_params["bboxes"]

        return {
            "extra_images": extra_images,
            "extra_bboxes": extra_bboxes,
        }

    def apply(self, img, extra_images=None, ratios=None, **params):
        if extra_images is None:
            return img
        
        image_sizes = [im.shape[:2] for im in [img, extra_images]]
        image_h, image_w = list(zip(*image_sizes))
        image_h, image_w = max(image_h), max(image_w)
        image_final = np.zeros((image_h, image_w, img.shape[-1]))
        for im, r in zip([img, extra_images], ratios):
            image_final[:im.shape[0], :im.shape[1], :] += im * r
        image_final /= 2
        return image_final.astype(img.dtype)

    def apply_to_bboxes(self, bboxes, extra_bboxes=None, **params):
        if extra_bboxes is None:
            return bboxes

        return list(set(bboxes + extra_bboxes))
