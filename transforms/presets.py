import albumentations as A
import cv2
import torch

from transforms import v2 as T
from transforms.album_transform import RandomShortestSize, RandomSizeCrop
from transforms.albumentations_warpper import AlbumentationsWrapper
from transforms.crop import RandomSizeCrop
from transforms.mix_transform import CachedMixUp, CachedMosaic, MixUp, Mosaic


def labels_getter(x):
    return x[-1]


basic = T.Compose([
    T.PILToTensor(),
    T.ConvertImageDtype(torch.float),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# train transform
hflip = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.PILToTensor(),
    T.ConvertImageDtype(torch.float),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

lsj = T.Compose([
    T.ScaleJitter(target_size=(1024, 1024), antialias=True),
    T.RandomCrop(size=(1024, 1024), pad_if_needed=True, fill=(123.0, 117.0, 104.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.PILToTensor(),
    T.ConvertImageDtype(torch.float),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    T.SanitizeBoundingBox(labels_getter=labels_getter),
])

lsj_1536 = T.Compose([
    T.ScaleJitter(target_size=(1536, 1536), antialias=True),
    T.RandomCrop(size=(1536, 1536), pad_if_needed=True, fill=(123.0, 117.0, 104.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.PILToTensor(),
    T.ConvertImageDtype(torch.float),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    T.SanitizeBoundingBox(labels_getter=labels_getter),
])

scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

multiscale = T.Compose([
    T.RandomShortestSize(min_size=scales, max_size=1333, antialias=True),
    T.RandomHorizontalFlip(p=0.5),
    T.PILToTensor(),
    T.ConvertImageDtype(torch.float),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

detr = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomChoice([
        T.RandomShortestSize(min_size=scales, max_size=1333, antialias=True),
        T.Compose([
            T.RandomShortestSize([400, 500, 600], antialias=True),
            RandomSizeCrop(384, 600),
            T.RandomShortestSize(min_size=scales, max_size=1333, antialias=True),
        ]),
    ]),
    T.PILToTensor(),
    T.ConvertImageDtype(torch.float),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    T.SanitizeBoundingBox(labels_getter=labels_getter),
])

ssd = T.Compose([
    T.RandomPhotometricDistort(),
    T.RandomZoomOut(fill=[123.0, 117.0, 104.0]),
    T.RandomIoUCrop(),
    T.RandomHorizontalFlip(p=0.5),
    T.PILToTensor(),
    T.ConvertImageDtype(torch.float),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    T.SanitizeBoundingBox(labels_getter=labels_getter),
])

ssdlite = T.Compose([
    T.RandomIoUCrop(),
    T.RandomHorizontalFlip(p=0.5),
    T.PILToTensor(),
    T.ConvertImageDtype(torch.float),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    T.SanitizeBoundingBox(labels_getter=labels_getter),
])

strong_album = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomChoice([
        T.RandomShortestSize(min_size=scales, max_size=1333, antialias=True),
        T.Compose([
            T.RandomShortestSize([400, 500, 600], antialias=True),
            RandomSizeCrop(384, 600),
            T.RandomShortestSize(min_size=scales, max_size=1333, antialias=True),
        ]),
    ]),
    AlbumentationsWrapper(
        A.Compose(
            [
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.0,
                    rotate_limit=0,
                    interpolation=1,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.5,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=(0.1, 0.3),
                    contrast_limit=(0.1, 0.3),
                    p=0.2,
                ),
                A.OneOf(
                    [
                        A.RGBShift(
                            r_shift_limit=10,
                            g_shift_limit=10,
                            b_shift_limit=10,
                            p=1.0,
                        ),
                        A.HueSaturationValue(
                            hue_shift_limit=20,
                            sat_shift_limit=30,
                            val_shift_limit=20,
                            p=1.0,
                        ),
                    ],
                    p=1.0,
                ),
                A.ImageCompression(quality_lower=85, quality_upper=95, p=0.2),
                A.ChannelShuffle(p=0.1),
                A.OneOf(
                    [
                        A.Blur(blur_limit=3, p=1.0),
                        A.MedianBlur(blur_limit=3, p=1.0),
                    ],
                    p=0.1,
                ),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"], min_visibility=0.0),
        )
    ),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.PILToTensor(),
    T.ConvertImageDtype(torch.float),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    T.SanitizeBoundingBox(labels_getter=labels_getter),
])

# albumentations strong data augmentation

scales_1200 = [int(t * 1.5) for t in [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]]

strong_album_1200_2000 = T.Compose([
    T.RandomChoice([
        T.RandomShortestSize(min_size=scales_1200, max_size=2000, antialias=True),
        T.Compose([
            T.RandomShortestSize([600, 750, 900], antialias=True),
            RandomSizeCrop(576, 900),
            T.RandomShortestSize(min_size=scales_1200, max_size=2000, antialias=True),
        ]),
    ]),
    AlbumentationsWrapper(
        A.Compose(
            [
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.0,
                    rotate_limit=0,
                    interpolation=1,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.5,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=(0.1, 0.3),
                    contrast_limit=(0.1, 0.3),
                    p=0.2,
                ),
                A.OneOf(
                    [
                        A.RGBShift(
                            r_shift_limit=10,
                            g_shift_limit=10,
                            b_shift_limit=10,
                            p=1.0,
                        ),
                        A.HueSaturationValue(
                            hue_shift_limit=20,
                            sat_shift_limit=30,
                            val_shift_limit=20,
                            p=1.0,
                        ),
                    ],
                    p=1.0,
                ),
                A.ImageCompression(quality_lower=85, quality_upper=95, p=0.2),
                A.ChannelShuffle(p=0.1),
                A.OneOf(
                    [
                        A.Blur(blur_limit=3, p=1.0),
                        A.MedianBlur(blur_limit=3, p=1.0),
                    ],
                    p=0.1,
                ),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc", label_fields=["labels"], min_visibility=0.0
            ),
        )
    ),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.PILToTensor(),
    T.ConvertImageDtype(torch.float),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    T.SanitizeBoundingBox(labels_getter=labels_getter),
])

rtdetr_transform = T.Compose([
    T.RandomPhotometricDistort(p=0.8),
    T.RandomZoomOut(p=0.5, fill=0, side_range=(1.0, 4.0)),
    T.RandomIoUCrop(),
    T.RandomHorizontalFlip(p=0.5),
    T.Resize(size=[640, 640], antialias=True),
    T.ToImageTensor(),
    T.ConvertImageDtype(torch.float),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    T.SanitizeBoundingBox(labels_getter=labels_getter),
])

# some transform examples related to mosaic, mixup, cached_mosaic and cached_mixup
# you may want to add flip, crop, resize transforms into them for better performance
mosaic = T.Compose([
    T.RandomHorizontalFlip(),
    Mosaic(),
    T.PILToTensor(),
    T.ConvertImageDtype(torch.float),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    T.SanitizeBoundingBox(labels_getter=labels_getter),
])

cached_mosaic = T.Compose([
    T.RandomHorizontalFlip(),
    CachedMosaic(),
    T.PILToTensor(),
    T.ConvertImageDtype(torch.float),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    T.SanitizeBoundingBox(labels_getter=labels_getter),
])

mixup = T.Compose([
    T.RandomHorizontalFlip(),
    MixUp(),
    T.PILToTensor(),
    T.ConvertImageDtype(torch.float),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

cached_mixup = T.Compose([
    T.RandomHorizontalFlip(),
    CachedMixUp(),
    T.PILToTensor(),
    T.ConvertImageDtype(torch.float),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

mixup_mosaic = T.Compose([
    T.RandomHorizontalFlip(),
    MixUp(),
    Mosaic(),
    T.PILToTensor(),
    T.ConvertImageDtype(torch.float),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    T.SanitizeBoundingBox(labels_getter=labels_getter),
])

cached_mixup_mosaic = T.Compose([
    T.RandomHorizontalFlip(),
    CachedMixUp(),
    CachedMosaic(),
    T.PILToTensor(),
    T.ConvertImageDtype(torch.float),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    T.SanitizeBoundingBox(labels_getter=labels_getter),
])

mosaic_mixup = T.Compose([
    T.RandomHorizontalFlip(),
    Mosaic(),
    MixUp(),
    T.PILToTensor(),
    T.ConvertImageDtype(torch.float),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    T.SanitizeBoundingBox(labels_getter=labels_getter),
])

cached_mosaic_mixup = T.Compose([
    T.RandomHorizontalFlip(),
    CachedMosaic(),
    CachedMixUp(),
    T.PILToTensor(),
    T.ConvertImageDtype(torch.float),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
