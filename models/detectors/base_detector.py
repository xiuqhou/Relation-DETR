import copy
import logging
import os
from typing import Dict, List, Optional, Union

import torch.distributed
import torch.jit
import torchvision
from torch import Tensor, nn
from torch.nn import functional as NF
from torchvision.models.detection.image_list import ImageList
from torchvision.ops import boxes as box_ops

from transforms import functional as F
from transforms import v2 as T
from transforms.functional import InterpolationMode
from util.misc import decode_labels, encode_labels, image_list_from_tensors
from util.utils import get_world_size, is_dist_avail_and_initialized


class EvalResize(nn.Module):
    """Resize transform friendly to ONNX and torchscript"""
    def __init__(
        self,
        min_size: int,
        max_size: Optional[int] = None,
        interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
        antialias: Optional[Union[str, bool]] = "warn",
    ):
        super().__init__()
        assert isinstance(min_size, int) and isinstance(max_size, int)
        self.min_size = min_size
        self.max_size = max_size
        self.interpolation = interpolation
        self.antialias = antialias

    def forward(self, image: Tensor):
        assert isinstance(image, Tensor), "Only one image Tensor is supported"
        if torchvision._is_tracing():
            from torch.onnx import operators
            orig_height, orig_width = operators.shape_as_tensor(image)[-2:]
        else:
            orig_height, orig_width = torch.tensor(image.shape[-2:])

        r = self.min_size / torch.min(orig_height, orig_width)
        if self.max_size is not None:
            r = torch.min(r, self.max_size / torch.max(orig_height, orig_width))

        new_width = (orig_width * r).to(orig_width.dtype)
        new_height = (orig_height * r).to(orig_width.dtype)

        return F.resize(
            image,
            size=(new_height, new_width),
            interpolation=self.interpolation,
            antialias=self.antialias
        )


class BaseDetector(nn.Module):
    def __init__(self, min_size=None, max_size=None, size_divisible=32):
        """Initialize BaseDetector. Before forward propagation, input images should be padded and batched,
        and optionally resized. For training mode, the resize and other augmentations are done in
        dataset transformation, whereas in evaluation mode, since model needs original shapes (before
        any augmentation) for calculating COCO metrics, I perform resize inside the forward function.
        Therefore, the input image MUST NOT have any augmentation for evaluation mode!

        :param min_size: the minimum threshold to resize input images, defaults to None
        :param max_size: the maximum threshold to resize input images, defaults to None
        """
        super().__init__()
        self.size_divisible = size_divisible
        size = [s for s in (min_size, max_size) if isinstance(s, (int, float))]
        if len(size) != 0:
            eval_transform = [EvalResize(min(size), max(size), antialias=True)]
        else:
            eval_transform = []
        eval_transform.append(T.ConvertImageDtype(torch.float))
        eval_transform.append(T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), False))
        self.eval_transform = nn.Sequential(*eval_transform)
        self._device = None

    @property
    def device(self):
        if self._device is None:
            self._device = next(iter(self.parameters())).device
        return self._device

    @property
    def CLASSES(self):
        """This returns the classes of the current model. By default, the class
        information is encoded in a tensor named self._classes_. If not registered,
        the function will use default [0, ..., num_classes - 1] as a replacement.

        :return: A list contains class information of the current model.
        """
        logger = logging.getLogger(os.path.basename(os.getcwd()) + "." + __name__)
        if not hasattr(self, "_classes_") or self._classes_ is None:
            logger.warn("register default classes for model")
            dummy_classes = tuple(str(s) for s in range(self.num_classes))
            self.register_buffer("_classes_", torch.tensor(encode_labels(dummy_classes)))
        return decode_labels(tuple(self._classes_.tolist()))

    @staticmethod
    def check_boxes(targets):
        for target_idx, target in enumerate(targets):
            boxes = target["boxes"]
            degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
            if degenerate_boxes.any():
                # print the first degenerate box
                bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                degen_bb: List[float] = boxes[bb_idx].tolist()
                torch._assert(
                    False,
                    "All bounding boxes should have positive height and width."
                    f" Found invalid box {degen_bb} for target at index {target_idx}.",
                )

    def preprocess_image(self, images: List[Tensor]) -> ImageList:
        """Preprocess normalization and make up a batch for input images.

        :param images: list of input images, each has shape of (c, h, w)
        :return: ImageList of the normalized and batched images
        """
        if isinstance(images, torch.Tensor):
            images = images.unbind(0)
        if not self.training and self.eval_transform:
            images = [self.eval_transform(image) for image in images]
        images = image_list_from_tensors(images, self.size_divisible)
        return images

    @torch.inference_mode()
    def preprocess(self, images: List[Tensor], targets: List[Dict] = None):
        if targets is not None:
            self.check_boxes(targets)
        return self.preprocess_image(images), targets

    @staticmethod
    def query_original_sizes(images):
        if torchvision._is_tracing():
            from torch.onnx import operators
            if isinstance(images, torch.Tensor):
                images = images.unbind(0)
            original_sizes = [operators.shape_as_tensor(m)[-2:] for m in images]
            original_sizes = torch.stack(original_sizes).to(images[0].device)
        else:
            original_sizes = [m.shape[-2:] for m in images]
            original_sizes = torch.as_tensor(original_sizes, device=images[0].device)
        return original_sizes


class DETRDetector(BaseDetector):
    def get_multi_levels(self, images: ImageList, mask: Tensor):
        # extract higher features matching proto_levels
        multi_level_feats = self.backbone(images.tensors)
        multi_level_feats = self.neck(multi_level_feats)

        # extract multi_level masks and pos_embeds
        interp_mask = lambda size: NF.interpolate(mask[None], size=size)[0].to(torch.bool)
        multi_level_masks = [interp_mask(feat.shape[-2:]) for feat in multi_level_feats]

        # extract multi_level_pos_embeds
        multi_level_pos_embeds = [self.position_embedding(m) for m in multi_level_masks]

        return multi_level_feats, multi_level_masks, multi_level_pos_embeds

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{
            "pred_logits": a,
            "pred_boxes": b
        } for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    @staticmethod
    def prepare_targets(images: ImageList, targets: List[Dict]):
        # avoid overriding the input
        targets = copy.deepcopy(targets)
        # change boxes format from xyxy to normalized cxcywh
        for image_size, target in zip(images.image_sizes, targets):
            h, w = image_size
            boxes = target["boxes"]
            boxes = box_ops._box_xyxy_to_cxcywh(boxes)
            boxes = boxes / boxes.new_tensor([w, h, w, h])
            target["boxes"] = boxes
        return targets

    @staticmethod
    def construct_mask(images: ImageList):
        # construct mask
        b, c, h, w = images.tensors.shape
        mask = images.tensors.new_ones((b, h, w))
        for img_id, image_size in enumerate(images.image_sizes):
            mask[img_id, :image_size[0], :image_size[1]] = 0
        return mask

    @torch.no_grad()
    def preprocess(self, images: List[Tensor], targets: List[Dict] = None):
        images = self.preprocess_image(images)
        mask = self.construct_mask(images)
        if targets:
            self.check_boxes(targets)
            targets = self.prepare_targets(images, targets)
        return images, targets, mask


class DNDETRDetector(DETRDetector):
    def compute_dn_loss(self, outputs, targets, **kwargs):
        losses = {}
        device = self.device

        # get num_boxes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        if "denoising_output" in outputs:
            denoising_output, denoising_groups, max_gt_num_per_image = (
                outputs["denoising_output"],
                outputs["denoising_groups"],
                outputs["max_gt_num_per_image"],
            )
            dn_idx = []
            for i in range(len(targets)):
                if len(targets[i]["labels"]) > 0:
                    group_index, target_index = torch.meshgrid(
                        torch.arange(denoising_groups, device=device),
                        torch.arange(len(targets[i]["labels"]), device=device),
                        indexing="ij",
                    )
                    output_idx = group_index * max_gt_num_per_image + target_index
                    output_idx = output_idx.flatten()
                    tgt_idx = target_index.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([], dtype=torch.long, device=device)
                dn_idx.append((output_idx, tgt_idx))

            # calculate matching loss
            denoising_losses = self.criterion.calculate_loss(
                denoising_output,
                targets,
                num_boxes=num_boxes * denoising_groups,
                indices=dn_idx,
                **kwargs,
            )
            losses.update({k + "_dn": v for k, v in denoising_losses.items()})

            # calculate auxiliary loss
            for i in range(len(denoising_output.get("aux_outputs", []))):
                denoising_output_aux = denoising_output["aux_outputs"][i]
                denoising_losses = self.criterion.calculate_loss(
                    denoising_output_aux,
                    targets,
                    num_boxes=num_boxes * denoising_groups,
                    indices=dn_idx,
                    **kwargs,
                )
                denoising_losses = {k + f"_dn_{i}": v for k, v in denoising_losses.items()}
                losses.update(denoising_losses)

        return losses

    def dn_post_process(self, outputs_class, outputs_coord, dn_metas):
        if dn_metas and "max_gt_num_per_image" in dn_metas:
            padding_size = dn_metas["max_gt_num_per_image"] * dn_metas["denoising_groups"]
            output_known_class = outputs_class[:, :, :padding_size, :]
            output_known_coord = outputs_coord[:, :, :padding_size, :]
            outputs_class = outputs_class[:, :, padding_size:, :]
            outputs_coord = outputs_coord[:, :, padding_size:, :]

            out = {
                "pred_logits": output_known_class[-1],
                "pred_boxes": output_known_coord[-1],
            }
            out["aux_outputs"] = self._set_aux_loss(output_known_class, output_known_coord)
            dn_metas["denoising_output"] = out
        return outputs_class, outputs_coord
