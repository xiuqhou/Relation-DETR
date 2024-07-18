from typing import List, Dict, Tuple

import torch
from torch import Tensor
from transforms import functional as F
from torchvision import ops


def _copy_paste(
    image: torch.Tensor,
    target: Dict[str, Tensor],
    paste_image: torch.Tensor,
    paste_target: Dict[str, Tensor],
    blending: bool = True,
    resize_interpolation: F.InterpolationMode = F.InterpolationMode.BILINEAR,
) -> Tuple[torch.Tensor, Dict[str, Tensor]]:
    # Random paste targets selection:
    num_masks = len(paste_target["masks"])

    if num_masks < 1:
        # Such degerante case with num_masks=0 can happen with LSJ
        # Let's just return (image, target)
        return image, target

    # We have to please torch script by explicitly specifying dtype as torch.long
    random_selection = torch.randint(
        0, num_masks, (num_masks,), device=paste_image.device
    )
    random_selection = torch.unique(random_selection).to(torch.long)

    paste_masks = paste_target["masks"][random_selection]
    paste_boxes = paste_target["boxes"][random_selection]
    paste_labels = paste_target["labels"][random_selection]

    masks = target["masks"]

    # We resize source and paste data if they have different sizes
    # This is something we introduced here as originally the algorithm works
    # on equal-sized data (for example, coming from LSJ data augmentations)
    size1 = image.shape[-2:]
    size2 = paste_image.shape[-2:]
    if size1 != size2:
        paste_image = F.resize(paste_image, size1, interpolation=resize_interpolation)
        paste_masks = F.resize(
            paste_masks, size1, interpolation=F.InterpolationMode.NEAREST
        )
        # resize bboxes:
        ratios = torch.tensor(
            (size1[1] / size2[1], size1[0] / size2[0]), device=paste_boxes.device
        )
        paste_boxes = paste_boxes.view(-1, 2, 2).mul(ratios).view(paste_boxes.shape)

    paste_alpha_mask = paste_masks.sum(dim=0) > 0

    if blending:
        paste_alpha_mask = F.gaussian_blur(
            paste_alpha_mask.unsqueeze(0),
            kernel_size=(5, 5),
            sigma=[
                2.0,
            ],
        )

    # Copy-paste images:
    image = (image * (~paste_alpha_mask)) + (paste_image * paste_alpha_mask)

    # Copy-paste masks:
    masks = masks * (~paste_alpha_mask)
    non_all_zero_masks = masks.sum((-1, -2)) > 0
    masks = masks[non_all_zero_masks]

    # Do a shallow copy of the target dict
    out_target = {k: v for k, v in target.items()}

    out_target["masks"] = torch.cat([masks, paste_masks])

    # Copy-paste boxes and labels
    boxes = ops.masks_to_boxes(masks)
    out_target["boxes"] = torch.cat([boxes, paste_boxes])

    labels = target["labels"][non_all_zero_masks]
    out_target["labels"] = torch.cat([labels, paste_labels])

    # Update additional optional keys: area and iscrowd if exist
    if "area" in target:
        out_target["area"] = out_target["masks"].sum((-1, -2)).to(torch.float32)

    if "iscrowd" in target and "iscrowd" in paste_target:
        # target['iscrowd'] size can be differ from mask size (non_all_zero_masks)
        # For example, if previous transforms geometrically modifies masks/boxes/labels but
        # does not update "iscrowd"
        if len(target["iscrowd"]) == len(non_all_zero_masks):
            iscrowd = target["iscrowd"][non_all_zero_masks]
            paste_iscrowd = paste_target["iscrowd"][random_selection]
            out_target["iscrowd"] = torch.cat([iscrowd, paste_iscrowd])

    # Check for degenerated boxes and remove them
    boxes = out_target["boxes"]
    degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
    if degenerate_boxes.any():
        valid_targets = ~degenerate_boxes.any(dim=1)

        out_target["boxes"] = boxes[valid_targets]
        out_target["masks"] = out_target["masks"][valid_targets]
        out_target["labels"] = out_target["labels"][valid_targets]

        if "area" in out_target:
            out_target["area"] = out_target["area"][valid_targets]
        if "iscrowd" in out_target and len(out_target["iscrowd"]) == len(valid_targets):
            out_target["iscrowd"] = out_target["iscrowd"][valid_targets]

    return image, out_target


class SimpleCopyPaste(torch.nn.Module):
    def __init__(
        self, blending=True, resize_interpolation=F.InterpolationMode.BILINEAR
    ):
        super().__init__()
        self.resize_interpolation = resize_interpolation
        self.blending = blending

    def forward(
        self, images: List[torch.Tensor], targets: List[Dict[str, Tensor]]
    ) -> Tuple[List[torch.Tensor], List[Dict[str, Tensor]]]:
        torch._assert(
            isinstance(images, (list, tuple))
            and all([isinstance(v, torch.Tensor) for v in images]),
            "images should be a list of tensors",
        )
        torch._assert(
            isinstance(targets, (list, tuple)) and len(images) == len(targets),
            "targets should be a list of the same size as images",
        )
        for target in targets:
            # Can not check for instance type dict with inside torch.jit.script
            # torch._assert(isinstance(target, dict), "targets item should be a dict")
            for k in ["masks", "boxes", "labels"]:
                torch._assert(k in target, f"Key {k} should be present in targets")
                torch._assert(
                    isinstance(target[k], torch.Tensor),
                    f"Value for the key {k} should be a tensor",
                )

        # images = [t1, t2, ..., tN]
        # Let's define paste_images as shifted list of input images
        # paste_images = [t2, t3, ..., tN, t1]
        # FYI: in TF they mix data on the dataset level
        images_rolled = images[-1:] + images[:-1]
        targets_rolled = targets[-1:] + targets[:-1]

        output_images: List[torch.Tensor] = []
        output_targets: List[Dict[str, Tensor]] = []

        for image, target, paste_image, paste_target in zip(
            images, targets, images_rolled, targets_rolled
        ):
            output_image, output_data = _copy_paste(
                image,
                target,
                paste_image,
                paste_target,
                blending=self.blending,
                resize_interpolation=self.resize_interpolation,
            )
            output_images.append(output_image)
            output_targets.append(output_data)

        return output_images, output_targets

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(blending={self.blending}, resize_interpolation={self.resize_interpolation})"
        return s
