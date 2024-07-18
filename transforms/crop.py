import random
from typing import Any, Dict, List

import torch
from torchvision.ops import box_iou

from transforms.v2 import Transform
from transforms.v2 import functional as F
from transforms.v2.utils import query_bounding_box, query_spatial_size
from util import datapoints


class RandomSizeCrop(Transform):
    def __init__(self, min_size: int, max_size: int):
        super().__init__()
        self.min_size = min_size
        self.max_size = max_size
    
    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        orig_h, orig_w = query_spatial_size(flat_inputs)
        crop_h = random.randint(self.min_size, min(orig_h, self.max_size))
        crop_w = random.randint(self.min_size, min(orig_w, self.max_size))
        
        # get crop region
        top = torch.randint(0, orig_h - crop_h + 1, size=(1,)).item()
        left = torch.randint(0, orig_w - crop_w + 1, size=(1,)).item()
        
        return {"left": left, "top": top, "height": crop_h, "width": crop_w}
    
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return F.crop(inpt, **params)


class BoxCenteredRandomSizeCrop(Transform):
    def __init__(self, min_size: int, max_size: int, sampler_options=None, trials: int = 40):
        super().__init__()
        self.min_size = min_size
        self.max_size = max_size
        self.trials = trials
        if sampler_options is None:
            sampler_options = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        self.options = sampler_options
    
    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        orig_h, orig_w = query_spatial_size(flat_inputs)
        bboxes = query_bounding_box(flat_inputs)
        best_iou = 0
        for _ in range(self.trials):
            idx = int(torch.randint(low=0, high=len(self.options), size=(1,)))
            min_jaccard_overlap = self.options[idx]
            crop_h = random.randint(self.min_size, min(orig_h, self.max_size))
            crop_w = random.randint(self.min_size, min(orig_w, self.max_size))
            
            # get crop region
            top = torch.randint(0, orig_h - crop_h + 1, size=(1,)).item()
            left = torch.randint(0, orig_w - crop_w + 1, size=(1,)).item()
            right = left + crop_w
            bottom = top + crop_h
            
            # check for any valid boxes with centers within the crop area
            xyxy_bboxes = F.convert_format_bounding_box(
                bboxes.as_subclass(torch.Tensor), bboxes.format, datapoints.BoundingBoxFormat.XYXY
            )
            cx = 0.5 * (xyxy_bboxes[..., 0] + xyxy_bboxes[..., 2])
            cy = 0.5 * (xyxy_bboxes[..., 1] + xyxy_bboxes[..., 3])
            is_within_crop_area = (left < cx) & (cx < right) & (top < cy) & (cy < bottom)
            if not is_within_crop_area.any():
                continue
            
            xyxy_bboxes = xyxy_bboxes[is_within_crop_area]
            ious = box_iou(
                xyxy_bboxes,
                xyxy_bboxes.new_tensor([[left, top, right, bottom]]),
            )
            cur_region = dict(
                top=top,
                left=left,
                height=crop_h,
                width=crop_w,
                is_within_crop_area=is_within_crop_area,
            )
            
            if ious.max() > best_iou:
                best_region = cur_region
            
            if ious.max() < min_jaccard_overlap:
                continue
            
            return cur_region
        
        return best_region
    
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if len(params) < 1:
            return inpt
        
        output = F.crop(
            inpt,
            top=params["top"],
            left=params["left"],
            height=params["height"],
            width=params["width"],
        )
        
        if isinstance(output, datapoints.BoundingBox):
            # We "mark" the invalid boxes as degenreate, and they can be
            # removed by a later call to SanitizeBoundingBox()
            output[~params["is_within_crop_area"]] = 0
        return output
