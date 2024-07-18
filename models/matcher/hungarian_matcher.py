import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn
from torchvision.ops.boxes import _box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """This class implements the Hungarian matching algorithm for bipartite graphs. It matches predicted bounding 
    boxes to ground truth boxes based on the minimum cost assignment. The cost is computed as a weighted sum of 
    classification, bounding box, and generalized intersection over union (IoU) costs. The focal loss is used to 
    weigh the classification cost. The HungarianMatcher class can be used in single or mixed assignment modes.
    The mixed assignment modes is introduced in `Align-DETR <https://arxiv.org/abs/2304.07527>`_.

    :param cost_class: The weight of the classification cost, defaults to 1
    :param cost_bbox: The weight of the bounding box cost, defaults to 1
    :param cost_giou: The weight of the generalized IoU cost, defaults to 1
    :param focal_alpha: The alpha parameter of the focal loss, defaults to 0.25
    :param focal_gamma: The gamma parameter of the focal loss, defaults to 2.0
    :param mixed_match: If True, mixed assignment is used, defaults to False
    """
    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        mixed_match: bool = False,
    ):
        super().__init__()

        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.mixed_match = mixed_match

    def calculate_class_cost(self, pred_logits, gt_labels, **kwargs):
        out_prob = pred_logits.sigmoid()

        # Compute the classification cost.
        neg_cost_class = -(1 - self.focal_alpha) * out_prob**self.focal_gamma * (1 - out_prob + 1e-6).log()
        pos_cost_class = -self.focal_alpha * (1 - out_prob)**self.focal_gamma * (out_prob + 1e-6).log()
        cost_class = pos_cost_class[:, gt_labels] - neg_cost_class[:, gt_labels]

        return cost_class

    def calculate_bbox_cost(self, pred_boxes, gt_boxes, **kwargs):
        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(pred_boxes, gt_boxes, p=1)
        return cost_bbox

    def calculate_giou_cost(self, pred_boxes, gt_boxes, **kwargs):
        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(_box_cxcywh_to_xyxy(pred_boxes), _box_cxcywh_to_xyxy(gt_boxes))
        return cost_giou

    @torch.no_grad()
    def calculate_cost(self, pred_boxes: Tensor, pred_logits: Tensor, gt_boxes: Tensor, gt_labels: Tensor):
        # Calculate class, bbox and giou cost
        cost_class = self.calculate_class_cost(pred_logits, gt_labels)
        cost_bbox = self.calculate_bbox_cost(pred_boxes, gt_boxes)
        cost_giou = self.calculate_giou_cost(pred_boxes, gt_boxes)

        # Final cost matrix
        c = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        return c

    @torch.no_grad()
    def forward(
        self, pred_boxes: Tensor, pred_logits: Tensor, gt_boxes: Tensor, gt_labels: Tensor, gt_copy: int = 1
    ):
        c = self.calculate_cost(pred_boxes, pred_logits, gt_boxes, gt_labels)

        # single assignment
        if not self.mixed_match:
            indices = linear_sum_assignment(c.cpu())
            return torch.as_tensor(indices[0]), torch.as_tensor(indices[1])

        # mixed assignment, used in AlignDETR
        gt_size = c.size(-1)
        num_queries = len(c)
        gt_copy = min(int(num_queries * 0.5 / gt_size), gt_copy) if gt_size > 0 else gt_copy
        src_ind, tgt_ind = linear_sum_assignment(c.cpu().repeat(1, gt_copy))
        tgt_ind = tgt_ind % gt_size
        tgt_ind, ind = torch.as_tensor(tgt_ind, dtype=torch.int64).sort()
        src_ind = torch.as_tensor(src_ind, dtype=torch.int64)[ind].view(-1)
        return src_ind, tgt_ind
