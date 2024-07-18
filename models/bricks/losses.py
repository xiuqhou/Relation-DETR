from torch.nn import functional as F


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    prob = inputs.sigmoid()
    target_score = targets.to(inputs.dtype)
    weight = (1 - alpha) * prob**gamma * (1 - targets) + targets * alpha * (1 - prob)**gamma
    # according to original implementation, sigmoid_focal_loss keep gradient on weight
    loss = F.binary_cross_entropy_with_logits(inputs, target_score, reduction="none")
    loss = loss * weight
    # we use sum/num to replace mean to avoid NaN
    return (loss.sum(1) / max(loss.shape[1], 1)).sum() / num_boxes


def vari_sigmoid_focal_loss(inputs, targets, gt_score, num_boxes, alpha: float = 0.25, gamma: float = 2):
    prob = inputs.sigmoid().detach()  # pytorch version of RT-DETR has detach while paddle version not
    target_score = targets * gt_score.unsqueeze(-1)
    weight = (1 - alpha) * prob.pow(gamma) * (1 - targets) + target_score
    loss = F.binary_cross_entropy_with_logits(inputs, target_score, weight=weight, reduction="none")
    # we use sum/num to replace mean to avoid NaN
    return (loss.sum(1) / max(loss.shape[1], 1)).sum() / num_boxes


def ia_bce_loss(inputs, targets, gt_score, num_boxes, k: float = 0.25, alpha: float = 0, gamma: float = 2):
    prob = inputs.sigmoid().detach()
    # calculate iou_aware_score and constrain the value following original implementation
    iou_aware_score = prob**k * gt_score.unsqueeze(-1)**(1 - k)
    iou_aware_score = iou_aware_score.clamp(min=0.01)
    target_score = targets * iou_aware_score
    weight = (1 - alpha) * prob.pow(gamma) * (1 - targets) + targets
    loss = F.binary_cross_entropy_with_logits(inputs, target_score, weight=weight, reduction="none")
    # we use sum/num to replace mean to avoid NaN
    return (loss.sum(1) / max(loss.shape[1], 1)).sum() / num_boxes
