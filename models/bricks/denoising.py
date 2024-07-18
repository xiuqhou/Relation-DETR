import torch
from torch import nn
from torchvision.ops import boxes as box_ops

from util.misc import inverse_sigmoid


class GenerateDNQueries(nn.Module):
    """Generate denoising queries for DN-DETR

    Args:
        num_queries (int): Number of total queries in DN-DETR. Default: 300
        num_classes (int): Number of total categories. Default: 80.
        label_embed_dim (int): The embedding dimension for label encoding. Default: 256.
        denoising_groups (int): Number of noised ground truth groups. Default: 5.
        label_noise_prob (float): The probability of the label being noised. Default: 0.2.
        box_noise_scale (float): Scaling factor for box noising. Default: 0.4
        with_indicator (bool): If True, add indicator in noised label/box queries.

    """
    def __init__(
        self,
        num_queries: int = 300,
        num_classes: int = 80,
        label_embed_dim: int = 256,
        denoising_groups: int = 5,
        label_noise_prob: float = 0.2,
        box_noise_scale: float = 0.4,
        with_indicator: bool = False,
    ):
        super(GenerateDNQueries, self).__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.label_embed_dim = label_embed_dim
        self.denoising_groups = denoising_groups
        self.label_noise_prob = label_noise_prob
        self.box_noise_scale = box_noise_scale
        self.with_indicator = with_indicator

        # leave one dim for indicator mentioned in DN-DETR
        if with_indicator:
            self.label_encoder = nn.Embedding(num_classes, label_embed_dim - 1)
        else:
            self.label_encoder = nn.Embedding(num_classes, label_embed_dim)

    @staticmethod
    def apply_label_noise(labels: torch.Tensor, label_noise_prob: float = 0.2, num_classes: int = 80):
        if label_noise_prob > 0:
            mask = torch.rand_like(labels.float()) < label_noise_prob
            noised_labels = torch.randint_like(labels, 0, num_classes)
            noised_labels = torch.where(mask, noised_labels, labels)
            return noised_labels
        else:
            return labels

    @staticmethod
    def apply_box_noise(boxes: torch.Tensor, box_noise_scale: float = 0.4):
        if box_noise_scale > 0:
            diff = torch.zeros_like(boxes)
            diff[:, :2] = boxes[:, 2:] / 2
            diff[:, 2:] = boxes[:, 2:]
            boxes += torch.mul((torch.rand_like(boxes) * 2 - 1.0), diff) * box_noise_scale
            boxes = boxes.clamp(min=0.0, max=1.0)
        return boxes

    def generate_query_masks(self, max_gt_num_per_image, device):
        noised_query_nums = max_gt_num_per_image * self.denoising_groups
        tgt_size = noised_query_nums + self.num_queries
        attn_mask = torch.zeros(tgt_size, tgt_size, device=device, dtype=torch.bool)
        # match query cannot see the reconstruct
        attn_mask[noised_query_nums:, :noised_query_nums] = True
        for i in range(self.denoising_groups):
            start_col = start_row = max_gt_num_per_image * i
            end_col = end_row = max_gt_num_per_image * (i + 1)
            assert noised_query_nums >= end_col and start_col >= 0, "check attn_mask"
            attn_mask[start_row:end_row, :start_col] = True
            attn_mask[start_row:end_row, end_col:noised_query_nums] = True
        return attn_mask

    def forward(self, gt_labels_list, gt_boxes_list):
        """

        :param gt_labels_list: Ground truth bounding boxes per image
            with normalized coordinates in format ``(x, y, w, h)`` in shape ``(num_gts, 4)`
        :param gt_boxes_list: Classification labels per image in shape ``(num_gt, )``
        :return: Noised label queries, box queries, attention mask and denoising metas.
        """

        # concat ground truth labels and boxes in one batch
        # e.g. [tensor([0, 1, 2]), tensor([2, 3, 4])] -> tensor([0, 1, 2, 2, 3, 4])
        gt_labels = torch.cat(gt_labels_list)
        gt_boxes = torch.cat(gt_boxes_list)

        # For efficient denoising, repeat the original ground truth labels and boxes to
        # create more training denoising samples.
        # e.g. tensor([0, 1, 2, 2, 3, 4]) -> tensor([0, 1, 2, 2, 3, 4, 0, 1, 2, 2, 3, 4]) if group = 2.
        gt_labels = gt_labels.repeat(self.denoising_groups, 1).flatten()
        gt_boxes = gt_boxes.repeat(self.denoising_groups, 1)

        # set the device as "gt_labels"
        device = gt_labels.device
        assert len(gt_labels_list) == len(gt_boxes_list)

        batch_size = len(gt_labels_list)

        # the number of ground truth per image in one batch
        # e.g. [tensor([0, 1]), tensor([2, 3, 4])] -> gt_nums_per_image: [2, 3]
        # means there are 2 instances in the first image and 3 instances in the second image
        gt_nums_per_image = [x.numel() for x in gt_labels_list]

        # Add noise on labels and boxes
        noised_labels = self.apply_label_noise(gt_labels, self.label_noise_prob, self.num_classes)
        noised_boxes = self.apply_box_noise(gt_boxes, self.box_noise_scale)
        noised_boxes = inverse_sigmoid(noised_boxes)

        # encoding labels
        label_embedding = self.label_encoder(noised_labels)
        query_num = label_embedding.shape[0]

        # add indicator to label encoding if with_indicator == True
        if self.with_indicator:
            label_embedding = torch.cat([label_embedding, torch.ones([query_num, 1], device=device)], 1)

        # calculate the max number of ground truth in one image inside the batch.
        # e.g. gt_nums_per_image = [2, 3] which means
        # the first image has 2 instances and the second image has 3 instances
        # then the max_gt_num_per_image should be 3.
        max_gt_num_per_image = max(gt_nums_per_image)

        # the total denoising queries is depended on denoising groups and max number of instances.
        noised_query_nums = max_gt_num_per_image * self.denoising_groups

        # initialize the generated noised queries to zero.
        # And the zero initialized queries will be assigned with noised embeddings later.
        noised_label_queries = torch.zeros(batch_size, noised_query_nums, self.label_embed_dim, device=device)
        noised_box_queries = torch.zeros(batch_size, noised_query_nums, 4, device=device)

        # batch index per image: [0, 1, 2, 3] for batch_size == 4
        batch_idx = torch.arange(0, batch_size)

        # e.g. gt_nums_per_image = [2, 3]
        # batch_idx = [0, 1]
        # then the "batch_idx_per_instance" equals to [0, 0, 1, 1, 1]
        # which indicates which image the instance belongs to.
        # cuz the instances has been flattened before.
        batch_idx_per_instance = torch.repeat_interleave(batch_idx, torch.tensor(gt_nums_per_image).long())

        # indicate which image the noised labels belong to. For example:
        # noised label: tensor([0, 1, 2, 2, 3, 4, 0, 1, 2, 2, 3, 4])
        # batch_idx_per_group: tensor([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1])
        # which means the first label "tensor([0])"" belongs to "image_0".
        batch_idx_per_group = batch_idx_per_instance.repeat(self.denoising_groups, 1).flatten()

        # Cuz there might be different numbers of ground truth in each image of the same batch.
        # So there might be some padding part in noising queries.
        # Here we calculate the indexes for the valid queries and
        # fill them with the noised embeddings.
        # And leave the padding part to zeros.
        if len(gt_nums_per_image):
            valid_index_per_group = torch.cat([torch.arange(num) for num in gt_nums_per_image])
            valid_index_per_group = torch.cat([
                valid_index_per_group + max_gt_num_per_image * i for i in range(self.denoising_groups)
            ]).long()
        if len(batch_idx_per_group):
            noised_label_queries[(batch_idx_per_group, valid_index_per_group)] = label_embedding
            noised_box_queries[(batch_idx_per_group, valid_index_per_group)] = noised_boxes

        # generate attention masks for transformer layers
        attn_mask = self.generate_query_masks(max_gt_num_per_image, device)

        return (
            noised_label_queries,
            noised_box_queries,
            attn_mask,
            self.denoising_groups,
            max_gt_num_per_image,
        )


class GenerateCDNQueries(GenerateDNQueries):
    def __init__(
        self,
        num_queries: int = 300,
        num_classes: int = 80,
        label_embed_dim: int = 256,
        denoising_nums: int = 100,
        label_noise_prob: float = 0.5,
        box_noise_scale: float = 1.0,
    ):
        super().__init__(
            num_queries=num_queries,
            num_classes=num_classes,
            label_embed_dim=label_embed_dim,
            label_noise_prob=label_noise_prob,
            box_noise_scale=box_noise_scale,
            denoising_groups=1,
        )

        self.denoising_nums = denoising_nums
        self.label_encoder = nn.Embedding(num_classes, label_embed_dim)

    def apply_box_noise(self, boxes: torch.Tensor, box_noise_scale: float = 0.4):
        """

        :param boxes: Bounding boxes in format ``(x_c, y_c, w, h)`` with shape ``(num_boxes, 4)``
        :param box_noise_scale: Scaling factor for box noising, defaults to 0.4
        :return: Noised boxes
        """
        num_boxes = len(boxes) // self.denoising_groups // 2
        positive_idx = torch.arange(num_boxes, dtype=torch.long, device=boxes.device)
        positive_idx = positive_idx.unsqueeze(0).repeat(self.denoising_groups, 1)
        positive_idx += (
            torch.arange(self.denoising_groups, dtype=torch.long, device=boxes.device).unsqueeze(1) *
            num_boxes * 2
        )
        positive_idx = positive_idx.flatten()
        negative_idx = positive_idx + num_boxes
        if box_noise_scale > 0:
            diff = torch.zeros_like(boxes)
            diff[:, :2] = boxes[:, 2:] / 2
            diff[:, 2:] = boxes[:, 2:] / 2
            rand_sign = torch.randint_like(boxes, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0
            rand_part = torch.rand_like(boxes)
            rand_part[negative_idx] += 1.0
            rand_part *= rand_sign
            xyxy_boxes = box_ops._box_cxcywh_to_xyxy(boxes)
            xyxy_boxes += torch.mul(rand_part, diff) * box_noise_scale
            xyxy_boxes = xyxy_boxes.clamp(min=0.0, max=1.0)
            boxes = box_ops._box_xyxy_to_cxcywh(xyxy_boxes)

        return boxes

    def forward(self, gt_labels_list, gt_boxes_list):
        """

        :param gt_labels_list: Ground truth bounding boxes per image
            with normalized coordinates in format ``(x, y, w, h)`` in shape ``(num_gts, 4)``
        :param gt_boxes_list: Classification labels per image in shape ``(num_gt, )``
        :return: Noised label queries, box queries, attention mask and denoising metas.
        """
        # the number of ground truth per image in one batch
        # e.g. [tensor([0, 1]), tensor([2, 3, 4])] -> gt_nums_per_image: [2, 3]
        # means there are 2 instances in the first image and 3 instances in the second image
        gt_nums_per_image = [x.numel() for x in gt_labels_list]

        # calculate the max number of ground truth in one image inside the batch.
        # e.g. gt_nums_per_image = [2, 3] which means
        # the first image has 2 instances and the second image has 3 instances
        # then the max_gt_num_per_image should be 3.
        max_gt_num_per_image = max(gt_nums_per_image)

        # get denoising_groups, which is 1 for empty ground truth
        denoising_groups = self.denoising_nums * max_gt_num_per_image // max(max_gt_num_per_image**2, 1)
        self.denoising_groups = max(denoising_groups, 1)

        # concat ground truth labels and boxes in one batch
        # e.g. [tensor([0, 1, 2]), tensor([2, 3, 4])] -> tensor([0, 1, 2, 2, 3, 4])
        gt_labels = torch.cat(gt_labels_list)
        gt_boxes = torch.cat(gt_boxes_list)

        # For efficient denoising, repeat the original ground truth labels and boxes to
        # create more training denoising samples.
        # each group has positive and negative. e.g. if group = 2, tensor([0, 1, 2, 2, 3, 4]) ->
        # tensor([0, 1, 2, 2, 3, 4, 0, 1, 2, 2, 3, 4, 0, 1, 2, 2, 3, 4, 0, 1, 2, 2, 3, 4]).
        gt_labels = gt_labels.repeat(self.denoising_groups * 2, 1).flatten()
        gt_boxes = gt_boxes.repeat(self.denoising_groups * 2, 1)

        # set the device as "gt_labels"
        device = gt_labels.device
        assert len(gt_labels_list) == len(gt_boxes_list)

        batch_size = len(gt_labels_list)

        # Add noise on labels and boxes
        noised_labels = self.apply_label_noise(gt_labels, self.label_noise_prob * 0.5, self.num_classes)
        noised_boxes = self.apply_box_noise(gt_boxes, self.box_noise_scale)
        noised_boxes = inverse_sigmoid(noised_boxes)

        # encoding labels
        label_embedding = self.label_encoder(noised_labels)

        # the total denoising queries is depended on denoising groups and max number of instances.
        noised_query_nums = max_gt_num_per_image * self.denoising_groups * 2

        # initialize the generated noised queries to zero.
        # And the zero initialized queries will be assigned with noised embeddings later.
        noised_label_queries = torch.zeros(batch_size, noised_query_nums, self.label_embed_dim, device=device)
        noised_box_queries = torch.zeros(batch_size, noised_query_nums, 4, device=device)

        # batch index per image: [0, 1, 2, 3] for batch_size == 4
        batch_idx = torch.arange(0, batch_size)

        # e.g. gt_nums_per_image = [2, 3]
        # batch_idx = [0, 1]
        # then the "batch_idx_per_instance" equals to [0, 0, 1, 1, 1]
        # which indicates which image the instance belongs to.
        # cuz the instances has been flattened before.
        batch_idx_per_instance = torch.repeat_interleave(
            batch_idx, torch.tensor(gt_nums_per_image, dtype=torch.long)
        )

        # indicate which image the noised labels belong to. For example:
        # noised label: tensor([0, 1, 2, 2, 3, 4, 0, 1, 2, 2, 3, 4, 0, 1, 2, 2, 3, 4， 0, 1, 2, 2, 3, 4])
        # batch_idx_per_group: tensor([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1])
        # which means the first label "tensor([0])"" belongs to "image_0".
        batch_idx_per_group = batch_idx_per_instance.repeat(self.denoising_groups * 2, 1).flatten()

        # Cuz there might be different numbers of ground truth in each image of the same batch.
        # So there might be some padding part in noising queries.
        # Here we calculate the indexes for the valid queries and
        # fill them with the noised embeddings.
        # And leave the padding part to zeros.
        if len(gt_nums_per_image):
            valid_index_per_group = torch.cat([torch.arange(num) for num in gt_nums_per_image])
            valid_index_per_group = torch.cat([
                valid_index_per_group + max_gt_num_per_image * i for i in range(self.denoising_groups * 2)
            ]).long()
        if len(batch_idx_per_group):
            noised_label_queries[(batch_idx_per_group, valid_index_per_group)] = label_embedding
            noised_box_queries[(batch_idx_per_group, valid_index_per_group)] = noised_boxes

        # generate attention masks for transformer layers
        attn_mask = self.generate_query_masks(2 * max_gt_num_per_image, device)

        return (
            noised_label_queries,
            noised_box_queries,
            attn_mask,
            self.denoising_groups,
            max_gt_num_per_image * 2,
        )
