import torch
import torchvision
from torch import nn


class DETRBaseTransformer(nn.Module):
    """A base class that contains some methods commonly used in DETR transformer,
    such as DeformableTransformer, DabTransformer, DINOTransformer, AlignTransformer.

    """
    def __init__(self, num_feature_levels, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_feature_levels = num_feature_levels

    @staticmethod
    def flatten_multi_level(multi_level_elements):
        multi_level_elements = torch.cat(
            tensors=[e.flatten(-2) for e in multi_level_elements], dim=-1
        )  # (b, [c], s)
        if multi_level_elements.ndim == 3:
            multi_level_elements.transpose_(1, 2)
        return multi_level_elements

    def multi_level_misc(self, multi_level_masks):
        if torchvision._is_tracing():
            # torch.Tensor.shape exports not well for ONNX
            # use operators.shape_as_tensor istead
            from torch.onnx import operators
            spatial_shapes = [operators.shape_as_tensor(m)[-2:] for m in multi_level_masks]
            spatial_shapes = torch.stack(spatial_shapes).to(multi_level_masks[0].device)
        else:
            spatial_shapes = [m.shape[-2:] for m in multi_level_masks]
            spatial_shapes = multi_level_masks[0].new_tensor(spatial_shapes, dtype=torch.int64)
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        valid_ratios = self.multi_level_valid_ratios(multi_level_masks)
        return spatial_shapes, level_start_index, valid_ratios

    @staticmethod
    def get_valid_ratios(mask):
        b, h, w = mask.shape
        if h == 0 or w == 0:  # for empty Tensor
            return mask.new_ones((b, 2)).float()
        valid_h = torch.sum(~mask[:, :, 0], 1)
        valid_w = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_h.float() / h
        valid_ratio_w = valid_w.float() / w
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)  # [n, 2]
        return valid_ratio

    def multi_level_valid_ratios(self, multi_level_masks):
        return torch.stack([self.get_valid_ratios(m) for m in multi_level_masks], 1)

    @staticmethod
    def get_full_reference_points(spatial_shapes, valid_ratios):
        reference_points_list = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.arange(0.5, h + 0.5, device=spatial_shapes.device),
                torch.arange(0.5, w + 0.5, device=spatial_shapes.device),
                indexing="ij",
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * h)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * w)
            ref = torch.stack((ref_x, ref_y), -1)  # [n, h*w, 2]
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)  # [n, s, 2]
        return reference_points

    def get_reference(self, spatial_shapes, valid_ratios):
        # get full_reference_points, should be transferred using valid_ratios
        full_reference_points = self.get_full_reference_points(spatial_shapes, valid_ratios)
        reference_points = full_reference_points[:, :, None] * valid_ratios[:, None]
        # get proposals, reuse full_reference_points to speed up
        level_wh = full_reference_points.new_tensor([[i] for i in range(spatial_shapes.shape[0])])
        level_wh = 0.05 * 2.0**level_wh.repeat_interleave(spatial_shapes.prod(-1), 0)
        level_wh = level_wh.expand_as(full_reference_points)
        proposals = torch.cat([full_reference_points, level_wh], -1)
        return reference_points, proposals


class MultiLevelTransformer(DETRBaseTransformer):
    """A base class that contains methods based on level_embeds."""
    def __init__(self, num_feature_levels, embed_dim):
        super().__init__(num_feature_levels, embed_dim)
        self.level_embeds = nn.Parameter(torch.Tensor(num_feature_levels, embed_dim))
        self._init_weights_detr_transformer()

    def _init_weights_detr_transformer(self):
        nn.init.normal_(self.level_embeds)

    def get_lvl_pos_embed(self, multi_level_pos_embeds):
        multi_level_pos_embeds = [
            p + l.view(1, -1, 1, 1) for p, l in zip(multi_level_pos_embeds, self.level_embeds)
        ]
        return self.flatten_multi_level(multi_level_pos_embeds)


class TwostageTransformer(MultiLevelTransformer):
    """A base class that contains some methods commonly used in two-stage transformer,
    such as DeformableTransformer, DabTransformer, DINOTransformer, AlignTransformer.

    """
    def __init__(self, num_feature_levels, embed_dim):
        super().__init__(num_feature_levels, embed_dim)
        self.enc_output = nn.Linear(embed_dim, embed_dim)
        self.enc_output_norm = nn.LayerNorm(embed_dim)
        self._init_weights_two_stage_transformer()

    def _init_weights_two_stage_transformer(self):
        nn.init.xavier_uniform_(self.enc_output.weight)
        nn.init.constant_(self.enc_output.bias, 0.0)

    def get_encoder_output(self, memory, proposals, memory_padding_mask):
        output_proposals_valid = ((proposals > 0.01) & (proposals < 0.99)).all(-1, keepdim=True)
        proposals = torch.log(proposals / (1 - proposals))  # inverse_sigmoid
        invalid = memory_padding_mask.unsqueeze(-1) | ~output_proposals_valid
        proposals.masked_fill_(invalid, float("inf"))

        output_memory = memory * (~memory_padding_mask.unsqueeze(-1)) * (output_proposals_valid)
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, proposals
