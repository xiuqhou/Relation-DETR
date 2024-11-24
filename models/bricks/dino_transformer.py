import copy
import math

import torch
from torch import nn

from models.bricks.base_transformer import TwostageTransformer
from models.bricks.basic import MLP
from models.bricks.position_encoding import get_sine_pos_embed
from models.bricks.relation_transformer import (
    PositionRelationEmbedding,
    RelationTransformerDecoderLayer,
    RelationTransformerEncoderLayer,
)
from util.misc import inverse_sigmoid


class DINOTransformer(TwostageTransformer):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        num_classes: int,
        num_feature_levels: int = 4,
        two_stage_num_proposals: int = 900,
    ):
        super().__init__(num_feature_levels, encoder.embed_dim)
        # model parameters
        self.two_stage_num_proposals = two_stage_num_proposals
        self.num_classes = num_classes

        # model structure
        self.encoder = encoder
        self.decoder = decoder
        self.tgt_embed = nn.Embedding(two_stage_num_proposals, self.embed_dim)
        self.encoder_class_head = nn.Linear(self.embed_dim, num_classes)
        self.encoder_bbox_head = MLP(self.embed_dim, self.embed_dim, 4, 3)

        self.init_weights()

    def init_weights(self):
        # initialize embedding layers
        nn.init.normal_(self.tgt_embed.weight)
        # initilize encoder and hybrid classification layers
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.encoder_class_head.bias, bias_value)
        # initiailize encoder and hybrid regression layers
        nn.init.constant_(self.encoder_bbox_head.layers[-1].weight, 0.0)
        nn.init.constant_(self.encoder_bbox_head.layers[-1].bias, 0.0)

    def forward(
        self,
        multi_level_feats,
        multi_level_masks,
        multi_level_pos_embeds,
        noised_label_query=None,
        noised_box_query=None,
        attn_mask=None,
    ):
        # get input for encoder
        feat_flatten = self.flatten_multi_level(multi_level_feats)
        mask_flatten = self.flatten_multi_level(multi_level_masks)
        lvl_pos_embed_flatten = self.get_lvl_pos_embed(multi_level_pos_embeds)
        spatial_shapes, level_start_index, valid_ratios = self.multi_level_misc(multi_level_masks)
        reference_points, proposals = self.get_reference(spatial_shapes, valid_ratios)

        # transformer encoder
        memory = self.encoder(
            query=feat_flatten,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reference_points=reference_points,
        )

        # get encoder output, classes and coordinates
        output_memory, output_proposals = self.get_encoder_output(memory, proposals, mask_flatten)
        enc_outputs_class = self.encoder_class_head(output_memory)
        enc_outputs_coord = self.encoder_bbox_head(output_memory) + output_proposals
        enc_outputs_coord = enc_outputs_coord.sigmoid()

        # get topk output classes and coordinates
        topk, num_classes = self.two_stage_num_proposals, self.num_classes
        topk_index = torch.topk(enc_outputs_class.max(-1)[0], topk, dim=1)[1].unsqueeze(-1)
        enc_outputs_class = enc_outputs_class.gather(1, topk_index.expand(-1, -1, num_classes))
        enc_outputs_coord = enc_outputs_coord.gather(1, topk_index.expand(-1, -1, 4))

        # get target and reference points
        reference_points = enc_outputs_coord.detach()
        target = self.tgt_embed.weight.expand(multi_level_feats[0].shape[0], -1, -1)

        # combine with noised_label_query and noised_box_query for denoising training
        if noised_label_query is not None and noised_box_query is not None:
            target = torch.cat([noised_label_query, target], 1)
            reference_points = torch.cat([noised_box_query.sigmoid(), reference_points], 1)

        # decoder
        outputs_classes, outputs_coords = self.decoder(
            query=target,
            value=memory,
            key_padding_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            attn_mask=attn_mask,
        )

        return outputs_classes, outputs_coords, enc_outputs_class, enc_outputs_coord


DINOTransformerEncoderLayer = RelationTransformerEncoderLayer


class DINOTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: nn.Module, num_layers: int = 6):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.embed_dim = encoder_layer.embed_dim

        self.init_weights()

    def init_weights(self):
        # initialize encoder layers
        for layer in self.layers:
            if hasattr(layer, "init_weights"):
                layer.init_weights()

    def forward(
        self,
        query,
        spatial_shapes,
        level_start_index,
        reference_points,
        query_pos=None,
        query_key_padding_mask=None,
    ):
        for layer in self.layers:
            query = layer(
                query,
                query_pos,
                reference_points,
                spatial_shapes,
                level_start_index,
                query_key_padding_mask,
            )

        return query


class DINOTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, num_classes):
        super().__init__()
        # parameters
        self.embed_dim = decoder_layer.embed_dim
        self.num_heads = decoder_layer.num_heads
        self.num_layers = num_layers
        self.num_classes = num_classes

        # decoder layers and embedding
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.ref_point_head = MLP(2 * self.embed_dim, self.embed_dim, self.embed_dim, 2)

        # iterative bounding box refinement
        class_head = nn.Linear(self.embed_dim, num_classes)
        bbox_head = MLP(self.embed_dim, self.embed_dim, 4, 3)
        self.class_head = nn.ModuleList([copy.deepcopy(class_head) for _ in range(num_layers)])
        self.bbox_head = nn.ModuleList([copy.deepcopy(bbox_head) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(self.embed_dim)

        self.position_relation_embedding = PositionRelationEmbedding(16, self.num_heads)
        self.init_weights()

    def init_weights(self):
        # initialize decoder layers
        for layer in self.layers:
            if hasattr(layer, "init_weights"):
                layer.init_weights()
        # initialize decoder classification layers
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        for class_head in self.class_head:
            nn.init.constant_(class_head.bias, bias_value)
        # initiailize decoder regression layers
        for bbox_head in self.bbox_head:
            nn.init.constant_(bbox_head.layers[-1].weight, 0.0)
            nn.init.constant_(bbox_head.layers[-1].bias, 0.0)

    def forward(
        self,
        query,
        reference_points,
        value,
        spatial_shapes,
        level_start_index,
        valid_ratios,
        key_padding_mask=None,
        attn_mask=None,
    ):
        outputs_classes, outputs_coords = [], []
        valid_ratio_scale = torch.cat([valid_ratios, valid_ratios], -1)[:, None]

        pos_relation = attn_mask  # fallback pos_relation to attn_mask
        for layer_idx, layer in enumerate(self.layers):
            reference_points_input = reference_points.detach()[:, :, None] * valid_ratio_scale
            query_sine_embed = get_sine_pos_embed(
                reference_points_input[:, :, 0, :], self.embed_dim // 2
            )
            query_pos = self.ref_point_head(query_sine_embed)

            # relation embedding
            query = layer(
                query=query,
                query_pos=query_pos,
                reference_points=reference_points_input,
                value=value,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=key_padding_mask,
                self_attn_mask=pos_relation,
            )

            # get output, reference_points are not detached for look_forward_twice
            output_class = self.class_head[layer_idx](self.norm(query))
            output_coord = self.bbox_head[layer_idx](self.norm(query))
            output_coord = output_coord + inverse_sigmoid(reference_points)
            output_coord = output_coord.sigmoid()
            outputs_classes.append(output_class)
            outputs_coords.append(output_coord)

            if layer_idx == self.num_layers - 1:
                break

            # NOTE: Here we integrate position_relation_embedding into DINO
            src_boxes = tgt_boxes if layer_idx >= 1 else reference_points
            tgt_boxes = output_coord
            pos_relation = self.position_relation_embedding(src_boxes, tgt_boxes).flatten(0, 1)
            if attn_mask is not None:
                pos_relation.masked_fill_(attn_mask, float("-inf"))

            # iterative bounding box refinement
            reference_points = inverse_sigmoid(reference_points.detach())
            reference_points = self.bbox_head[layer_idx](query) + reference_points
            reference_points = reference_points.sigmoid()

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        return outputs_classes, outputs_coords


DINOTransformerDecoderLayer = RelationTransformerDecoderLayer
