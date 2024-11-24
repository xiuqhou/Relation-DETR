import copy
import math

import torch
from torch import nn

from models.bricks.base_transformer import TwostageTransformer
from models.bricks.basic import MLP
from models.bricks.dino_transformer import (
    DINOTransformerEncoder,
)
from models.bricks.position_encoding import get_sine_pos_embed
from models.bricks.relation_transformer import (
    PositionRelationEmbedding,
    RelationTransformerDecoderLayer,
    RelationTransformerEncoderLayer,
)
from util.misc import inverse_sigmoid


class DeformableTransformer(TwostageTransformer):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        num_classes: int,
        num_feature_levels: int = 4,
        two_stage_num_proposals: int = 300,
    ):
        super().__init__(num_feature_levels, encoder.embed_dim)
        # model parameters
        self.two_stage_num_proposals = two_stage_num_proposals
        self.num_classes = num_classes

        # model structure
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_class_head = nn.Linear(self.embed_dim, num_classes)
        self.encoder_bbox_head = MLP(self.embed_dim, self.embed_dim, 4, 3)
        self.pos_trans = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.pos_trans_norm = nn.LayerNorm(self.embed_dim)

        self.init_weights()

    def init_weights(self):
        # initilize encoder and hybrid classification layers
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.encoder_class_head.bias, bias_value)
        # initiailize encoder and hybrid regression layers
        nn.init.constant_(self.encoder_bbox_head.layers[-1].weight, 0.0)
        nn.init.constant_(self.encoder_bbox_head.layers[-1].bias, 0.0)

        # initialize pos_trans
        nn.init.xavier_uniform_(self.pos_trans.weight)

    def forward(
        self,
        multi_level_feats,
        multi_level_masks,
        multi_level_pos_embeds,
    ):
        # get input for encoder
        feat_flatten = self.flatten_multi_level(multi_level_feats)
        mask_flatten = self.flatten_multi_level(multi_level_masks)
        lvl_pos_embed_flatten = self.get_lvl_pos_embed(multi_level_pos_embeds)
        spatial_shapes, level_start_index, valid_ratios = self.multi_level_misc(multi_level_masks)
        reference_points, proposals = self.get_reference(spatial_shapes, valid_ratios)

        # encoder
        memory = self.encoder(
            query=feat_flatten,
            query_pos=lvl_pos_embed_flatten,
            spatial_shapes=spatial_shapes,
            query_key_padding_mask=mask_flatten,
            level_start_index=level_start_index,
            reference_points=reference_points,
        )

        # get encoder output, classes and coordinates
        output_memory, output_proposals = self.get_encoder_output(memory, proposals, mask_flatten)
        enc_outputs_class = self.encoder_class_head(output_memory)
        enc_outputs_coord = self.encoder_bbox_head(output_memory) + output_proposals
        enc_outputs_coord = enc_outputs_coord.sigmoid()

        # select topk
        topk = self.two_stage_num_proposals
        topk_index = torch.topk(enc_outputs_class[:, :, 0], topk, dim=1)[1].unsqueeze(-1)
        topk_enc_outputs_coord = enc_outputs_coord.gather(1, topk_index.expand(-1, -1, 4))

        # get query(target) and reference points
        # NOTE: original implementation calculates query and query_pos together.
        # To keep the interface the same with Dab, DN and DINO, we split the
        # calculation of query_pos into the DeformableDecoder
        reference_points = topk_enc_outputs_coord.detach()
        # nn.Linear can not perceive the arrangement order of elements
        # so exchange_xy=True/False does not matter results
        query_sine_embed = get_sine_pos_embed(
            reference_points, self.embed_dim // 2, exchange_xy=False
        )
        target = self.pos_trans_norm(self.pos_trans(query_sine_embed))

        # decoder
        outputs_classes, outputs_coords = self.decoder(
            query=target,
            value=memory,
            key_padding_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
        )

        return outputs_classes, outputs_coords, enc_outputs_class, enc_outputs_coord


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, num_classes):
        super().__init__()
        # parameters
        self.embed_dim = decoder_layer.embed_dim
        self.num_heads = decoder_layer.num_heads
        self.num_layers = num_layers
        self.num_classes = num_classes

        # decoder layers and embedding
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        # NOTE: the ref_point_head of Deformable is split from pos_trans and pos_norm,
        # which is different from DINO
        self.ref_point_head = nn.Sequential(
            nn.Linear(2 * self.embed_dim, self.embed_dim), nn.LayerNorm(self.embed_dim)
        )

        # iterative bounding box refinement
        class_head = nn.Linear(self.embed_dim, num_classes)
        bbox_head = MLP(self.embed_dim, self.embed_dim, 4, 3)
        self.class_head = nn.ModuleList([copy.deepcopy(class_head) for _ in range(num_layers)])
        self.bbox_head = nn.ModuleList([copy.deepcopy(bbox_head) for _ in range(num_layers)])

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

        # initialize ref_point_head
        nn.init.xavier_uniform_(self.ref_point_head[0].weight)

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
        # NOTE: the difference between DeformableDecoder and DabDecoder is that
        # Deformable does not introduce reference refinement for query pos
        query_sine_embed = get_sine_pos_embed(
            reference_points, self.embed_dim // 2, exchange_xy=False
        )
        query_pos = self.ref_point_head(query_sine_embed)

        outputs_classes, outputs_coords = [], []
        valid_ratio_scale = torch.cat([valid_ratios, valid_ratios], -1)[:, None]

        for layer_idx, layer in enumerate(self.layers):
            reference_points_input = reference_points.detach()[:, :, None] * valid_ratio_scale

            query = layer(
                query=query,
                query_pos=query_pos,
                reference_points=reference_points_input,
                value=value,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=key_padding_mask,
                self_attn_mask=attn_mask,
            )

            # get output
            output_class = self.class_head[layer_idx](query)
            output_coord = self.bbox_head[layer_idx](query) + inverse_sigmoid(reference_points)
            output_coord = output_coord.sigmoid()
            outputs_classes.append(output_class)
            outputs_coords.append(output_coord)

            if layer_idx == self.num_layers - 1:
                break

            # NOTE: Here we integrate position_relation_embedding into DN-Deformable-DETR
            src_boxes = tgt_boxes if layer_idx >= 1 else reference_points
            tgt_boxes = output_coord
            pos_relation = self.position_relation_embedding(src_boxes, tgt_boxes).flatten(0, 1)
            if attn_mask is not None:
                pos_relation.masked_fill_(attn_mask, float("-inf"))

            # iterative bounding box refinement
            reference_points = output_coord.detach()

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        return outputs_classes, outputs_coords


DeformableTransformerEncoder = DINOTransformerEncoder
DeformableTransformerEncoderLayer = RelationTransformerEncoderLayer
DeformableTransformerDecoderLayer = RelationTransformerDecoderLayer
