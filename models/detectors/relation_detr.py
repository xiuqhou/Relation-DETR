import copy
from typing import Dict, List

from torch import Tensor, nn

from models.bricks.denoising import GenerateCDNQueries
from models.detectors.base_detector import DNDETRDetector


class RelationDETR(DNDETRDetector):
    def __init__(
        # model structure
        self,
        backbone: nn.Module,
        neck: nn.Module,
        position_embedding: nn.Module,
        transformer: nn.Module,
        criterion: nn.Module,
        postprocessor: nn.Module,
        # model parameters
        num_classes: int,
        num_queries: int = 900,
        hybrid_assign: int = 6,
        denoising_nums: int = 100,
        # model variants
        min_size: int = None,
        max_size: int = None,
    ):
        super().__init__(min_size, max_size)
        # define model parameters
        self.num_classes = num_classes
        embed_dim = transformer.embed_dim
        self.hybrid_assign = hybrid_assign

        # define model structures
        self.backbone = backbone
        self.neck = neck
        self.position_embedding = position_embedding
        self.transformer = transformer
        self.criterion = criterion
        self.postprocessor = postprocessor
        self.denoising_generator = GenerateCDNQueries(
            num_queries=num_queries,
            num_classes=num_classes,
            label_embed_dim=embed_dim,
            denoising_nums=denoising_nums,
            label_noise_prob=0.5,
            box_noise_scale=1.0,
        )

    def forward(self, images: List[Tensor], targets: List[Dict] = None):
        # get original image sizes, used for postprocess
        original_image_sizes = self.query_original_sizes(images)
        images, targets, mask = self.preprocess(images, targets)

        # get multi-level features, masks, and pos_embeds
        multi_levels = self.get_multi_levels(images, mask)
        multi_level_feats, multi_level_masks, multi_level_pos_embeds = multi_levels

        if self.training:
            # collect ground truth for denoising generation
            gt_labels_list = [t["labels"] for t in targets]
            gt_boxes_list = [t["boxes"] for t in targets]
            noised_results = self.denoising_generator(gt_labels_list, gt_boxes_list)
            noised_label_queries = noised_results[0]
            noised_box_queries = noised_results[1]
            attn_mask = noised_results[2]
            denoising_groups = noised_results[3]
            max_gt_num_per_image = noised_results[4]
        else:
            noised_label_queries = None
            noised_box_queries = None
            attn_mask = None
            denoising_groups = None
            max_gt_num_per_image = None

        # feed into transformer
        (
            outputs_class,
            outputs_coord,
            enc_class,
            enc_coord,
            hybrid_class,
            hybrid_coord,
            hybrid_enc_class,
            hybrid_enc_coord,
        ) = self.transformer(
            multi_level_feats,
            multi_level_masks,
            multi_level_pos_embeds,
            noised_label_queries,
            noised_box_queries,
            attn_mask=attn_mask,
        )

        # hack implemantation for distributed training
        outputs_class[0] += self.denoising_generator.label_encoder.weight[0, 0] * 0.0

        # denoising postprocessing
        if denoising_groups is not None and max_gt_num_per_image is not None:
            dn_metas = {
                "denoising_groups": denoising_groups,
                "max_gt_num_per_image": max_gt_num_per_image
            }
            outputs_class, outputs_coord = self.dn_post_process(
                outputs_class, outputs_coord, dn_metas
            )

        # prepare for loss computation
        output = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        output["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)
        output["enc_outputs"] = {"pred_logits": enc_class, "pred_boxes": enc_coord}

        if self.training:
            # prepare for hybrid loss computation
            hybrid_metas = {"pred_logits": hybrid_class[-1], "pred_boxes": hybrid_coord[-1]}
            hybrid_metas["aux_outputs"] = self._set_aux_loss(hybrid_class, hybrid_coord)
            hybrid_metas["enc_outputs"] = {
                "pred_logits": hybrid_enc_class,
                "pred_boxes": hybrid_enc_coord
            }

            # compute loss
            loss_dict = self.criterion(output, targets)
            dn_losses = self.compute_dn_loss(dn_metas, targets)
            loss_dict.update(dn_losses)

            # compute hybrid loss
            multi_targets = copy.deepcopy(targets)
            for t in multi_targets:
                t["boxes"] = t["boxes"].repeat(self.hybrid_assign, 1)
                t["labels"] = t["labels"].repeat(self.hybrid_assign)
            hybrid_losses = self.criterion(hybrid_metas, multi_targets)
            loss_dict.update({k + "_hybrid": v for k, v in hybrid_losses.items()})

            # loss reweighting
            weight_dict = self.criterion.weight_dict
            loss_dict = dict((k, loss_dict[k] * weight_dict[k])
                             for k in loss_dict.keys()
                             if k in weight_dict)
            return loss_dict

        detections = self.postprocessor(output, original_image_sizes)
        return detections
