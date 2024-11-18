from typing import Dict, List

from torch import Tensor, nn

from models.detectors.base_detector import DETRDetector


class DabDeformableDETR(DETRDetector):
    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module,
        position_embedding: nn.Module,
        transformer: nn.Module,
        criterion: nn.Module,
        postprocessor: nn.Module,
        num_classes: int,
        min_size: int = None,
        max_size: int = None,
    ):
        super().__init__(min_size, max_size)
        # define model parameters
        self.num_classes = num_classes

        # define model strctures
        self.backbone = backbone
        self.neck = neck
        self.position_embedding = position_embedding
        self.transformer = transformer
        self.criterion = criterion
        self.postprocessor = postprocessor

    def forward(self, images: List[Tensor], targets: List[Dict] = None):
        # get original image sizes, used for postprocess
        original_image_sizes = self.query_original_sizes(images)
        images, targets, mask = self.preprocess(images, targets)

        # get multi-level features, masks, and pos_embeds
        multi_levels = self.get_multi_levels(images, mask)
        multi_level_feats, multi_level_masks, multi_level_pos_embeds = multi_levels

        # feed into transformer
        outputs_class, outputs_coord, enc_class, enc_coord = self.transformer(
            multi_level_feats, multi_level_masks, multi_level_pos_embeds
        )

        # prepare for loss computation
        output = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        output["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)
        output["enc_outputs"] = {"pred_logits": enc_class, "pred_boxes": enc_coord}

        if self.training:
            # compute loss
            loss_dict = self.criterion(output, targets)

            # loss reweighting
            weight_dict = self.criterion.weight_dict
            loss_dict = dict((k, loss_dict[k] * weight_dict[k])
                             for k in loss_dict.keys()
                             if k in weight_dict)
            return loss_dict

        detections = self.postprocessor(output, original_image_sizes)
        return detections
