from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops.stochastic_depth import StochasticDepth

from models.backbones.base_backbone import BaseBackbone
from models.bricks.misc import Conv2dNormActivation, Permute
from util.lazy_load import LazyCall as L
from util.lazy_load import instantiate
from util.utils import load_checkpoint


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class CNBlock(nn.Module):
    def __init__(
        self,
        dim,
        layer_scale: float,
        stochastic_depth_prob: float,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True),
            Permute([0, 2, 3, 1]),
            norm_layer(dim),
            nn.Linear(in_features=dim, out_features=4 * dim, bias=True),
            nn.GELU(),
            nn.Linear(in_features=4 * dim, out_features=dim, bias=True),
            Permute([0, 3, 1, 2]),
        )
        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * layer_scale)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def forward(self, input: Tensor) -> Tensor:
        result = self.layer_scale * self.block(input)
        result = self.stochastic_depth(result)
        result += input
        return result


class CNBlockConfig:
    # Stores information listed at Section 3 of the ConvNeXt paper
    def __init__(
        self,
        input_channels: int,
        out_channels: Optional[int],
        num_layers: int,
    ) -> None:
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "input_channels={input_channels}"
        s += ", out_channels={out_channels}"
        s += ", num_layers={num_layers}"
        s += ")"
        return s.format(**self.__dict__)


class ConvNeXt(nn.Module):
    def __init__(
        self,
        block_setting: List[CNBlockConfig],
        stochastic_depth_prob: float = 0.0,
        layer_scale: float = 1e-6,
        num_classes: int = 1000,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        if not block_setting:
            raise ValueError("The block_setting should not be empty")
        elif not (
            isinstance(block_setting, Sequence) and all([isinstance(s, CNBlockConfig) for s in block_setting])
        ):
            raise TypeError("The block_setting should be List[CNBlockConfig]")

        if block is None:
            block = CNBlock

        if norm_layer is None:
            norm_layer = partial(LayerNorm2d, eps=1e-6)

        layers: List[nn.Module] = []

        # Stem
        firstconv_output_channels = block_setting[0].input_channels
        layers.append(
            Conv2dNormActivation(
                3,
                firstconv_output_channels,
                kernel_size=4,
                stride=4,
                padding=0,
                norm_layer=norm_layer,
                activation_layer=None,
                bias=True,
            )
        )

        total_stage_blocks = sum(cnf.num_layers for cnf in block_setting)
        stage_block_id = 0
        for cnf in block_setting:
            # Bottlenecks
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
                stage.append(block(cnf.input_channels, layer_scale, sd_prob))
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            if cnf.out_channels is not None:
                # Downsampling
                layers.append(
                    nn.Sequential(
                        norm_layer(cnf.input_channels),
                        nn.Conv2d(cnf.input_channels, cnf.out_channels, kernel_size=2, stride=2),
                    )
                )

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        lastblock = block_setting[-1]
        lastconv_output_channels = (
            lastblock.out_channels if lastblock.out_channels is not None else lastblock.input_channels
        )
        self.classifier = nn.Sequential(
            norm_layer(lastconv_output_channels), nn.Flatten(1),
            nn.Linear(lastconv_output_channels, num_classes)
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


class ConvNeXtBackbone(BaseBackbone):

    # yapf: disable
    model_weights = {
        # The following weights are from torchvision
        "conv_t": "https://download.pytorch.org/models/convnext_tiny-983f1562.pth",
        "conv_s": "https://download.pytorch.org/models/convnext_small-0c510722.pth",
        "conv_b": "https://download.pytorch.org/models/convnext_base-6075fbad.pth",
        "conv_l": "https://download.pytorch.org/models/convnext_large-ea097f82.pth",
    }

    model_arch = {
        "conv_t": L(ConvNeXt)(
            block_setting=[
                CNBlockConfig(96, 192, 3),
                CNBlockConfig(192, 384, 3),
                CNBlockConfig(384, 768, 9),
                CNBlockConfig(768, None, 3),
            ],
            stochastic_depth_prob=0.1,
            url=model_weights["conv_t"],
        ),
        "conv_s": L(ConvNeXt)(
            block_setting=[
                CNBlockConfig(96, 192, 3),
                CNBlockConfig(192, 384, 3),
                CNBlockConfig(384, 768, 27),
                CNBlockConfig(768, None, 3),
            ],
            stochastic_depth_prob=0.4,
            url=model_weights["conv_s"],
        ),
        "conv_b": L(ConvNeXt)(
            block_setting = [
                CNBlockConfig(128, 256, 3),
                CNBlockConfig(256, 512, 3),
                CNBlockConfig(512, 1024, 27),
                CNBlockConfig(1024, None, 3),
            ],
            stochastic_depth_prob=0.5,
            url=model_weights["conv_b"],
        ),
        "conv_l": L(ConvNeXt)(
            block_setting = [
                CNBlockConfig(192, 384, 3),
                CNBlockConfig(384, 768, 3),
                CNBlockConfig(768, 1536, 27),
                CNBlockConfig(1536, None, 3),
            ],
            stochastic_depth_prob=0.5,
            url=model_weights["conv_l"],
        )
    }
    # yapf: enable

    def __new__(
        self,
        arch: str,
        weights: Union[str, Dict] = None,
        return_indices: Tuple[int] = (0, 1, 2, 3),
        freeze_indices: Tuple = (),
        **kwargs,
    ):
        # get parameters and instantiate backbone
        model_config = self.get_instantiate_config(self, ConvNeXt, arch, kwargs)
        default_weight = model_config.pop("url", None)
        convnext = instantiate(model_config)

        # load state dict
        weights = load_checkpoint(default_weight if weights is None else weights)
        if isinstance(weights, Dict):
            weights = weights["model"] if "model" in weights else weights
        self.load_state_dict(convnext, weights)

        # freeze stages
        self._freeze_stages(self, convnext, freeze_indices)

        # create feature extractor
        return_layers = [f"features.{2 * idx + 1}" for idx in return_indices]
        convnext = create_feature_extractor(convnext, return_layers)
        convnext.num_channels = [model_config.block_setting[i].input_channels for i in return_indices]
        convnext.return_layers = return_layers
        return convnext

    def _freeze_stages(self, model: nn.Module, freeze_indices: Tuple[int]):
        # freeze stem
        if len(freeze_indices) > 0:
            self.freeze_module(model.features[0])

        for idx in freeze_indices:
            # freeze layers
            self.freeze_module(model.features[2 * idx + 1])
            # freeze downsample layers
            if 2 * idx + 2 < len(model.features):
                self.freeze_module(model.features[2 * idx + 2])
