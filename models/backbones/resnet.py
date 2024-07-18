from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import torch
from torch import Tensor, nn
from torch.nn.modules.batchnorm import _BatchNorm
from torchvision.models.feature_extraction import create_feature_extractor

from models.backbones.base_backbone import BaseBackbone
from models.bricks.deform_conv2d_pack import DeformConv2dPack
from models.bricks.misc import FrozenBatchNorm2d
from util.lazy_load import LazyCall as L
from util.lazy_load import instantiate
from util.tune_mode_convbn import \
    turn_on_efficient_conv_bn_eval_for_single_model
from util.utils import load_checkpoint


def conv3x3(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv3x3_dcn(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1
) -> DeformConv2dPack:
    """3x3 deformable convolution with padding"""
    return DeformConv2dPack(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        with_dcn: bool = False,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        if with_dcn:
            self.conv2 = conv3x3_dcn(planes, planes)
        else:
            self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        with_dcn: bool = False,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)

        conv2 = conv3x3_dcn if with_dcn else conv3x3
        self.conv2 = conv2(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)

        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        stage_with_dcn: Optional[List[bool]] = None,  # we only add an extra parameter
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if stage_with_dcn is None:
            stage_with_dcn = [False] * 4
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], with_dcn=stage_with_dcn[0])
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
            with_dcn=stage_with_dcn[1],
        )
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
            with_dcn=stage_with_dcn[2],
        )
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
            with_dcn=stage_with_dcn[3],
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
        with_dcn: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
                with_dcn,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    with_dcn=with_dcn,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer in eval mode."""
        super(ResNet, self).train(mode)
        for m in self.modules():
            # trick: eval have effect on BatchNorm only
            if isinstance(m, _BatchNorm):
                m.eval()


class ResNetBackbone(BaseBackbone):

    # yapf: disable
    model_weights = {
        # The following weights are from torchvision
        "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
        "resnet34": "https://download.pytorch.org/models/resnet34-b627a593.pth",
        "resnet50_v1": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
        "resnet50_v2": "https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
        "resnet101_v1": "https://download.pytorch.org/models/resnet101-63fe2227.pth",
        "resnet101_v2": "https://download.pytorch.org/models/resnet101-cd907fc2.pth",
        "resnet152_v1": "https://download.pytorch.org/models/resnet152-394f9c45.pth",
        "resnet152_v2": "https://download.pytorch.org/models/resnet152-f82ba261.pth",
        "resnext50_32x4d_v1": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
        "resnext50_32x4d_v2": "https://download.pytorch.org/models/resnext50_32x4d-1a0047aa.pth",
        "resnext101_32x8d_v1": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
        "resnext101_32x8d_v2": "https://download.pytorch.org/models/resnext101_32x8d-110c445d.pth",
        "resnext101_64x4d": "https://download.pytorch.org/models/resnext101_64x4d-173b62eb.pth",
        "wide_resnet50_2_v1": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
        "wide_resnet50_2_v2": "https://download.pytorch.org/models/wide_resnet50_2-9ba9bcbe.pth",
        "wide_resnet101_2_v1": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
        "wide_resnet101_2_v2": "https://download.pytorch.org/models/wide_resnet101_2-d733dc28.pth",
        # The following weights are transfomed from mmpretrain
        "resnext101_32x4d":
        "https://github.com/xiuqhou/pretrained_weights/releases/download/v1.0.1-beta/resnext101_32x4d-e0fa3dd5.pth",
    }

    model_arch = {
        "resnet18": L(ResNet)(block=BasicBlock, layers=(2, 2, 2, 2), url=model_weights["resnet18"]),
        "resnet34": L(ResNet)(block=BasicBlock, layers=(3, 4, 6, 3), url=model_weights["resnet34"]),
        "resnet50": L(ResNet)(block=Bottleneck, layers=(3, 4, 6, 3), url=model_weights["resnet50_v2"]),
        "resnet101": L(ResNet)(block=Bottleneck, layers=(3, 4, 23, 3), url=model_weights["resnet101_v2"]),
        "resnet152": L(ResNet)(block=Bottleneck, layers=(3, 8, 36, 3), url=model_weights["resnet152_v2"]),
        "resnext50_32x4d": L(ResNet)(
            block=Bottleneck,
            layers=(3, 4, 6, 3),
            groups=32,
            width_per_group=4,
            url=model_weights["resnext50_32x4d_v2"],
        ),
        "resnext101_32x4d": L(ResNet)(
            block=Bottleneck,
            layers=(3, 4, 23, 3),
            groups=32,
            width_per_group=4,
            url=model_weights["resnext101_32x4d"],
        ),
        "resnext101_32x8d": L(ResNet)(
            block=Bottleneck,
            layers=(3, 4, 23, 3),
            groups=32,
            width_per_group=8,
            url=model_weights["resnext101_32x8d_v2"],
        ),
        "resnext101_64x4d": L(ResNet)(
            block=Bottleneck,
            layers=(3, 4, 23, 3),
            groups=64,
            width_per_group=4,
            url=model_weights["resnext101_64x4d"],
        ),
        "wide_resnet50_2": L(ResNet)(
            block=Bottleneck,
            layers=(3, 4, 6, 3),
            width_per_group=64 * 2,
            url=model_weights["wide_resnet50_2_v2"],
        ),
        "wide_resnet101_2": L(ResNet)(
            block=Bottleneck,
            layers=(3, 4, 23, 3),
            width_per_group=64 * 2,
            url=model_weights["wide_resnet101_2_v2"],
        ),
    }
    # yapf: enable

    def __new__(
        self,
        arch: str,
        weights: Dict = None,
        return_indices: Tuple[int] = (0, 1, 2, 3),
        freeze_indices: Tuple = (),
        fuse_conv_bn: bool = True,
        **kwargs,
    ):
        # get parameters and instantiate backbone
        model_config = self.get_instantiate_config(self, ResNet, arch, kwargs)
        default_weight = model_config.pop("url", None)
        resnet = instantiate(model_config)

        # load state dict
        weights = load_checkpoint(default_weight if weights is None else weights)
        if isinstance(weights, Dict):
            weights = weights["model"] if "model" in weights else weights
        self.load_state_dict(resnet, weights)

        # freeze stages
        self._freeze_stages(self, resnet, freeze_indices)

        # fuse conv bn
        if fuse_conv_bn:
            turn_on_efficient_conv_bn_eval_for_single_model(resnet)
        # create feature extractor
        return_layers = [f"layer{idx + 1}" for idx in return_indices]
        resnet = create_feature_extractor(
            resnet, return_layers, tracer_kwargs={"leaf_modules": [FrozenBatchNorm2d]}
        )
        resnet.num_channels = [
            64 * model_config.block.expansion * 2**idx for idx in return_indices
        ]
        resnet.return_layers = return_layers
        return resnet

    def _freeze_stages(self, model: nn.Module, freeze_indices: Tuple[int]):
        # freeze stem
        if len(freeze_indices) > 0:
            self.freeze_module(model.conv1)
            self.freeze_module(model.bn1)

        # freeze layers
        for idx in freeze_indices:
            self.freeze_module(model.get_submodule(f"layer{idx+1}"))
