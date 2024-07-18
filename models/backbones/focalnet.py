from collections import OrderedDict
from functools import partial
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops import StochasticDepth

from models.backbones.base_backbone import BaseBackbone
from util.lazy_load import LazyCall as L
from util.lazy_load import instantiate
from util.utils import load_checkpoint


class Mlp(nn.Module):
    """Multilayer perceptron."""
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        activation_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = activation_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class FocalModulation(nn.Module):
    """ Focal Modulation

    Args:
        dim (int): Number of input channels.
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        focal_level (int): Number of focal levels
        focal_window (int): Focal window size at focal level 1
        focal_factor (int, default=2): Step to increase the focal window
        use_postln (bool, default=False): Whether use post-modulation layernorm
    """
    def __init__(
        self,
        dim: int,
        proj_drop=0.,
        focal_level=2,
        focal_window=7,
        focal_factor=2,
        use_postln_in_modulation=False,
        normalize_modulator=False
    ):

        super().__init__()
        self.dim = dim

        # specific args for focalv3
        self.focal_level = focal_level
        self.focal_window = focal_window
        self.focal_factor = focal_factor
        self.use_postln_in_modulation = use_postln_in_modulation
        self.normalize_modulator = normalize_modulator

        self.f = nn.Linear(dim, 2 * dim + (self.focal_level + 1))
        self.h = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, groups=1)

        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.focal_layers = nn.ModuleList()

        if self.use_postln_in_modulation:
            self.ln = nn.LayerNorm(dim)

        for k in range(self.focal_level):
            kernel_size = self.focal_factor * k + self.focal_window
            self.focal_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        dim,
                        dim,
                        kernel_size=kernel_size,
                        stride=1,
                        groups=dim,
                        padding=kernel_size // 2,
                        bias=False
                    ),
                    nn.GELU(),
                )
            )

    def forward(self, x):
        """ Forward function.
        Args:
            x: input features with shape of (B, H, W, C)
        """
        C = x.shape[-1]

        # pre linear projection
        x = self.f(x).permute(0, 3, 1, 2).contiguous()
        q, ctx, gates = torch.split(x, (C, C, self.focal_level + 1), 1)

        # context aggregation
        ctx_all = 0
        for l in range(self.focal_level):
            ctx = self.focal_layers[l](ctx)
            ctx_all = ctx_all + ctx * gates[:, l:l + 1]
        ctx_global = self.act(ctx.mean(2, keepdim=True).mean(3, keepdim=True))
        ctx_all = ctx_all + ctx_global * gates[:, self.focal_level:]

        # normalize context
        if self.normalize_modulator:
            ctx_all = ctx_all / (self.focal_level + 1)

        # focal modulation
        x_out = q * self.h(ctx_all)
        x_out = x_out.permute(0, 2, 3, 1).contiguous()
        if self.use_postln_in_modulation:
            x_out = self.ln(x_out)

        # post linear projection
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        return x_out


class FocalModulationBlock(nn.Module):
    """ Focal Modulation Block.
    Args:
        dim (int): Number of input channels.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        focal_level (int): number of focal levels
        focal_window (int): focal kernel size at level 1
    """
    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        focal_level: int = 2,
        focal_window: int = 9,
        dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        use_postln: bool = False,
        use_postln_in_modulation: bool = False,
        normalize_modulator: bool = False,
        use_layerscale: bool = False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.modulation = FocalModulation(
            dim,
            focal_window=focal_window,
            focal_level=focal_level,
            proj_drop=dropout,
            use_postln_in_modulation=use_postln_in_modulation,
            normalize_modulator=normalize_modulator,
        )
        self.drop_path = StochasticDepth(stochastic_depth_prob, "row")
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim, hidden_features=int(dim * mlp_ratio), activation_layer=nn.GELU, drop=dropout
        )

        if use_layerscale:
            self.gamma_1 = nn.Parameter(torch.full((dim,), 1e-4))
            self.gamma_2 = nn.Parameter(torch.full((dim,), 1e-4))
        else:
            self.gamma_1 = self.gamma_2 = 1.0

        self.forward = self._forward_post if use_postln else self._forward_pre

    def _forward_post(self, x):
        x = x + self.drop_path(self.gamma_1 * self.norm1(self.modulation(x)))
        x = x + self.drop_path(self.gamma_2 * self.norm2(self.mlp(x)))
        return x

    def _forward_pre(self, x):
        x = x + self.drop_path(self.gamma_1 * self.modulation(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
        use_conv_embed (bool): Whether use overlapped convolution for patch embedding. Default: False
        is_stem (bool): Is the stem block or not.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        patch_size: List[int] = (4, 4),
        norm_layer: nn.Module = nn.LayerNorm,
        use_conv_embed: bool = False,
        is_stem: bool = False,
    ):
        super().__init__()
        self.patch_size = patch_size

        if use_conv_embed:
            # if we choose to use conv embedding, then we treat the stem and non-stem differently
            if is_stem:
                self.proj = nn.Conv2d(in_channels, hidden_channels, kernel_size=7, stride=4, padding=2)
            else:
                self.proj = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=2, padding=1)
        else:
            self.proj = nn.Conv2d(in_channels, hidden_channels, kernel_size=patch_size, stride=patch_size)

        self.norm = norm_layer(hidden_channels)

    def forward(self, input):
        """Forward function.
        Args:
            input (Tensor[N, H, W, C]): The input tensor or 4-dimensions.
        """
        H, W, _ = input.shape[-3:]
        # pad feature maps to multiples of patch size
        pad_r = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
        pad_b = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
        input = F.pad(input, (0, 0, 0, pad_r, 0, pad_b))

        # perform patch embed
        input = input.permute(0, 3, 1, 2)
        input = self.proj(input)
        input = input.permute(0, 2, 3, 1)

        input = self.norm(input)

        return input


class FocalNet(nn.Module):
    """Implement paper `Focal Modulation Networks <https://arxiv.org/pdf/2203.11926.pdf>`_

    Args:
        patch_size (int | tuple(int)): Patch size. Default: 4.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        dropout (float): Dropout rate.
        stochastic_depth_prob (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        focal_levels (Sequence[int]): Number of focal levels at four stages
        focal_windows (Sequence[int]): Focal window sizes at first focal level at four stages
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        use_conv_embed (bool): Whether use overlapped convolution for patch embedding
    """
    def __init__(
        self,
        patch_size: List[int],
        embed_dim: int,
        depths: List[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        stochastic_depth_prob: float = 0.3,  # 0.3 or 0.4 works better for large+ models
        norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-5),
        focal_levels=[3, 3, 3, 3],
        focal_windows=[3, 3, 3, 3],
        use_conv_embed=False,
        use_postln=False,
        use_postln_in_modulation=False,
        use_layerscale=False,
        normalize_modulator=False,
    ):
        super().__init__()
        self.num_layers = len(depths)

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            in_channels=3,
            hidden_channels=embed_dim,
            patch_size=patch_size,
            norm_layer=norm_layer,
            use_conv_embed=use_conv_embed,
            is_stem=True
        )

        self.pos_drop = nn.Dropout(p=dropout)

        self.layers = nn.ModuleList()
        total_stage_blocks = sum(depths)
        stage_block_id = 0
        # build FocalNet blocks
        for i_stage in range(len(depths)):
            blocks: List[nn.Module] = []
            dim = embed_dim * 2**i_stage
            for _ in range(depths[i_stage]):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / (total_stage_blocks - 1)
                blocks.append(
                    FocalModulationBlock(
                        dim,
                        mlp_ratio=mlp_ratio,
                        focal_level=focal_levels[i_stage],
                        focal_window=focal_windows[i_stage],
                        dropout=dropout,
                        stochastic_depth_prob=sd_prob,
                        norm_layer=norm_layer,
                        use_postln=use_postln,
                        use_postln_in_modulation=use_postln_in_modulation,
                        normalize_modulator=normalize_modulator,
                        use_layerscale=use_layerscale,
                    )
                )
                stage_block_id += 1
            stage = OrderedDict(blocks=nn.Sequential(*blocks))
            # add patch merging layer
            if i_stage < (len(depths) - 1):
                stage["downsample"] = PatchEmbed(dim, int(2 * dim), (2, 2), norm_layer, use_conv_embed)
            self.layers.append(nn.Sequential(stage))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # (b, c, h, w) -> (b, h, w, c)
        x = x.permute(0, 2, 3, 1)
        x = self.pos_drop(self.patch_embed(x))

        for i in range(self.num_layers):
            x = self.layers[i](x)

        return x


class PostProcess(nn.Module):
    def __init__(
        self,
        in_channels: int,
        return_indices: Tuple[int] = (0, 1, 2, 3),
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        self.return_indices = return_indices
        for channel, idx in zip(in_channels, return_indices):
            self.add_module(f"norm{idx}", norm_layer(channel))

    def forward(self, multi_level_feats: Dict[str, torch.Tensor]):
        for idx, (key, value) in zip(self.return_indices, multi_level_feats.items()):
            feat = getattr(self, f"norm{idx}")(value).permute(0, 3, 1, 2).contiguous()
            multi_level_feats[key] = feat
        return multi_level_feats


class FocalNetBackbone(BaseBackbone):
    model_weights = {
        # The following weights are from original repository
        "focalnet_tiny_srf":
        "https://github.com/microsoft/FocalNet/releases/download/v1.0.0/focalnet_tiny_srf.pth",
        "focalnet_tiny_lrf":
        "https://github.com/microsoft/FocalNet/releases/download/v1.0.0/focalnet_tiny_lrf.pth",
        "focalnet_small_srf":
        "https://github.com/microsoft/FocalNet/releases/download/v1.0.0/focalnet_small_srf.pth",
        "focalnet_small_lrf":
        "https://github.com/microsoft/FocalNet/releases/download/v1.0.0/focalnet_small_lrf.pth",
        "focalnet_base_srf":
        "https://github.com/microsoft/FocalNet/releases/download/v1.0.0/focalnet_base_srf.pth",
        "focalnet_base_lrf":
        "https://github.com/microsoft/FocalNet/releases/download/v1.0.0/focalnet_base_lrf.pth",
        "focalnet_large_lrf_384":
        "https://github.com/microsoft/FocalNet/releases/download/v1.0.0/focalnet_large_lrf_384.pth",
        "focalnet_large_lrf_384_fl4":
        "https://github.com/microsoft/FocalNet/releases/download/v1.0.0/focalnet_large_lrf_384_fl4.pth",
        "focalnet_xlarge_lrf_384":
        "https://github.com/microsoft/FocalNet/releases/download/v1.0.0/focalnet_xlarge_lrf_384.pth",
        "focalnet_xlarge_lrf_384_fl4":
        "https://github.com/microsoft/FocalNet/releases/download/v1.0.0/focalnet_xlarge_lrf_384_fl4.pth",
        # The following weights are from huggingface
        "focalnet_large_fl4_dino_o365":
        "https://huggingface.co/microsoft/focalnet-large-fl4-dino-o365/resolve/main/focalnet_large_fl4_pretrained_on_o365.pth",
        "focalnet_large_fl4_dino_o365_coco":
        "https://huggingface.co/microsoft/focalnet-large-fl4-dino-o365-cocoft/resolve/main/focalnet_large_fl4_o365_finetuned_on_coco.pth",
    }
    # yapf: disable
    model_arch = {
        "focalnet_tiny_srf": L(FocalNet)(
            embed_dim=96,
            patch_size=(4, 4),
            depths=(2, 2, 6, 2),
            stochastic_depth_prob=0.2,
            focal_levels=(2, 2, 2, 2),
            focal_windows=(3, 3, 3, 3),
            url=model_weights["focalnet_tiny_srf"],
        ),
        "focalnet_tiny_lrf": L(FocalNet)(
            embed_dim=96,
            patch_size=(4, 4),
            depths=(2, 2, 18, 2),
            stochastic_depth_prob=0.2,
            focal_levels=(3, 3, 3, 3),
            focal_windows=(3, 3, 3, 3),
            url=model_weights["focalnet_tiny_lrf"],
        ),
        "focalnet_small_srf": L(FocalNet)(
            embed_dim=96,
            patch_size=(4, 4),
            depths=(2, 2, 18, 2),
            stochastic_depth_prob=0.3,
            focal_levels=(2, 2, 2, 2),
            focal_windows=(3, 3, 3, 3),
            url=model_weights["focalnet_small_srf"],
        ),
        "focalnet_small_lrf": L(FocalNet)(
            embed_dim=96,
            patch_size=(4, 4),
            depths=(2, 2, 18, 2),
            stochastic_depth_prob=0.3,
            focal_levels=(3, 3, 3, 3),
            focal_windows=(3, 3, 3, 3),
            url=model_weights["focalnet_small_lrf"],
        ),
        "focalnet_base_srf": L(FocalNet)(
            embed_dim=128,
            patch_size=(4, 4),
            depths=(2, 2, 18, 2),
            stochastic_depth_prob=0.5,
            focal_levels=(2, 2, 2, 2),
            focal_windows=(3, 3, 3, 3),
            url=model_weights["focalnet_base_srf"],
        ),
        "focalnet_base_lrf": L(FocalNet)(
            embed_dim=128,
            patch_size=(4, 4),
            depths=(2, 2, 18, 2),
            stochastic_depth_prob=0.5,
            focal_levels=(3, 3, 3, 3),
            focal_windows=(3, 3, 3, 3),
            url=model_weights["focalnet_base_lrf"],
        ),
        "focalnet_large_lrf": L(FocalNet)(
            embed_dim=192,
            patch_size=(4, 4),
            depths=(2, 2, 18, 2),
            stochastic_depth_prob=0.5,
            focal_levels=(3, 3, 3, 3),
            focal_windows=(5, 5, 5, 5),
            use_conv_embed=True,
            use_postln=True,
            use_postln_in_modulation=False,
            use_layerscale=True,
            normalize_modulator=False,
            url=model_weights["focalnet_large_lrf_384"],
        ),
        "focalnet_large_lrf_fl4": L(FocalNet)(
            embed_dim=192,
            patch_size=(4, 4),
            depths=(2, 2, 18, 2),
            stochastic_depth_prob=0.5,
            focal_levels=(4, 4, 4, 4),
            focal_windows=(3, 3, 3, 3),
            use_conv_embed=True,
            use_postln=True,
            use_postln_in_modulation=False,
            use_layerscale=True,
            normalize_modulator=True,
            url=model_weights["focalnet_large_lrf_384_fl4"],
        ),
        "focalnet_xlarge_lrf": L(FocalNet)(
            embed_dim=256,
            patch_size=(4, 4),
            depths=(2, 2, 18, 2),
            stochastic_depth_prob=0.5,
            focal_levels=(3, 3, 3, 3),
            focal_windows=(5, 5, 5, 5),
            use_conv_embed=True,
            use_postln=True,
            use_postln_in_modulation=False,
            use_layerscale=True,
            normalize_modulator=False,
            url=model_weights["focalnet_xlarge_lrf_384"],
        ),
        "focalnet_xlarge_lrf_fl4": L(FocalNet)(
            embed_dim=256,
            patch_size=(4, 4),
            depths=(2, 2, 18, 2),
            stochastic_depth_prob=0.5,
            focal_levels=(4, 4, 4, 4),
            focal_windows=(3, 3, 3, 3),
            use_conv_embed=True,
            use_postln=True,
            use_postln_in_modulation=False,
            use_layerscale=True,
            normalize_modulator=True,
            url=model_weights["focalnet_xlarge_lrf_384_fl4"],
        ),
        "focalnet_huge_fl3": L(FocalNet)(
            embed_dim=352,
            patch_size=(4, 4),
            depths=(2, 2, 18, 2),
            stochastic_depth_prob=0.5,
            focal_levels=(3, 3, 3, 3),
            focal_windows=(3, 3, 3, 3),
            use_conv_embed=True,
            use_postln=True,
            use_layerscale=True,
            use_postln_in_modulation=True,
            normalize_modulator=False,
        ),
        "focalnet_huge_fl4": L(FocalNet)(
            embed_dim=352,
            patch_size=(4, 4),
            depths=(2, 2, 18, 2),
            stochastic_depth_prob=0.5,
            focal_levels=(4, 4, 4, 4),
            focal_windows=(3, 3, 3, 3),
            use_conv_embed=True,
            use_postln=True,
            use_postln_in_modulation=True,
            use_layerscale=True,
            normalize_modulator=False,
        ),
    }
    # yapf: enable

    def __new__(
        self,
        arch: str,
        weights: Dict = None,
        return_indices: Tuple[int] = (0, 1, 2, 3),
        freeze_indices: Tuple = (),
        **kwargs,
    ):
        # get parameters and instantiate backbone
        model_config = self.get_instantiate_config(self, FocalNet, arch, kwargs)
        default_weight = model_config.pop("url", None)
        focalnet = instantiate(model_config)

        # load state dict
        weights = load_checkpoint(default_weight if weights is None else weights)
        if isinstance(weights, Dict):
            weights = weights["model"] if "model" in weights else weights
        self.load_state_dict(focalnet, weights)

        # freeze stages
        self._freeze_stages(self, focalnet, freeze_indices)

        # create feature extractor
        return_layers = [f"layers.{idx}.blocks" for idx in return_indices]
        focalnet = create_feature_extractor(focalnet, return_layers)
        focalnet.num_channels = [model_config.embed_dim * 2**idx for idx in return_indices]

        # add post_process for feature extractor
        post_process = PostProcess(focalnet.num_channels, return_indices, model_config.norm_layer)
        backbone = nn.Sequential(focalnet, post_process)
        backbone.num_channels = focalnet.num_channels
        backbone.return_layers = return_layers
        return backbone

    def _freeze_stages(self, model: nn.Module, freeze_indices: Tuple[int]):
        # freeze patch embed
        if len(freeze_indices) > 0:
            self.freeze_module(model.patch_embed)

        # freeze layers
        for idx in freeze_indices:
            self.freeze_module(model.layers[idx])
