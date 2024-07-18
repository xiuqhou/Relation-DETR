import math
from collections import OrderedDict
from functools import partial
from math import pi
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models.convnext import LayerNorm2d
from torchvision.models.vision_transformer import ConvStemConfig, MLPBlock
from torchvision.ops.stochastic_depth import StochasticDepth

from models.backbones.base_backbone import BaseBackbone
from models.bricks.misc import Conv2dNormActivation
from util.lazy_load import LazyCall as L
from util.lazy_load import instantiate
from util.utils import load_checkpoint

try:
    import xformers.ops as xops
    HAS_XFORMER = True
except:
    HAS_XFORMER = False


def window_partition(x, window_size):
    """Partition into non-overlapping windows with padding if needed.

    :param x: input tokens with [B, H, W, C].
    :param window_size: window size.
    :return: windows and padded height and width
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(windows, window_size, pad_hw, hw):
    """Window unpartition into original sequences and removing padding.

    :param windows: input tokens with [B * num_windows, window_size, window_size, C].
    :param window_size: window size.
    :param pad_hw: padded height and width (Hp, Wp).
    :param hw: original height and width (H, W) before padding.
    :return: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def rotate_half(x):
    x = x.view(*x.shape[:-1], x.shape[-1] // 2, 2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return x.view(*x.shape[:-2], x.shape[-2] * x.shape[-1])


class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(
        self,
        dim,
        pt_seq_len=16,
        ft_seq_len=None,
        custom_freqs=None,
        freqs_for='lang',
        theta=10000,
        max_freq=10,
        num_freqs=1,
    ):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta**(torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')
        if ft_seq_len is None:
            ft_seq_len = pt_seq_len

        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len
        t = t.expand(ft_seq_len, -1)
        t = torch.stack([t.T, t], -1)
        freqs = t.unsqueeze(-1) * freqs
        freqs = freqs.repeat_interleave(2, -1).view(ft_seq_len, ft_seq_len, -1)
        freqs_cos = freqs.cos().view(-1, freqs.shape[-1])
        freqs_sin = freqs.sin().view(-1, freqs.shape[-1])

        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

    def forward(self, t):
        return t * self.freqs_cos + rotate_half(t) * self.freqs_sin


class SwiGLU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = None,
        out_channels: int = None,
        norm_layer: nn.Module = nn.LayerNorm,
        activation_layer: nn.Module = nn.SiLU,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels

        self.w1 = nn.Linear(in_channels, hidden_channels, bias=bias)
        self.w2 = nn.Linear(in_channels, hidden_channels, bias=bias)
        self.act = activation_layer()
        self.ffn_ln = nn.Identity() if norm_layer is None else norm_layer(hidden_channels)
        self.w3 = nn.Linear(hidden_channels, out_channels, bias=bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = self.act(x1) * x2
        x = self.ffn_ln(hidden)
        x = self.w3(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        qk_scale=None,
        attn_head_dim=None,
        rope=None,
        xattn=True,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q_proj = nn.Linear(dim, all_head_dim, bias=False)
        self.k_proj = nn.Linear(dim, all_head_dim, bias=False)
        self.v_proj = nn.Linear(dim, all_head_dim, bias=False)

        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.rope = rope
        self.xattn = xattn
        self.proj = nn.Linear(all_head_dim, dim)

        if not HAS_XFORMER:
            self.xattn = False

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.view(B, -1, C)
        N = H * W

        q = F.linear(input=x, weight=self.q_proj.weight, bias=self.q_bias)
        k = F.linear(input=x, weight=self.k_proj.weight, bias=None)
        v = F.linear(input=x, weight=self.v_proj.weight, bias=self.v_bias)

        q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)  # B, num_heads, N, C
        k = k.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

        ## rope
        q = self.rope(q).type_as(v)
        k = self.rope(k).type_as(v)

        if self.xattn:
            q = q.permute(0, 2, 1, 3)  # B, num_heads, N, C -> B, N, num_heads, C
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)

            x = xops.memory_efficient_attention(q, k, v)
            x = x.reshape(B, N, -1)
        else:
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))
            attn = attn.softmax(dim=-1).type_as(x)
            x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        x = x.view(B, H, W, C)

        return x


class ResBottleneckBlock(nn.Module):
    """
    The standard bottleneck residual block without the last activation layer.
    It contains 3 conv layers with kernels 1x1, 3x3, 1x1.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        bottleneck_channels,
        inplace=True,
        norm_layer=nn.LayerNorm,
        activation_layer=nn.GELU,
    ):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            bottleneck_channels (int): number of output channels for the 3x3
                "bottleneck" conv layers.
            norm (str or callable): normalization for all conv layers.
                See :func:`layers.get_norm` for supported format.
            act_layer (callable): activation for all conv layers.
        """
        super().__init__(in_channels, out_channels, 1)

        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, 1, bias=False)
        self.norm1 = norm_layer(bottleneck_channels)
        self.act1 = activation_layer(inplace=inplace)

        self.conv2 = nn.Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            3,
            padding=1,
            bias=False,
        )
        self.norm2 = norm_layer(bottleneck_channels)
        self.act2 = activation_layer(inplace=inplace)

        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, 1, bias=False)
        self.norm3 = norm_layer(out_channels)

        for layer in [self.conv1, self.conv2, self.conv3]:
            nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
        for layer in [self.norm1, self.norm2]:
            layer.weight.data.fill_(1.0)
            layer.bias.data.zero_()
        # zero init last norm layer.
        self.norm3.weight.data.zero_()
        self.norm3.bias.data.zero_()

    def forward(self, x):
        out = x
        for layer in self.children():
            out = layer(out)

        out = x + out
        return out


class EncoderBlock(nn.Module):
    """Transformer encoder block."""
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        drop_path: float = 0.0,
        rope: nn.Module = None,
        use_swiglu: bool = False,
        window_size: int = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        # NOTE: Different from pytorch ViT, may use rope attention
        self.use_rope = rope is not None
        if rope is not None:
            self.self_attention = Attention(hidden_dim, num_heads, rope=rope)
        else:
            self.self_attention = nn.MultiheadAttention(
                hidden_dim, num_heads, dropout=attention_dropout, batch_first=True
            )
        self.dropout = nn.Dropout(dropout)

        # NOTE: Different from pytorch ViT, we add StochasticDepth
        self.stochastic_depth = StochasticDepth(drop_path, mode="row")

        # NOTE: Different from pytorch ViT, may use SwiGLU as MLPBlock
        self.ln_2 = norm_layer(hidden_dim)
        if use_swiglu:
            self.mlp = SwiGLU(hidden_dim, mlp_dim, norm_layer=norm_layer)
        else:
            self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)
        self.window_size = window_size

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)

        if self.use_rope:
            # NOTE: for backbone variants of EVA02, remove batch_class_token
            batch_class_token, x = x[:, :1, :], x[:, 1:, :]
            b, n, d = x.shape
            n_h = n_w = int(n**0.5)
            assert n == n_h * n_w, "height and width of image must be equal"
            x = x.view(b, n_h, n_w, d)

            # window partition following EVA02
            if self.window_size > 0:
                x, pad_hw = window_partition(x, self.window_size)
            x = self.self_attention(x)

            # NOTE: reverse window partition, following EVA02
            if self.window_size > 0:
                x = window_unpartition(x, self.window_size, pad_hw, (n_h, n_w))
            x = x.view(b, n, d)
            x = torch.cat([batch_class_token, x], dim=1)
        else:
            x, _ = self.self_attention(x, x, x, need_weights=False)

        x = self.stochastic_depth(self.dropout(x))
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        y = self.stochastic_depth(y)
        return x + y


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation"""
    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        drop_path_rate: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        use_rope: bool = False,
        use_swiglu: bool = False,
        window_block_indexes: Tuple[int] = (),
        patch_size: int = None,
        image_size: int = None,
        window_size: int = 0,
    ):
        super().__init__()
        # NOTE: EVA02 use different pos_embed
        self.use_rope = use_rope
        if self.use_rope:
            # Initialize absolute positional embedding with pretrain image size.
            num_patches = (224 // patch_size) * (224 // patch_size)
            num_positions = num_patches + 1
            self.pos_embedding = nn.Parameter(torch.zeros(1, num_positions, hidden_dim))
            nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        else:
            # Note that batch_size is on the first dim because
            # we have batch_first=True in nn.MultiAttention() by default
            self.pos_embedding = nn.Parameter(
                torch.empty(1, seq_length, hidden_dim).normal_(std=0.02)
            )  # from BERT
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()

        # rope
        if use_rope:
            assert image_size is not None and patch_size is not None, """
                image_size and patch_size cannot be None when using repo
            """
            self.rope_win = VisionRotaryEmbeddingFast(
                dim=hidden_dim // num_heads // 2,
                pt_seq_len=patch_size,
                ft_seq_len=window_size,
            )
            self.rope_glb = VisionRotaryEmbeddingFast(
                dim=hidden_dim // num_heads // 2,
                pt_seq_len=patch_size,
                ft_seq_len=image_size // patch_size,
            )
        # NOTE: stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        for i in range(num_layers):
            if use_rope:
                rope = self.rope_win if i in window_block_indexes else self.rope_glb
            else:
                rope = None
            cur_window_size = window_size if use_rope and i in window_block_indexes else 0
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                drop_path=dpr[i],
                rope=rope,
                use_swiglu=use_swiglu,
                window_size=cur_window_size,
                norm_layer=norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        if self.use_rope:
            # remove batch_class_token
            cls_embedding, pos_embedding = self.pos_embedding[:, :1], self.pos_embedding[:, 1:]
            patch_size = int(pos_embedding.shape[1]**0.5)
            assert patch_size * patch_size == pos_embedding.shape[1]

            image_size = int((input.shape[1] - 1)**0.5)
            assert image_size * image_size + 1 == input.shape[1]
            if patch_size != image_size:
                pos_embedding = pos_embedding.view(1, patch_size, patch_size, -1)
                pos_embedding = F.interpolate(
                    pos_embedding.permute(0, 3, 1, 2),
                    size=(image_size, image_size),
                    mode="bicubic",
                    align_corners=False,
                )
                pos_embedding = pos_embedding.permute(0, 2, 3, 1)
            pos_embedding = pos_embedding.view(1, image_size * image_size, -1)
            pos_embedding = torch.cat([cls_embedding, pos_embedding], 1)
        else:
            pos_embedding = self.pos_embedding
        input = input + pos_embedding
        return self.ln(self.layers(self.dropout(input)))


class VisionTransformer(nn.Module):
    """This module implements Vision Transformer as per https://arxiv.org/abs/2010.11929.
    and Vision Transformer (ViT) backbone as per :paper:`vitdet`. Exploring Plain Vision
    Transformer Backbones for Object Detection", https://arxiv.org/abs/2203.16527.
    """
    def __init__(
        self,
        image_size: int,
        patch_size: int = 16,
        num_layers: int = 12,
        num_heads: int = 12,
        hidden_dim: int = 768,
        mlp_dim: int = 3072,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 1000,
        representation_size: Optional[int] = None,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        conv_stem_configs: Optional[List[ConvStemConfig]] = None,
        # following EVA-02
        drop_path_rate: float = 0.0,
        use_rope: bool = False,
        use_swiglu: bool = False,
        window_size: int = 0,
        window_block_indexes: Tuple = (),
    ):
        super().__init__()
        torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.representation_size = representation_size
        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer

        if conv_stem_configs is not None:
            # As per https://arxiv.org/abs/2106.14881
            seq_proj = nn.Sequential()
            prev_channels = 3
            for i, conv_stem_layer_config in enumerate(conv_stem_configs):
                seq_proj.add_module(
                    f"conv_bn_relu_{i}",
                    Conv2dNormActivation(
                        in_channels=prev_channels,
                        out_channels=conv_stem_layer_config.out_channels,
                        kernel_size=conv_stem_layer_config.kernel_size,
                        stride=conv_stem_layer_config.stride,
                        norm_layer=conv_stem_layer_config.norm_layer,
                        activation_layer=conv_stem_layer_config.activation_layer,
                    ),
                )
                prev_channels = conv_stem_layer_config.out_channels
            seq_proj.add_module(
                "conv_last", nn.Conv2d(in_channels=prev_channels, out_channels=hidden_dim, kernel_size=1)
            )
            self.conv_proj: nn.Module = seq_proj
        else:
            self.conv_proj = nn.Conv2d(
                in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
            )

        seq_length = (image_size // patch_size)**2

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        seq_length += 1

        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            drop_path_rate,
            norm_layer,
            use_rope,
            use_swiglu,
            window_block_indexes,
            patch_size,
            image_size,
            window_size,
        )
        self.seq_length = seq_length

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        if representation_size is None:
            heads_layers["head"] = nn.Linear(hidden_dim, num_classes)
        else:
            heads_layers["pre_logits"] = nn.Linear(hidden_dim, representation_size)
            heads_layers["act"] = nn.Tanh()
            heads_layers["head"] = nn.Linear(representation_size, num_classes)

        self.heads = nn.Sequential(heads_layers)

        if isinstance(self.conv_proj, nn.Conv2d):
            # Init the patchify stem
            fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
            nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
            if self.conv_proj.bias is not None:
                nn.init.zeros_(self.conv_proj.bias)
        elif self.conv_proj.conv_last is not None and isinstance(self.conv_proj.conv_last, nn.Conv2d):
            # Init the last 1x1 conv of the conv stem
            nn.init.normal_(
                self.conv_proj.conv_last.weight,
                mean=0.0,
                std=math.sqrt(2.0 / self.conv_proj.conv_last.out_channels)
            )
            if self.conv_proj.conv_last.bias is not None:
                nn.init.zeros_(self.conv_proj.conv_last.bias)

        if hasattr(self.heads, "pre_logits") and isinstance(self.heads.pre_logits, nn.Linear):
            fan_in = self.heads.pre_logits.in_features
            nn.init.trunc_normal_(self.heads.pre_logits.weight, std=math.sqrt(1 / fan_in))
            nn.init.zeros_(self.heads.pre_logits.bias)

        if isinstance(self.heads.head, nn.Linear):
            nn.init.zeros_(self.heads.head.weight)
            nn.init.zeros_(self.heads.head.bias)

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = self.heads(x)

        return x


class VisionTransformerNoHead(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.heads

    def _process_input(self, x: Tensor) -> Tensor:
        h, w = x.shape[-2:]
        torch._assert(
            h <= self.image_size, f"Image height must be smaller than {self.image_size} but got {h}!"
        )
        torch._assert(
            w <= self.image_size, f"Image width must be smaller than {self.image_size} but got {w}!"
        )

        x = F.pad(x, (0, self.image_size - w, 0, self.image_size - h), value=0)
        n, _, h, w = x.shape
        n_h = h // self.patch_size
        n_w = w // self.patch_size

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x):
        h, w = x.shape[-2:]
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        # remove class token
        x = x[:, 1:]
        num_seq = x.shape[1]
        size = int(num_seq**0.5)
        assert size * size == num_seq
        x = x.view(n, size, size, -1)
        n_h = h // self.patch_size
        n_w = w // self.patch_size
        x = x[:, :n_h, :n_w, :].contiguous().permute(0, 3, 1, 2)

        # (b, c, h, w)
        return x


class SimpleFeaturePyramid(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factors, extra_block=False, norm_layer=LayerNorm2d):
        super(SimpleFeaturePyramid, self).__init__()
        self.scale_factors = scale_factors

        for scale in scale_factors:
            out_dim = in_channels
            if scale == 4.0:
                layers = [
                    nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2),
                    norm_layer(in_channels // 2),
                    nn.GELU(),
                    nn.ConvTranspose2d(in_channels // 2, in_channels // 4, kernel_size=2, stride=2),
                ]
                out_dim = in_channels // 4
            elif scale == 2.0:
                layers = [nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)]
                out_dim = in_channels // 2
            elif scale == 1.0:
                layers = []
            elif scale == 0.5:
                layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported yet.")

            layers.extend([
                Conv2dNormActivation(out_dim, out_channels, kernel_size=1, norm_layer=norm_layer),
                Conv2dNormActivation(
                    out_channels, out_channels, kernel_size=3, padding=1, norm_layer=norm_layer
                ),
            ])
            layers = nn.Sequential(*layers)

            stage = 4 - int(math.log2(scale))
            self.add_module(f"stage_{stage}", layers)

        self.in_channels = in_channels
        self.extra_block = extra_block

    def forward(self, x):
        results = {}

        for stage_index in range(6):
            cur_stage = getattr(self, f"stage_{stage_index}", None)
            if cur_stage is None:
                continue

            cur_stage_feature = cur_stage(x)
            # layer1, layer2, layer3, layer4
            #   p2,     p3,     p4,     p5
            results[f"layer{stage_index - 1}"] = cur_stage_feature
            last_stage_index = stage_index

        if self.extra_block:
            extra_feature = F.max_pool2d(cur_stage_feature, kernel_size=1, stride=2, padding=0)
            results[f"layer{last_stage_index}"] = extra_feature
        return results


class VisionTransformerBackbone(BaseBackbone):

    model_weights = {
        # The following weights are from torchvision
        "vit_b_16":
            "https://download.pytorch.org/models/vit_b_16-c867db91.pth",
        "vit_b_16_swag_e2e_v1":
            "https://download.pytorch.org/models/vit_b_16_swag-9ac1b537.pth",
        "vit_b_16_swag_linear_v1":
            "https://download.pytorch.org/models/vit_b_16_lc_swag-4e70ced5.pth",
        "vit_b_32":
            "https://download.pytorch.org/models/vit_b_32-d86f8d99.pth",
        "vit_l_16":
            "https://download.pytorch.org/models/vit_l_16-852ce7e3.pth",
        "vit_l_16_swag_e2e_v1":
            "https://download.pytorch.org/models/vit_l_16_swag-4f3808c9.pth",
        "vit_l_16_swag_linear_v1":
            "https://download.pytorch.org/models/vit_l_16_lc_swag-4d563306.pth",
        "vit_l_32":
            "https://download.pytorch.org/models/vit_l_32-c7638314.pth",
        "vit_h_14_swag_e2e_v1":
            "https://download.pytorch.org/models/vit_h_14_swag-80465313.pth",
        "vit_h_14_swag_linear_v1":
            "https://download.pytorch.org/models/vit_h_14_lc_swag-c1eb923e.pth",
        "eva02_B_pt_in21k_p14":
            "https://hf-mirror.com/Yuxin-CV/EVA-02/raw/main/eva02/pt/eva02_B_pt_in21k_p14.pt",
        "eva02_B_pt_in21k_p14to16":
            "https://hf-mirror.com/Yuxin-CV/EVA-02/raw/main/eva02/pt/eva02_B_pt_in21k_p14to16.pt"
    }

    model_arch = {
        "vit_b_16": L(VisionTransformerNoHead)(
            image_size=224,
            mlp_dim=3072,
            url=model_weights["vit_b_16"],
        ),
        "vit_b_32": L(VisionTransformerNoHead)(
            image_size=224,
            patch_size=32,
            mlp_dim=3072,
            url=model_weights["vit_b_32"],
        ),
        "vit_l_16": L(VisionTransformerNoHead)(
            image_size=224,
            num_layers=24,
            num_heads=16,
            hidden_dim=1024,
            mlp_dim=4096,
            url=model_weights["vit_l_16"],
        ),
        "vit_l_32": L(VisionTransformerNoHead)(
            image_size=224,
            patch_size=32,
            num_layers=24,
            num_heads=16,
            hidden_dim=1024,
            mlp_dim=4096,
            url=model_weights["vit_l_32"],
        ),
        "vit_h_14": L(VisionTransformerNoHead)(
            image_size=224,
            patch_size=14,
            num_layers=32,
            num_heads=16,
            hidden_dim=1280,
            mlp_dim=5120,
            url=model_weights["vit_h_14_swag_e2e_v1"],
        ),
        "eva_02_vit_b_4attn_1024": L(VisionTransformerNoHead)(
            image_size=1024,
            hidden_dim=768,
            mlp_dim=2048,
            drop_path_rate=0.1,
            use_rope=True,
            use_swiglu=True,
            window_size=16,
            window_block_indexes=(0, 1, 3, 4, 6, 7, 9, 10),
        ),
        "eva_02_vit_b_6attn_win32_1536": L(VisionTransformerNoHead)(
            image_size=1536,
            hidden_dim=768,
            mlp_dim=2048,
            drop_path_rate=0.1,
            use_rope=True,
            use_swiglu=True,
            window_size=32,
            window_block_indexes=(0, 2, 4, 6, 8, 10),
        ),
        "eva_02_vit_l_4attn_1024": L(VisionTransformerNoHead)(
            image_size=1024,
            num_layers=24,
            num_heads=16,
            hidden_dim=1024,
            mlp_dim=2730,
            drop_path_rate=0.4,
            use_rope=True,
            use_swiglu=True,
            window_size=16,
            window_block_indexes=[
                0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, \
                13, 14, 15, 16, 18, 19, 20, 21, 22
            ]
        ),
        "eva_02_vit_l_8attn_1536": L(VisionTransformerNoHead)(
            image_size=1536,
            num_layers=24,
            num_heads=16,
            hidden_dim=1024,
            mlp_dim=2730,
            drop_path_rate=0.3,
            use_rope=True,
            use_swiglu=True,
            window_size=16,
            window_block_indexes=[
                0, 1, 3, 4, 6, 7, 9, 10, 12, \
                13, 15, 16, 18, 19, 21, 22
            ]
        ),
        "eva_02_vit_l_8attn_win32_1536": L(VisionTransformerNoHead)(
            image_size=1536,
            num_layers=24,
            num_heads=16,
            hidden_dim=1024,
            mlp_dim=2730,
            drop_path_rate=0.4,
            use_rope=True,
            use_swiglu=True,
            window_size=32,
            window_block_indexes=[
                0, 1, 3, 4, 6, 7, 9, 10, 12, \
                13, 15, 16, 18, 19, 21, 22
            ],
        )
    }

    def __new__(
        self,
        arch: str,
        weights: Dict = None,
        return_indices: Tuple[int] = (0, 1, 2, 3),
        **kwargs,
    ):
        # get parameters and instantiate backbone
        model_config = self.get_instantiate_config(self, VisionTransformer, arch, kwargs)
        default_weight = model_config.pop("url", None)
        if "image_size" in model_config and "patch_size" in model_config:
            divise = model_config["image_size"] / model_config["patch_size"]
            model_config["image_size"] = math.ceil(divise) * model_config["patch_size"]
        vit = instantiate(model_config)

        # load state dict
        weights = load_checkpoint(default_weight if weights is None else weights)
        if isinstance(weights, Dict):
            weights = weights["model"] if "model" in weights else weights
        self.load_state_dict(vit, weights)

        scale_factors = [2**(2 - key) for key in return_indices]
        fpn = SimpleFeaturePyramid(
            vit.hidden_dim,
            256,
            scale_factors=scale_factors,
            extra_block=4 in return_indices,
        )
        backbone = nn.Sequential(vit, fpn)
        backbone.num_channels = [256] * len(return_indices)

        # collect key names of backbone output
        backbone.return_layers = [f"layer{idx + 1}" for idx in return_indices]
        return backbone
