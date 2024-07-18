import math
import os
import warnings

import torch
import torchvision
from torch import Tensor, nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn import functional as F
from torch.nn.init import constant_, xavier_uniform_
from torch.utils.cpp_extension import load

_C = None
if torch.cuda.is_available():
    try:
        _C = load(
            "MultiScaleDeformableAttention",
            sources=[f"{os.path.dirname(__file__)}/ops/cuda/ms_deform_attn_cuda.cu"],
            extra_cflags=["-O2"],
            verbose=True,
        )
    except Exception as e:
        warnings.warn(f"Failed to load MultiScaleDeformableAttention C++ extension: {e}")
else:
    warnings.warn("No cuda is available, skip loading MultiScaleDeformableAttention C++ extention")


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n - 1) == 0) and n != 0


class MultiScaleDeformableAttnFunction(Function):
    @staticmethod
    def forward(
        ctx,
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step,
    ):
        ctx.im2col_step = im2col_step
        output = _C.ms_deform_attn_forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            ctx.im2col_step,
        )
        ctx.save_for_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        (
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
        ) = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = _C.ms_deform_attn_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            grad_output,
            ctx.im2col_step,
        )

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


def bilinear_grid_sample(im, grid, align_corners=False):
    """Given an input and a flow-field grid, computes the output using input
    values and pixel locations from grid. Supported only bilinear interpolation
    method to sample the input pixels.

    Args:
        im (torch.Tensor): Input feature map, shape (N, C, H, W)
        grid (torch.Tensor): Point coordinates, shape (N, Hg, Wg, 2)
        align_corners (bool): If set to True, the extrema (-1 and 1) are
            considered as referring to the center points of the input's
            corner pixels. If set to False, they are instead considered as
            referring to the corner points of the input's corner pixels,
            making the sampling more resolution agnostic.

    Returns:
        torch.Tensor: A tensor with sampled points, shape (N, C, Hg, Wg)
    """
    n, c, h, w = im.shape
    gn, gh, gw, _ = grid.shape
    assert n == gn

    x = grid[:, :, :, 0]
    y = grid[:, :, :, 1]

    if align_corners:
        x = ((x + 1) / 2) * (w - 1)
        y = ((y + 1) / 2) * (h - 1)
    else:
        x = ((x + 1) * w - 1) / 2
        y = ((y + 1) * h - 1) / 2

    x = x.view(n, -1)
    y = y.view(n, -1)

    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    x1 = x0 + 1
    y1 = y0 + 1

    wa = ((x1 - x) * (y1 - y)).unsqueeze(1)
    wb = ((x1 - x) * (y - y0)).unsqueeze(1)
    wc = ((x - x0) * (y1 - y)).unsqueeze(1)
    wd = ((x - x0) * (y - y0)).unsqueeze(1)

    # Apply default for grid_sample function zero padding
    im_padded = F.pad(im, pad=[1, 1, 1, 1], mode='constant', value=0)
    padded_h = h + 2
    padded_w = w + 2
    # save points positions after padding
    x0, x1, y0, y1 = x0 + 1, x1 + 1, y0 + 1, y1 + 1

    # Clip coordinates to padded image size
    x0 = torch.clamp_(x0, 0, padded_w - 1)
    x1 = torch.clamp_(x1, 0, padded_w - 1)
    y0 = torch.clamp_(y0, 0, padded_h - 1)
    y1 = torch.clamp_(y1, 0, padded_h - 1)

    im_padded = im_padded.view(n, c, -1)

    x0_y0 = (x0 + y0 * padded_w).unsqueeze(1).expand(-1, c, -1)
    x0_y1 = (x0 + y1 * padded_w).unsqueeze(1).expand(-1, c, -1)
    x1_y0 = (x1 + y0 * padded_w).unsqueeze(1).expand(-1, c, -1)
    x1_y1 = (x1 + y1 * padded_w).unsqueeze(1).expand(-1, c, -1)

    Ia = torch.gather(im_padded, 2, x0_y0)
    Ib = torch.gather(im_padded, 2, x0_y1)
    Ic = torch.gather(im_padded, 2, x1_y0)
    Id = torch.gather(im_padded, 2, x1_y1)

    return (Ia * wa + Ib * wb + Ic * wc + Id * wd).reshape(n, c, gh, gw)


def multi_scale_deformable_attn_pytorch(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split(value_spatial_shapes.prod(-1).unbind(0), dim=1)
    sampling_grids = 2 * sampling_locations - 1
    if torchvision._is_tracing():
        # avoid iteration warning on torch.Tensor
        # convert Tensor to list[Tensor] instead
        value_spatial_shapes = [b.unbind(0) for b in value_spatial_shapes.unbind(0)]
    else:
        # use list to avoid small kernel launching when indexing spatial shapes
        value_spatial_shapes = value_spatial_shapes.tolist()
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        # bs, H_*W_, num_heads, embed_dims ->
        # bs, H_*W_, num_heads*embed_dims ->
        # bs, num_heads*embed_dims, H_*W_ ->
        # bs*num_heads, embed_dims, H_, W_
        value_l_ = (value_list[level].flatten(2).transpose(1, 2).reshape(bs * num_heads, embed_dims, H_, W_))
        # bs, num_queries, num_heads, num_points, 2 ->
        # bs, num_heads, num_queries, num_points, 2 ->
        # bs*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        # bs*num_heads, embed_dims, num_queries, num_points
        if torchvision._is_tracing():
            sampling_value_l_ = bilinear_grid_sample(
                value_l_,
                sampling_grid_l_.contiguous(),
                align_corners=False,
            )
        else:
            sampling_value_l_ = F.grid_sample(
                value_l_,
                sampling_grid_l_,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
        sampling_value_list.append(sampling_value_l_)
    # (bs, num_queries, num_heads, num_levels, num_points) ->
    # (bs, num_heads, num_queries, num_levels, num_points) ->
    # (bs, num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points
    )
    output = torch.stack(sampling_value_list, dim=-2).flatten(-2)
    output = (output * attention_weights).sum(-1)
    output = output.view(bs, num_heads * embed_dims, num_queries)
    return output.transpose(1, 2).contiguous()


class MultiScaleDeformableAttention(nn.Module):
    """Multi-Scale Deformable Attention Module used in Deformable-DETR

    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.
    """
    def __init__(
        self,
        embed_dim: int = 256,
        num_levels: int = 4,
        num_heads: int = 8,
        num_points: int = 4,
        img2col_step: int = 64,
    ):
        """Initialization function of MultiScaleDeformableAttention

        :param embed_dim: The embedding dimension of Attention, defaults to 256
        :param num_levels: The number of feature map used in Attention, defaults to 4
        :param num_heads: The number of attention heads, defaults to 8
        :param num_points: The number of sampling points for each query
            in each head, defaults to 4
        :param img2col_step: The step used in image_to_column, defaults to 64
        """
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                "embed_dim must be divisible by num_heads, but got {} and {}".format(embed_dim, num_heads)
            )
        head_dim = embed_dim // num_heads

        if not _is_power_of_2(head_dim):
            warnings.warn(
                """
                You'd better set embed_dim in MSDeformAttn to make sure that
                each dim of the attention head a power of 2, which is more efficient.
                """
            )

        self.im2col_step = img2col_step
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        # num_heads * num_points and num_levels for multi-level feature inputs
        self.sampling_offsets = nn.Linear(embed_dim, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dim, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self.init_weights()

    def init_weights(self):
        """Default initialization for parameters of the module"""
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32)
        thetas = thetas * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True)[0]
        grid_init = grid_init.view(self.num_heads, 1, 1, 2)
        grid_init = grid_init.repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def forward(
        self,
        query: Tensor,
        reference_points: Tensor,
        value: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        key_padding_mask: Tensor,
    ) -> Tensor:
        """Forward function of MultiScaleDeformableAttention

        :param query: query embeddings with shape (batch_size, num_query, embed_dim)
        :param reference_points: the normalized reference points with shape
            (batch_size, num_query, num_levels, 2), all_elements is range in [0, 1],
            top-left (0, 0), bottom-right (1, 1), including padding area. or
            (batch_size, num_query, num_levels, 4), add additional two dimensions (h, w)
            to form reference boxes
        :param value: value embeddings with shape (batch_size, num_value, embed_dim)
        :param spatial_shapes: spatial shapes of features in different levels.
            with shape (num_levels, 2), last dimension represents (h, w)
        :param level_start_index: the start index of each level. A tensor with shape
            (num_levels,), which can be represented as [0, h_0 * w_0, h_0 * w_0 + h_1 * w_1, ...]
        :param key_padding_mask: ByteTensor for query, with shape (batch_size, num_value)
        :return: forward results with shape (batch_size, num_query, embed_dim)
        """
        batch_size, num_query, _ = query.shape
        batch_size, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        # value projection
        value = self.value_proj(value)
        # fill "0" for the padding part
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], float(0))

        value = value.view(batch_size, num_value, self.num_heads, self.embed_dim // self.num_heads)
        sampling_offsets = self.sampling_offsets(query).view(
            batch_size, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )
        # total num_levels * num_points features
        attention_weights = self.attention_weights(query).view(
            batch_size, num_query, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = attention_weights.softmax(-1)
        attention_weights = attention_weights.view(
            batch_size,
            num_query,
            self.num_heads,
            self.num_levels,
            self.num_points,
        )

        # batch_size, num_query, num_heads, num_levels, num_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :] +
                sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2] +
                sampling_offsets / self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".format(
                    reference_points.shape[-1]
                )
            )

        # the original impl for fp32 training
        if _C is not None and value.is_cuda:
            output = MultiScaleDeformableAttnFunction.apply(
                value.to(torch.float32),
                spatial_shapes,
                level_start_index,
                sampling_locations,
                attention_weights,
                self.im2col_step,
            )
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights
            )

        if value.dtype != torch.float32:
            output = output.to(value.dtype)

        output = self.output_proj(output)

        return output
