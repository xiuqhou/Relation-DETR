from __future__ import annotations

from enum import Enum
from types import ModuleType
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple, Type, Union

import PIL.Image
import torch
from torch._C import DisableTorchFunction

from transforms import InterpolationMode


class Datapoint(torch.Tensor):
    __F: Optional[ModuleType] = None

    @staticmethod
    def _to_tensor(
        data: Any,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str, int]] = None,
        requires_grad: Optional[bool] = None,
    ) -> torch.Tensor:
        if requires_grad is None:
            requires_grad = data.requires_grad if isinstance(data, torch.Tensor) else False
        return torch.as_tensor(data, dtype=dtype, device=device).requires_grad_(requires_grad)

    @classmethod
    def wrap_like(cls: Type[D], other: D, tensor: torch.Tensor) -> D:
        raise NotImplementedError

    _NO_WRAPPING_EXCEPTIONS = {
        torch.Tensor.clone: lambda cls, input, output: cls.wrap_like(input, output),
        torch.Tensor.to: lambda cls, input, output: cls.wrap_like(input, output),
        # We don't need to wrap the output of `Tensor.requires_grad_`, since it is an inplace operation and thus
        # retains the type automatically
        torch.Tensor.requires_grad_: lambda cls, input, output: output,
    }

    @classmethod
    def __torch_function__(
        cls,
        func: Callable[..., torch.Tensor],
        types: Tuple[Type[torch.Tensor], ...],
        args: Sequence[Any] = (),
        kwargs: Optional[Mapping[str, Any]] = None,
    ) -> torch.Tensor:
        """For general information about how the __torch_function__ protocol works,
        see https://pytorch.org/docs/stable/notes/extending.html#extending-torch

        TL;DR: Every time a PyTorch operator is called, it goes through the inputs and looks for the
        ``__torch_function__`` method. If one is found, it is invoked with the operator as ``func`` as well as the
        ``args`` and ``kwargs`` of the original call.

        The default behavior of :class:`~torch.Tensor`'s is to retain a custom tensor type. For the :class:`Datapoint`
        use case, this has two downsides:

        1. Since some :class:`Datapoint`'s require metadata to be constructed, the default wrapping, i.e.
           ``return cls(func(*args, **kwargs))``, will fail for them.
        2. For most operations, there is no way of knowing if the input type is still valid for the output.

        For these reasons, the automatic output wrapping is turned off for most operators. The only exceptions are
        listed in :attr:`Datapoint._NO_WRAPPING_EXCEPTIONS`
        """
        # Since super().__torch_function__ has no hook to prevent the coercing of the output into the input type, we
        # need to reimplement the functionality.

        if not all(issubclass(cls, t) for t in types):
            return NotImplemented

        with DisableTorchFunction():
            output = func(*args, **kwargs or dict())

            wrapper = cls._NO_WRAPPING_EXCEPTIONS.get(func)
            # Apart from `func` needing to be an exception, we also require the primary operand, i.e. `args[0]`, to be
            # an instance of the class that `__torch_function__` was invoked on. The __torch_function__ protocol will
            # invoke this method on *all* types involved in the computation by walking the MRO upwards. For example,
            # `torch.Tensor(...).to(datapoints.Image(...))` will invoke `datapoints.Image.__torch_function__` with
            # `args = (torch.Tensor(), datapoints.Image())` first. Without this guard, the original `torch.Tensor` would
            # be wrapped into a `datapoints.Image`.
            if wrapper and isinstance(args[0], cls):
                return wrapper(cls, args[0], output)

            # Inplace `func`'s, canonically identified with a trailing underscore in their name like `.add_(...)`,
            # will retain the input type. Thus, we need to unwrap here.
            if isinstance(output, cls):
                return output.as_subclass(torch.Tensor)

            return output

    def _make_repr(self, **kwargs: Any) -> str:
        # This is a poor man's implementation of the proposal in https://github.com/pytorch/pytorch/issues/76532.
        # If that ever gets implemented, remove this in favor of the solution on the `torch.Tensor` class.
        extra_repr = ", ".join(f"{key}={value}" for key, value in kwargs.items())
        return f"{super().__repr__()[:-1]}, {extra_repr})"

    @property
    def _F(self) -> ModuleType:
        # This implements a lazy import of the functional to get around the cyclic import. This import is deferred
        # until the first time we need reference to the functional module and it's shared across all instances of
        # the class. This approach avoids the DataLoader issue described at
        # https://github.com/pytorch/vision/pull/6476#discussion_r953588621
        if Datapoint.__F is None:
            from transforms.v2 import functional

            Datapoint.__F = functional
        return Datapoint.__F

    @property
    def data(self) -> torch.Tensor:
        return self.as_subclass(torch.Tensor)

    def horizontal_flip(self) -> Datapoint:
        return self

    def vertical_flip(self) -> Datapoint:
        return self

    # TODO: We have to ignore override mypy error as there is torch.Tensor built-in deprecated op: Tensor.resize
    # https://github.com/pytorch/pytorch/blob/e8727994eb7cdb2ab642749d6549bc497563aa06/torch/_tensor.py#L588-L593
    def resize(  # type: ignore[override]
        self,
        size: List[int],
        interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
        max_size: Optional[int] = None,
        antialias: Optional[Union[str, bool]] = "warn",
    ) -> Datapoint:
        return self

    def crop(self, top: int, left: int, height: int, width: int) -> Datapoint:
        return self

    def center_crop(self, output_size: List[int]) -> Datapoint:
        return self

    def resized_crop(
        self,
        top: int,
        left: int,
        height: int,
        width: int,
        size: List[int],
        interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
        antialias: Optional[Union[str, bool]] = "warn",
    ) -> Datapoint:
        return self

    def pad(
        self,
        padding: List[int],
        fill: Optional[Union[int, float, List[float]]] = None,
        padding_mode: str = "constant",
    ) -> Datapoint:
        return self

    def rotate(
        self,
        angle: float,
        interpolation: Union[InterpolationMode, int] = InterpolationMode.NEAREST,
        expand: bool = False,
        center: Optional[List[float]] = None,
        fill: Optional[List[float]] = None,
    ) -> Datapoint:
        return self

    def affine(
        self,
        angle: Union[int, float],
        translate: List[float],
        scale: float,
        shear: List[float],
        interpolation: Union[InterpolationMode, int] = InterpolationMode.NEAREST,
        fill: Optional[List[float]] = None,
        center: Optional[List[float]] = None,
    ) -> Datapoint:
        return self

    def perspective(
        self,
        startpoints: Optional[List[List[int]]],
        endpoints: Optional[List[List[int]]],
        interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
        fill: Optional[List[float]] = None,
        coefficients: Optional[List[float]] = None,
    ) -> Datapoint:
        return self

    def elastic(
        self,
        displacement: torch.Tensor,
        interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
        fill: Optional[List[float]] = None,
    ) -> Datapoint:
        return self

    def rgb_to_grayscale(self, num_output_channels: int = 1) -> Datapoint:
        return self

    def adjust_brightness(self, brightness_factor: float) -> Datapoint:
        return self

    def adjust_saturation(self, saturation_factor: float) -> Datapoint:
        return self

    def adjust_contrast(self, contrast_factor: float) -> Datapoint:
        return self

    def adjust_sharpness(self, sharpness_factor: float) -> Datapoint:
        return self

    def adjust_hue(self, hue_factor: float) -> Datapoint:
        return self

    def adjust_gamma(self, gamma: float, gain: float = 1) -> Datapoint:
        return self

    def posterize(self, bits: int) -> Datapoint:
        return self

    def solarize(self, threshold: float) -> Datapoint:
        return self

    def autocontrast(self) -> Datapoint:
        return self

    def equalize(self) -> Datapoint:
        return self

    def invert(self) -> Datapoint:
        return self

    def gaussian_blur(
        self, kernel_size: List[int], sigma: Optional[List[float]] = None
    ) -> Datapoint:
        return self


class Image(Datapoint):
    @classmethod
    def _wrap(cls, tensor: torch.Tensor) -> Image:
        image = tensor.as_subclass(cls)
        return image

    def __new__(
        cls,
        data: Any,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str, int]] = None,
        requires_grad: Optional[bool] = None,
    ) -> Image:
        if isinstance(data, PIL.Image.Image):
            from transforms import functional as F

            data = F.pil_to_tensor(data)

        tensor = cls._to_tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)
        if tensor.ndim < 2:
            raise ValueError
        elif tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)

        return cls._wrap(tensor)

    @classmethod
    def wrap_like(cls, other: Image, tensor: torch.Tensor) -> Image:
        return cls._wrap(tensor)

    def __repr__(self, *, tensor_contents: Any = None) -> str:  # type: ignore[override]
        return self._make_repr()

    @property
    def spatial_size(self) -> Tuple[int, int]:
        return tuple(self.shape[-2:])  # type: ignore[return-value]

    @property
    def num_channels(self) -> int:
        return self.shape[-3]

    def horizontal_flip(self) -> Image:
        output = self._F.horizontal_flip_image_tensor(self.as_subclass(torch.Tensor))
        return Image.wrap_like(self, output)

    def vertical_flip(self) -> Image:
        output = self._F.vertical_flip_image_tensor(self.as_subclass(torch.Tensor))
        return Image.wrap_like(self, output)

    def resize(  # type: ignore[override]
        self,
        size: List[int],
        interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
        max_size: Optional[int] = None,
        antialias: Optional[Union[str, bool]] = "warn",
    ) -> Image:
        output = self._F.resize_image_tensor(
            self.as_subclass(torch.Tensor),
            size,
            interpolation=interpolation,
            max_size=max_size,
            antialias=antialias,
        )
        return Image.wrap_like(self, output)

    def crop(self, top: int, left: int, height: int, width: int) -> Image:
        output = self._F.crop_image_tensor(self.as_subclass(torch.Tensor), top, left, height, width)
        return Image.wrap_like(self, output)

    def center_crop(self, output_size: List[int]) -> Image:
        output = self._F.center_crop_image_tensor(
            self.as_subclass(torch.Tensor), output_size=output_size
        )
        return Image.wrap_like(self, output)

    def resized_crop(
        self,
        top: int,
        left: int,
        height: int,
        width: int,
        size: List[int],
        interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
        antialias: Optional[Union[str, bool]] = "warn",
    ) -> Image:
        output = self._F.resized_crop_image_tensor(
            self.as_subclass(torch.Tensor),
            top,
            left,
            height,
            width,
            size=list(size),
            interpolation=interpolation,
            antialias=antialias,
        )
        return Image.wrap_like(self, output)

    def pad(
        self,
        padding: List[int],
        fill: Optional[Union[int, float, List[float]]] = None,
        padding_mode: str = "constant",
    ) -> Image:
        output = self._F.pad_image_tensor(
            self.as_subclass(torch.Tensor),
            padding,
            fill=fill,
            padding_mode=padding_mode,
        )
        return Image.wrap_like(self, output)

    def rotate(
        self,
        angle: float,
        interpolation: Union[InterpolationMode, int] = InterpolationMode.NEAREST,
        expand: bool = False,
        center: Optional[List[float]] = None,
        fill: Optional[List[float]] = None,
    ) -> Image:
        output = self._F.rotate_image_tensor(
            self.as_subclass(torch.Tensor),
            angle,
            interpolation=interpolation,
            expand=expand,
            fill=fill,
            center=center,
        )
        return Image.wrap_like(self, output)

    def affine(
        self,
        angle: Union[int, float],
        translate: List[float],
        scale: float,
        shear: List[float],
        interpolation: Union[InterpolationMode, int] = InterpolationMode.NEAREST,
        fill: Optional[List[float]] = None,
        center: Optional[List[float]] = None,
    ) -> Image:
        output = self._F.affine_image_tensor(
            self.as_subclass(torch.Tensor),
            angle,
            translate=translate,
            scale=scale,
            shear=shear,
            interpolation=interpolation,
            fill=fill,
            center=center,
        )
        return Image.wrap_like(self, output)

    def perspective(
        self,
        startpoints: Optional[List[List[int]]],
        endpoints: Optional[List[List[int]]],
        interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
        fill: Optional[List[float]] = None,
        coefficients: Optional[List[float]] = None,
    ) -> Image:
        output = self._F.perspective_image_tensor(
            self.as_subclass(torch.Tensor),
            startpoints,
            endpoints,
            interpolation=interpolation,
            fill=fill,
            coefficients=coefficients,
        )
        return Image.wrap_like(self, output)

    def elastic(
        self,
        displacement: torch.Tensor,
        interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
        fill: Optional[List[float]] = None,
    ) -> Image:
        output = self._F.elastic_image_tensor(
            self.as_subclass(torch.Tensor),
            displacement,
            interpolation=interpolation,
            fill=fill,
        )
        return Image.wrap_like(self, output)

    def rgb_to_grayscale(self, num_output_channels: int = 1) -> Image:
        output = self._F.rgb_to_grayscale_image_tensor(
            self.as_subclass(torch.Tensor), num_output_channels=num_output_channels
        )
        return Image.wrap_like(self, output)

    def adjust_brightness(self, brightness_factor: float) -> Image:
        output = self._F.adjust_brightness_image_tensor(
            self.as_subclass(torch.Tensor), brightness_factor=brightness_factor
        )
        return Image.wrap_like(self, output)

    def adjust_saturation(self, saturation_factor: float) -> Image:
        output = self._F.adjust_saturation_image_tensor(
            self.as_subclass(torch.Tensor), saturation_factor=saturation_factor
        )
        return Image.wrap_like(self, output)

    def adjust_contrast(self, contrast_factor: float) -> Image:
        output = self._F.adjust_contrast_image_tensor(
            self.as_subclass(torch.Tensor), contrast_factor=contrast_factor
        )
        return Image.wrap_like(self, output)

    def adjust_sharpness(self, sharpness_factor: float) -> Image:
        output = self._F.adjust_sharpness_image_tensor(
            self.as_subclass(torch.Tensor), sharpness_factor=sharpness_factor
        )
        return Image.wrap_like(self, output)

    def adjust_hue(self, hue_factor: float) -> Image:
        output = self._F.adjust_hue_image_tensor(
            self.as_subclass(torch.Tensor), hue_factor=hue_factor
        )
        return Image.wrap_like(self, output)

    def adjust_gamma(self, gamma: float, gain: float = 1) -> Image:
        output = self._F.adjust_gamma_image_tensor(
            self.as_subclass(torch.Tensor), gamma=gamma, gain=gain
        )
        return Image.wrap_like(self, output)

    def posterize(self, bits: int) -> Image:
        output = self._F.posterize_image_tensor(self.as_subclass(torch.Tensor), bits=bits)
        return Image.wrap_like(self, output)

    def solarize(self, threshold: float) -> Image:
        output = self._F.solarize_image_tensor(self.as_subclass(torch.Tensor), threshold=threshold)
        return Image.wrap_like(self, output)

    def autocontrast(self) -> Image:
        output = self._F.autocontrast_image_tensor(self.as_subclass(torch.Tensor))
        return Image.wrap_like(self, output)

    def equalize(self) -> Image:
        output = self._F.equalize_image_tensor(self.as_subclass(torch.Tensor))
        return Image.wrap_like(self, output)

    def invert(self) -> Image:
        output = self._F.invert_image_tensor(self.as_subclass(torch.Tensor))
        return Image.wrap_like(self, output)

    def gaussian_blur(self, kernel_size: List[int], sigma: Optional[List[float]] = None) -> Image:
        output = self._F.gaussian_blur_image_tensor(
            self.as_subclass(torch.Tensor), kernel_size=kernel_size, sigma=sigma
        )
        return Image.wrap_like(self, output)

    def normalize(self, mean: List[float], std: List[float], inplace: bool = False) -> Image:
        output = self._F.normalize_image_tensor(
            self.as_subclass(torch.Tensor), mean=mean, std=std, inplace=inplace
        )
        return Image.wrap_like(self, output)


class BoundingBoxFormat(Enum):
    XYXY = "XYXY"
    XYWH = "XYWH"
    CXCYWH = "CXCYWH"


class BoundingBox(Datapoint):
    format: BoundingBoxFormat
    spatial_size: Tuple[int, int]

    @classmethod
    def _wrap(
        cls,
        tensor: torch.Tensor,
        *,
        format: BoundingBoxFormat,
        spatial_size: Tuple[int, int],
    ) -> BoundingBox:
        bounding_box = tensor.as_subclass(cls)
        bounding_box.format = format
        bounding_box.spatial_size = spatial_size
        return bounding_box

    def __new__(
        cls,
        data: Any,
        *,
        format: Union[BoundingBoxFormat, str],
        spatial_size: Tuple[int, int],
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str, int]] = None,
        requires_grad: Optional[bool] = None,
    ) -> BoundingBox:
        tensor = cls._to_tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)

        if isinstance(format, str):
            format = BoundingBoxFormat[format.upper()]

        return cls._wrap(tensor, format=format, spatial_size=spatial_size)

    @classmethod
    def wrap_like(
        cls,
        other: BoundingBox,
        tensor: torch.Tensor,
        *,
        format: Optional[BoundingBoxFormat] = None,
        spatial_size: Optional[Tuple[int, int]] = None,
    ) -> BoundingBox:
        """Wrap a :class:`torch.Tensor` as :class:`BoundingBox` from a reference.

        Args:
            other (BoundingBox): Reference bounding box.
            tensor (Tensor): Tensor to be wrapped as :class:`BoundingBox`
            format (BoundingBoxFormat, str, optional): Format of the bounding box.  If omitted, it is taken from the
                reference.
            spatial_size (two-tuple of ints, optional): Height and width of the corresponding image or video. If
                omitted, it is taken from the reference.

        """
        if isinstance(format, str):
            format = BoundingBoxFormat[format.upper()]

        return cls._wrap(
            tensor,
            format=format if format is not None else other.format,
            spatial_size=spatial_size if spatial_size is not None else other.spatial_size,
        )

    def __repr__(self, *, tensor_contents: Any = None) -> str:  # type: ignore[override]
        return self._make_repr(format=self.format, spatial_size=self.spatial_size)

    def horizontal_flip(self) -> BoundingBox:
        output = self._F.horizontal_flip_bounding_box(
            self.as_subclass(torch.Tensor),
            format=self.format,
            spatial_size=self.spatial_size,
        )
        return BoundingBox.wrap_like(self, output)

    def vertical_flip(self) -> BoundingBox:
        output = self._F.vertical_flip_bounding_box(
            self.as_subclass(torch.Tensor),
            format=self.format,
            spatial_size=self.spatial_size,
        )
        return BoundingBox.wrap_like(self, output)

    def resize(  # type: ignore[override]
        self,
        size: List[int],
        interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
        max_size: Optional[int] = None,
        antialias: Optional[Union[str, bool]] = "warn",
    ) -> BoundingBox:
        output, spatial_size = self._F.resize_bounding_box(
            self.as_subclass(torch.Tensor),
            spatial_size=self.spatial_size,
            size=size,
            max_size=max_size,
        )
        return BoundingBox.wrap_like(self, output, spatial_size=spatial_size)

    def crop(self, top: int, left: int, height: int, width: int) -> BoundingBox:
        output, spatial_size = self._F.crop_bounding_box(
            self.as_subclass(torch.Tensor),
            self.format,
            top=top,
            left=left,
            height=height,
            width=width,
        )
        return BoundingBox.wrap_like(self, output, spatial_size=spatial_size)

    def center_crop(self, output_size: List[int]) -> BoundingBox:
        output, spatial_size = self._F.center_crop_bounding_box(
            self.as_subclass(torch.Tensor),
            format=self.format,
            spatial_size=self.spatial_size,
            output_size=output_size,
        )
        return BoundingBox.wrap_like(self, output, spatial_size=spatial_size)

    def resized_crop(
        self,
        top: int,
        left: int,
        height: int,
        width: int,
        size: List[int],
        interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
        antialias: Optional[Union[str, bool]] = "warn",
    ) -> BoundingBox:
        output, spatial_size = self._F.resized_crop_bounding_box(
            self.as_subclass(torch.Tensor),
            self.format,
            top,
            left,
            height,
            width,
            size=size,
        )
        return BoundingBox.wrap_like(self, output, spatial_size=spatial_size)

    def pad(
        self,
        padding: Union[int, Sequence[int]],
        fill: Optional[Union[int, float, List[float]]] = None,
        padding_mode: str = "constant",
    ) -> BoundingBox:
        output, spatial_size = self._F.pad_bounding_box(
            self.as_subclass(torch.Tensor),
            format=self.format,
            spatial_size=self.spatial_size,
            padding=padding,
            padding_mode=padding_mode,
        )
        return BoundingBox.wrap_like(self, output, spatial_size=spatial_size)

    def rotate(
        self,
        angle: float,
        interpolation: Union[InterpolationMode, int] = InterpolationMode.NEAREST,
        expand: bool = False,
        center: Optional[List[float]] = None,
        fill: Optional[List[float]] = None,
    ) -> BoundingBox:
        output, spatial_size = self._F.rotate_bounding_box(
            self.as_subclass(torch.Tensor),
            format=self.format,
            spatial_size=self.spatial_size,
            angle=angle,
            expand=expand,
            center=center,
        )
        return BoundingBox.wrap_like(self, output, spatial_size=spatial_size)

    def affine(
        self,
        angle: Union[int, float],
        translate: List[float],
        scale: float,
        shear: List[float],
        interpolation: Union[InterpolationMode, int] = InterpolationMode.NEAREST,
        fill: Optional[List[float]] = None,
        center: Optional[List[float]] = None,
    ) -> BoundingBox:
        output = self._F.affine_bounding_box(
            self.as_subclass(torch.Tensor),
            self.format,
            self.spatial_size,
            angle,
            translate=translate,
            scale=scale,
            shear=shear,
            center=center,
        )
        return BoundingBox.wrap_like(self, output)

    def perspective(
        self,
        startpoints: Optional[List[List[int]]],
        endpoints: Optional[List[List[int]]],
        interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
        fill: Optional[List[float]] = None,
        coefficients: Optional[List[float]] = None,
    ) -> BoundingBox:
        output = self._F.perspective_bounding_box(
            self.as_subclass(torch.Tensor),
            format=self.format,
            spatial_size=self.spatial_size,
            startpoints=startpoints,
            endpoints=endpoints,
            coefficients=coefficients,
        )
        return BoundingBox.wrap_like(self, output)

    def elastic(
        self,
        displacement: torch.Tensor,
        interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
        fill: Optional[List[float]] = None,
    ) -> BoundingBox:
        output = self._F.elastic_bounding_box(
            self.as_subclass(torch.Tensor),
            self.format,
            self.spatial_size,
            displacement=displacement,
        )
        return BoundingBox.wrap_like(self, output)


class Mask(Datapoint):
    """[BETA] :class:`torch.Tensor` subclass for segmentation and detection masks.

    Args:
        data (tensor-like, PIL.Image.Image): Any data that can be turned into a tensor with :func:`torch.as_tensor` as
            well as PIL images.
        dtype (torch.dtype, optional): Desired data type of the bounding box. If omitted, will be inferred from
            ``data``.
        device (torch.device, optional): Desired device of the bounding box. If omitted and ``data`` is a
            :class:`torch.Tensor`, the device is taken from it. Otherwise, the bounding box is constructed on the CPU.
        requires_grad (bool, optional): Whether autograd should record operations on the bounding box. If omitted and
            ``data`` is a :class:`torch.Tensor`, the value is taken from it. Otherwise, defaults to ``False``.
    """

    @classmethod
    def _wrap(cls, tensor: torch.Tensor) -> Mask:
        return tensor.as_subclass(cls)

    def __new__(
        cls,
        data: Any,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str, int]] = None,
        requires_grad: Optional[bool] = None,
    ) -> Mask:
        if isinstance(data, PIL.Image.Image):
            from transforms.v2 import functional as F

            data = F.pil_to_tensor(data)

        tensor = cls._to_tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)
        return cls._wrap(tensor)

    @classmethod
    def wrap_like(
        cls,
        other: Mask,
        tensor: torch.Tensor,
    ) -> Mask:
        return cls._wrap(tensor)

    @property
    def spatial_size(self) -> Tuple[int, int]:
        return tuple(self.shape[-2:])  # type: ignore[return-value]

    def horizontal_flip(self) -> Mask:
        output = self._F.horizontal_flip_mask(self.as_subclass(torch.Tensor))
        return Mask.wrap_like(self, output)

    def vertical_flip(self) -> Mask:
        output = self._F.vertical_flip_mask(self.as_subclass(torch.Tensor))
        return Mask.wrap_like(self, output)

    def resize(  # type: ignore[override]
        self,
        size: List[int],
        interpolation: Union[InterpolationMode, int] = InterpolationMode.NEAREST,
        max_size: Optional[int] = None,
        antialias: Optional[Union[str, bool]] = "warn",
    ) -> Mask:
        output = self._F.resize_mask(self.as_subclass(torch.Tensor), size, max_size=max_size)
        return Mask.wrap_like(self, output)

    def crop(self, top: int, left: int, height: int, width: int) -> Mask:
        output = self._F.crop_mask(self.as_subclass(torch.Tensor), top, left, height, width)
        return Mask.wrap_like(self, output)

    def center_crop(self, output_size: List[int]) -> Mask:
        output = self._F.center_crop_mask(self.as_subclass(torch.Tensor), output_size=output_size)
        return Mask.wrap_like(self, output)

    def resized_crop(
        self,
        top: int,
        left: int,
        height: int,
        width: int,
        size: List[int],
        interpolation: Union[InterpolationMode, int] = InterpolationMode.NEAREST,
        antialias: Optional[Union[str, bool]] = "warn",
    ) -> Mask:
        output = self._F.resized_crop_mask(
            self.as_subclass(torch.Tensor), top, left, height, width, size=size
        )
        return Mask.wrap_like(self, output)

    def pad(
        self,
        padding: List[int],
        fill: Optional[Union[int, float, List[float]]] = None,
        padding_mode: str = "constant",
    ) -> Mask:
        output = self._F.pad_mask(
            self.as_subclass(torch.Tensor),
            padding,
            padding_mode=padding_mode,
            fill=fill,
        )
        return Mask.wrap_like(self, output)

    def rotate(
        self,
        angle: float,
        interpolation: Union[InterpolationMode, int] = InterpolationMode.NEAREST,
        expand: bool = False,
        center: Optional[List[float]] = None,
        fill: Optional[List[float]] = None,
    ) -> Mask:
        output = self._F.rotate_mask(
            self.as_subclass(torch.Tensor),
            angle,
            expand=expand,
            center=center,
            fill=fill,
        )
        return Mask.wrap_like(self, output)

    def affine(
        self,
        angle: Union[int, float],
        translate: List[float],
        scale: float,
        shear: List[float],
        interpolation: Union[InterpolationMode, int] = InterpolationMode.NEAREST,
        fill: Optional[List[float]] = None,
        center: Optional[List[float]] = None,
    ) -> Mask:
        output = self._F.affine_mask(
            self.as_subclass(torch.Tensor),
            angle,
            translate=translate,
            scale=scale,
            shear=shear,
            fill=fill,
            center=center,
        )
        return Mask.wrap_like(self, output)

    def perspective(
        self,
        startpoints: Optional[List[List[int]]],
        endpoints: Optional[List[List[int]]],
        interpolation: Union[InterpolationMode, int] = InterpolationMode.NEAREST,
        fill: Optional[List[float]] = None,
        coefficients: Optional[List[float]] = None,
    ) -> Mask:
        output = self._F.perspective_mask(
            self.as_subclass(torch.Tensor),
            startpoints,
            endpoints,
            fill=fill,
            coefficients=coefficients,
        )
        return Mask.wrap_like(self, output)

    def elastic(
        self,
        displacement: torch.Tensor,
        interpolation: Union[InterpolationMode, int] = InterpolationMode.NEAREST,
        fill: Optional[List[float]] = None,
    ) -> Mask:
        output = self._F.elastic_mask(self.as_subclass(torch.Tensor), displacement, fill=fill)
        return Mask.wrap_like(self, output)


class Video(Datapoint):
    """[BETA] :class:`torch.Tensor` subclass for videos.

    Args:
        data (tensor-like): Any data that can be turned into a tensor with :func:`torch.as_tensor`.
        dtype (torch.dtype, optional): Desired data type of the bounding box. If omitted, will be inferred from
            ``data``.
        device (torch.device, optional): Desired device of the bounding box. If omitted and ``data`` is a
            :class:`torch.Tensor`, the device is taken from it. Otherwise, the bounding box is constructed on the CPU.
        requires_grad (bool, optional): Whether autograd should record operations on the bounding box. If omitted and
            ``data`` is a :class:`torch.Tensor`, the value is taken from it. Otherwise, defaults to ``False``.
    """

    @classmethod
    def _wrap(cls, tensor: torch.Tensor) -> Video:
        video = tensor.as_subclass(cls)
        return video

    def __new__(
        cls,
        data: Any,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str, int]] = None,
        requires_grad: Optional[bool] = None,
    ) -> Video:
        tensor = cls._to_tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)
        if data.ndim < 4:
            raise ValueError
        return cls._wrap(tensor)

    @classmethod
    def wrap_like(cls, other: Video, tensor: torch.Tensor) -> Video:
        return cls._wrap(tensor)

    def __repr__(self, *, tensor_contents: Any = None) -> str:  # type: ignore[override]
        return self._make_repr()

    @property
    def spatial_size(self) -> Tuple[int, int]:
        return tuple(self.shape[-2:])  # type: ignore[return-value]

    @property
    def num_channels(self) -> int:
        return self.shape[-3]

    @property
    def num_frames(self) -> int:
        return self.shape[-4]


_ImageType = Union[torch.Tensor, PIL.Image.Image, Image]
_ImageTypeJIT = torch.Tensor
_TensorImageType = Union[torch.Tensor, Image]
_TensorImageTypeJIT = torch.Tensor
_InputType = Union[torch.Tensor, PIL.Image.Image, Datapoint]
_InputTypeJIT = torch.Tensor
_VideoType = Union[torch.Tensor, Video]
_VideoTypeJIT = torch.Tensor
_TensorVideoType = Union[torch.Tensor, Video]
_TensorVideoTypeJIT = torch.Tensor
_FillType = Union[int, float, Sequence[int], Sequence[float], None]
_FillTypeJIT = Optional[List[float]]
