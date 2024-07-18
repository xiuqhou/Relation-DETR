import copy
import functools
import logging
import math
import os
import random
from datetime import datetime
from functools import partial
from typing import List

import accelerate
import numpy as np
import torch
import torchvision
from accelerate.logging import get_logger
from fvcore.common.file_io import PathManager
from torch import Tensor
from torchvision.models.detection.image_list import ImageList

from util import utils
from util.collect_env import collect_env_info
from util.logger import setup_logger


def replace_prefix(string, prefix, replacement):
    if string.startswith(prefix):
        string = replacement + string[len(prefix):]
    return string


def inverse_sigmoid(x, eps: float = 1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def get_total_grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    device = parameters[0].device
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
        norm_type,
    )
    return total_norm


# _onnx_batch_images() is an implementation of
# batch_images() that is supported by ONNX tracing.
@torch.jit.unused
def _onnx_batch_images(images: List[Tensor], size_divisible: int = 32) -> Tensor:
    max_size = []
    for i in range(images[0].dim()):
        max_size_i = torch.max(torch.stack([img.shape[i]
                                            for img in images]).to(torch.float32)).to(torch.int64)
        max_size.append(max_size_i)
    stride = size_divisible
    max_size[1] = (torch.ceil((max_size[1].to(torch.float32)) / stride) * stride).to(torch.int64)
    max_size[2] = (torch.ceil((max_size[2].to(torch.float32)) / stride) * stride).to(torch.int64)
    max_size = tuple(max_size)

    # work around for
    # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    # which is not yet supported in onnx
    padded_imgs = []
    for img in images:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
        padded_imgs.append(padded_img)

    return torch.stack(padded_imgs)


def image_list_from_tensors(images: List[Tensor], size_divisible=32, fill_value=0):
    if isinstance(images, ImageList):
        return images

    # check channels of images
    batched_channel = images[0].shape[0]
    assert all(
        batched_channel == image.shape[0] for image in images
    ), f"all images must have the same channel but got {list(map(lambda x: x.shape[0], images))}"

    # get original_shapes and batched_shape
    original_shapes = list(map(lambda x: x.shape[-2:], images))

    # get batched shapes, divisible by size_divisible
    if torchvision._is_tracing():
        # batch_images() does not export well to ONNX
        # call _onnx_batch_images() instead
        batched_images = _onnx_batch_images(images, size_divisible)
        return ImageList(batched_images, original_shapes)

    original_h, original_w = list(zip(*original_shapes))
    batched_h, batched_w = max(original_h), max(original_w)
    batched_h = int(math.ceil(float(batched_h) / size_divisible) * size_divisible)
    batched_w = int(math.ceil(float(batched_w) / size_divisible) * size_divisible)

    # generate batched image tensors
    batched_shape = (len(images), batched_channel, batched_h, batched_w)
    batched_images = images[0].new_full(batched_shape, fill_value)
    for idx, image in enumerate(images):
        batched_images[idx, :, :image.shape[1], :image.shape[2]].copy_(image)

    batched_images = ImageList(batched_images, original_shapes)
    return batched_images


def _highlight(code, filename):
    try:
        import pygments
    except ImportError:
        return code

    from pygments.formatters import Terminal256Formatter
    from pygments.lexers import Python3Lexer, YamlLexer

    lexer = Python3Lexer() if filename.endswith(".py") else YamlLexer()
    code = pygments.highlight(code, lexer, Terminal256Formatter(style="monokai"))
    return code


def default_setup(args, cfg, accelerator):
    output_dir = getattr(cfg, "output_dir", None)
    rank = accelerator.local_process_index

    if accelerator.is_main_process and output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # capture warning.warns information into logging
    logging.captureWarnings(True)
    train_log_file = os.path.join(output_dir, "training.log")
    set_logger = partial(setup_logger, output=train_log_file, distributed_rank=rank)
    # setup loggers from warnings, accelerate, detection framworks
    root_logger_name = os.path.basename(os.getcwd())
    list(map(lambda x: set_logger(name=x), ["py.warnings", "accelerate", root_logger_name]))
    logger = get_logger(root_logger_name + "." + __name__)
    logger.info("Rank of current process: {}, World size: {}".format(rank, utils.get_world_size()))
    logger.info("Environment info: \n" + collect_env_info())

    logger.info("Command line arguments: " + str(args))
    if hasattr(args, "config_file") and args.config_file != "":
        logger.info(
            "Contents of args.config_file={}:\n{}".format(
                args.config_file,
                _highlight(PathManager.open(args.config_file, "r").read(), args.config_file),
            )
        )

    # make sure each worker has a different, yet deterministic seed if specified
    if hasattr(args, "seed") and args.seed and args.seed > 0:
        seed = args.seed
    else:
        seed = (
            os.getpid() + int(datetime.now().strftime("%S%f")) +
            int.from_bytes(os.urandom(2), "big")
        )
    logger.info("Using the random seed: {}".format(seed))
    accelerate.utils.set_seed(seed, device_specific=True)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def fixed_generator():
    g = torch.Generator()
    g.manual_seed(0)
    return g


def to_device(inputs, device):
    if isinstance(inputs, (tuple, list)):
        return type(inputs)([to_device(i, device) for i in inputs])
    if isinstance(inputs, dict):
        return type(inputs)({
            to_device(k, device): to_device(v, device)
            for k, v in inputs.items()
        })
    if isinstance(inputs, torch.Tensor):
        return inputs.to(device, non_blocking=True)
    return inputs


def deepcopy(inputs):
    if isinstance(inputs, (tuple, list)):
        return type(inputs)([deepcopy(i) for i in inputs])
    if isinstance(inputs, dict):
        return type(inputs)({deepcopy(k): deepcopy(v) for k, v in inputs.items()})
    if isinstance(inputs, torch.Tensor):
        return inputs.clone().detach()
    return copy.deepcopy(inputs)


@functools.lru_cache
def encode_labels(labels: List[str]):
    """Encode a list of string to a list of int, for example: ["l1", "Label2", "n"]
    will be encoded as: [108,  49,  -1,  76,  97,  98, 101, 108,  50,  -1, 110,  -1].
    Each letter will be converted using ord() function in Python.

    :param labels: A list of str to be encoded.
    :return: A list of int, in which -1 is used as delimiters to split strings.
    """
    assert [isinstance(s, str) for s in labels], "All elements must be strings"
    int_list = []
    for label in labels:
        int_list += [ord(s) for s in label]
        int_list += [-1]
    return int_list


@functools.lru_cache
def decode_labels(ints: List[int]):
    """Decode a list of int to a list of string, for example: [108, 49, -1, 76, 50, -1, 110, -1]
    will be decoded as: ["l1", "L2", "n"]. Each number will be converted to a letter using chr()
    function in Python, and -1 is used as delimiters to split strings.

    :param ints: A list of int to be converted.
    :return: A list of string.
    """
    string_list = []
    string = ""
    for number in ints:
        if number != -1:
            string += chr(number)
        else:
            string_list.append(string)
            string = ""
    return string_list
