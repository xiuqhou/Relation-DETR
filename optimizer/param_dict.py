from typing import List, Tuple, Union

from torch import nn


def match_name_keywords(name: str, name_keywords: Union[Tuple, List, str]):
    if isinstance(name_keywords, str):
        name_keywords = [name_keywords]
    for b in name_keywords:
        if b in name:
            return True
    return False

def basic_param(model, lr):
    return [{"params": [p for p in model.parameters() if p.requires_grad], "lr": lr}]

def finetune_backbone_param(model, lr):
    return [
        {
            "params": [
                p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad
            ]
        },
        {
            "params": [
                p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad
            ],
            "lr": lr * 0.1,
        },
    ]


def finetune_backbone_with_no_norm_weight_decay(model, lr):
    norm_classes = (
        nn.modules.batchnorm._BatchNorm,
        nn.LayerNorm,
        nn.GroupNorm,
        nn.modules.instancenorm._InstanceNorm,
        nn.LocalResponseNorm,
    )
    backbone_norm = []
    other_norm = []
    backbone = []
    other = []
    for name, module in model.named_modules():
        if next(module.children(), None):
            if "backbone" in name:
                backbone.extend(p for p in module.parameters(recurse=False) if p.requires_grad)
            else:
                other.extend(p for p in module.parameters(recurse=False) if p.requires_grad)
        elif isinstance(module, norm_classes):
            if "backbone" in name:
                backbone_norm.extend(p for p in module.parameters() if p.requires_grad)
            else:
                other_norm.extend(p for p in module.parameters() if p.requires_grad)
        else:
            if "backbone" in name:
                backbone.extend(p for p in module.parameters() if p.requires_grad)
            else:
                other.extend(p for p in module.parameters() if p.requires_grad)
    return [
        {
            "params": other,
        },
        {
            "params": backbone_norm,
            "lr": lr * 0.1,
            "weight_decay": 0,
        },
        {
            "params": other_norm,
            "weight_decay": 0,
        },
        {
            "params": backbone,
            "lr": lr * 0.1,
        },
    ]


def finetune_backbone_and_linear_projection(model, lr):
    linear_keywords = ("reference_points", "sampling_offsets")
    norm_bias_keywords = ("norm", "bias")
    backbone = []
    backbone_norm = []
    linear_projection = []
    linear_projection_norm = []
    other = []
    other_norm = []
    for name, parameters in model.named_parameters():
        if not parameters.requires_grad:
            continue
        if (
            match_name_keywords(name, "backbone")
            and not match_name_keywords(name, linear_keywords)
            and match_name_keywords(name, norm_bias_keywords)
        ):
            backbone_norm.append(parameters)
        elif (
            match_name_keywords(name, "backbone")
            and not match_name_keywords(name, linear_keywords)
            and not match_name_keywords(name, norm_bias_keywords)
        ):
            backbone.append(parameters)
        elif (
            not match_name_keywords(name, "backbone")
            and match_name_keywords(name, linear_keywords)
            and match_name_keywords(name, norm_bias_keywords)
        ):
            linear_projection_norm.append(parameters)
        elif (
            not match_name_keywords(name, "backbone")
            and match_name_keywords(name, linear_keywords)
            and not match_name_keywords(name, norm_bias_keywords)
        ):
            linear_projection.append(parameters)
        elif match_name_keywords(name, norm_bias_keywords):
            other_norm.append(parameters)
        else:
            other.append(parameters)

    return [
        {
            "params": other,
        },
        {
            "params": backbone,
            "lr": lr * 0.1,
        },
        {
            "params": backbone_norm,
            "lr": lr * 0.1,
            "weight_decay": 0,
        },
        {
            "params": linear_projection,
            "lr": lr * 0.1,
        },
        {
            "params": linear_projection_norm,
            "lr": lr * 0.1,
            "weight_decay": 0,
        },
        {
            "params": other_norm,
            "weight_decay": 0,
        },
    ]
