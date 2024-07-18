import copy
import inspect
import logging
import os
from typing import Dict

from omegaconf import DictConfig
from torch import nn

from util.utils import load_state_dict as _load_state_dict


class BaseBackbone:
    @staticmethod
    def load_state_dict(model: nn.Module, state_dict: Dict):
        if state_dict is None:
            return
        assert isinstance(state_dict, Dict), "state_dict must be OrderedDict."
        _load_state_dict(model, state_dict)

    @staticmethod
    def freeze_module(module: nn.Module):
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    def get_instantiate_config(self, func_name, arch, extra_params):
        # log some necessary information about backbone
        logger = logging.getLogger(os.path.basename(os.getcwd()) + "." + __name__)
        assert arch is None or arch in self.model_arch, \
            f"Expected architecture in {self.model_arch.keys()} but got {arch}"
        logger.info(f"Backbone architecture: {arch}")

        # merge parameters from self.arch with extra params
        model_config = copy.deepcopy(self.model_arch[arch]) if arch is not None else {}
        for name, param in inspect.signature(func_name).parameters.items():
            # get default, current and modified params
            default = param.default if param.default is not inspect.Parameter.empty else None
            modified_param = extra_params.get(name, None)
            if isinstance(model_config, Dict):
                cur_param = model_config.get(name, None)
            elif isinstance(model_config, DictConfig):
                cur_param = getattr(model_config, name, None)
            else:
                cur_param = None

            # choose the high-prior parameter
            if cur_param is not None:
                default = cur_param
            if modified_param is not None:
                default = modified_param

            # replace parameters in model_config
            if isinstance(model_config, Dict):
                model_config[name] = default
            elif isinstance(model_config, DictConfig):
                setattr(model_config, name, default)
            else:
                raise TypeError("Only Dict and DictConfig supported.")

        return model_config
