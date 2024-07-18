import ast
import dataclasses
import importlib
import importlib.util
import inspect
import logging
import os
import pydoc
import sys
import tempfile
from collections import abc
from typing import Any, List

from omegaconf import DictConfig
from yapf.yapflib.yapf_api import FormatCode

try:
    from ast import unparse
except ImportError:
    from astunparse import unparse


class Config:
    def __init__(self, file_path, partials=()):
        file_ext = os.path.splitext(file_path)[-1]
        assert file_ext == ".py", \
            "Currently we only support load .py as config"
        self.partials = partials

        if len(partials) == 0:
            # do not process code, directly load them
            config_dir = os.path.dirname(file_path)
            module_name = os.path.splitext(os.path.basename(file_path))[0]
            sys.path.insert(0, config_dir)
            mod = importlib.import_module(module_name)
            sys.path.pop(0)
            self.__dict__.update({
                name: value
                for name, value in mod.__dict__.items()
                if not name.startswith("__")
            })

            # delete imported module
            del sys.modules[module_name]
        else:
            # load and process code
            with open(file_path, "r") as f:
                code = f.read()
                code = self.partial_optim(code)

            # load the processed code
            with tempfile.TemporaryDirectory() as temp_config_dir:
                # write code into a temporary file, modified from mmcv
                temp_config_file = tempfile.NamedTemporaryFile(
                    dir=temp_config_dir, suffix=file_ext, delete=False, mode="w"
                )
                temp_config_name = os.path.basename(temp_config_file.name)
                code, _ = FormatCode(code)
                temp_config_file.write(code)
                temp_config_file.close()

                # modify sys path to load and load the module
                temp_module_name = os.path.splitext(temp_config_name)[0]
                sys.path.insert(0, temp_config_dir)
                mod = importlib.import_module(temp_module_name)
                sys.path.pop(0)
                self.__dict__.update({
                    name: value
                    for name, value in mod.__dict__.items()
                    if not name.startswith("__")
                })

                # delete imported module and close temp file
                del sys.modules[temp_module_name]
                os.unlink(temp_config_file.name)

    def partial_optim(self, code):
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                if type(node.value) == ast.Call:
                    assign_target = unparse(node.targets[0]).rstrip("\n")
                    variant = assign_target.replace("'", '"')
                    if assign_target in self.partials or variant in self.partials:
                        node.value = ast.Call(
                            func=ast.Name(id="partial", ctx=ast.Load()),
                            args=[node.value.func] + node.value.args,
                            keywords=[] + node.value.keywords,
                        )
        ast_string = "from functools import partial\n" + unparse(tree)
        return ast_string


class LazyConfig:
    def __init__(self, file_path, name_space={}, lazy={}):
        self.lazy = lazy
        with open(file_path, "r") as f:
            code = f.read()
        if len(self.lazy) != 0:
            code = self.replace_call_with_lazy_call(code)
        exec(code, name_space)
        self.__dict__ = {k: v for k, v in name_space.items() if k != "__builtins__"}

    def replace_call_with_lazy_call(self, code):
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                if type(node.value) == ast.Call:
                    assign_target = unparse(node.targets[0]).rstrip("\n")
                    variant = assign_target.replace("'", '"')
                    if assign_target in self.lazy or variant in self.lazy:
                        node.value = ast.Call(
                            func=ast.Call(
                                func=ast.Name(id="L", ctx=ast.Load()),
                                args=[node.value.func],
                                keywords=[],
                            ),
                            args=node.value.args,
                            keywords=node.value.keywords,
                        )
        ast_string = "from util.lazy_load import LazyCall as L\n" + unparse(tree)
        return ast_string


def is_dataclass(obj):
    """Returns True if obj is a dataclass or an instance of a
    dataclass."""
    cls = obj if isinstance(obj, type) and not isinstance(obj, type(List[int])) else type(obj)
    return hasattr(cls, "__dataclass_fields__")


def locate(name: str) -> Any:
    """
    Locate and return an object ``x`` using an input string ``{x.__module__}.{x.__qualname__}``,
    such as "module.submodule.class_name".

    Raise Exception if it cannot be found.
    """
    obj = pydoc.locate(name)

    # Some cases (e.g. torch.optim.sgd.SGD) not handled correctly
    # by pydoc.locate. Try a private function from hydra.
    if obj is None:
        try:
            # from hydra.utils import get_method - will print many errors
            from hydra.utils import _locate
        except ImportError as e:
            raise ImportError(f"Cannot dynamically locate object {name}!") from e
        else:
            obj = _locate(name)  # it raises if fails

    return obj


def _convert_target_to_string(t: Any) -> str:
    """
    Inverse of ``locate()``.

    Args:
        t: any object with ``__module__`` and ``__qualname__``
    """
    module, qualname = t.__module__, t.__qualname__

    # Compress the path to this object, e.g. ``module.submodule._impl.class``
    # may become ``module.submodule.class``, if the later also resolves to the same
    # object. This simplifies the string, and also is less affected by moving the
    # class implementation.
    module_parts = module.split(".")
    for k in range(1, len(module_parts)):
        prefix = ".".join(module_parts[:k])
        candidate = f"{prefix}.{qualname}"
        try:
            if locate(candidate) is t:
                return candidate
        except ImportError:
            pass
    return f"{module}.{qualname}"


class LazyCall:
    """
    Wrap a callable so that when it's called, the call will not be executed,
    but returns a dict that describes the call.
    LazyCall object has to be called with only keyword arguments. Positional
    arguments are not yet supported.
    Examples:
    ::
        from detectron2.config import instantiate, LazyCall
        layer_cfg = LazyCall(nn.Conv2d)(in_channels=32, out_channels=32)
        layer_cfg.out_channels = 64   # can edit it afterwards
        layer = instantiate(layer_cfg)
    """
    def __init__(self, target):
        if not (callable(target) or isinstance(target, (str, abc.Mapping))):
            raise TypeError(
                f"target of LazyCall must be a callable or defines a callable! Got {target}"
            )
        self._target = target

    def __call__(self, *args, **kwargs):
        if is_dataclass(self._target):
            # omegaconf object cannot hold dataclass type
            # https://github.com/omry/omegaconf/issues/784
            target = _convert_target_to_string(self._target)
        else:
            target = self._target
        variable_args, arg_kwargs = self.transfer_args_into_kwargs(args)

        kwargs.update(arg_kwargs)
        kwargs["_target_"] = target
        kwargs["_variable_args_"] = variable_args

        return DictConfig(content=kwargs, flags={"allow_objects": True})

    def transfer_args_into_kwargs(self, args):
        kwargs = {}
        variable_args = None
        params = inspect.signature(self._target).parameters
        for arg_ind, (name, param) in enumerate(params.items()):
            if arg_ind >= len(args):
                break
            if param.kind == inspect._ParameterKind.VAR_POSITIONAL:
                variable_args = args[arg_ind:]
                break
            else:
                kwargs[name] = args[arg_ind]
        return variable_args, kwargs


def instantiate(cfg):
    """
    Recursively instantiate objects defined in dictionaries by
    "_target_" and arguments.

    Args:
        cfg: a dict-like object with "_target_" that defines the caller, and
            other keys that define the arguments

    Returns:
        object instantiated by cfg
    """
    from omegaconf import DictConfig, ListConfig, OmegaConf

    if isinstance(cfg, ListConfig):
        lst = [instantiate(x) for x in cfg]
        return ListConfig(lst, flags={"allow_objects": True})
    if isinstance(cfg, list):
        # Specialize for list, because many classes take
        # list[objects] as arguments, such as ResNet, DatasetMapper
        return [instantiate(x) for x in cfg]

    # If input is a DictConfig backed by dataclasses (i.e. omegaconf's structured config),
    # instantiate it to the actual dataclass.
    if isinstance(cfg, DictConfig) and dataclasses.is_dataclass(cfg._metadata.object_type):
        return OmegaConf.to_object(cfg)

    if isinstance(cfg, abc.Mapping) and "_target_" in cfg:
        # conceptually equivalent to hydra.utils.instantiate(cfg) with _convert_=all,
        # but faster: https://github.com/facebookresearch/hydra/issues/1200
        cfg = {k: instantiate(v) for k, v in cfg.items()}
        cls = cfg.pop("_target_")
        variable_args = cfg.pop("_variable_args_")
        cls = instantiate(cls)

        if isinstance(cls, str):
            cls_name = cls
            cls = locate(cls_name)
            assert cls is not None, cls_name
        else:
            try:
                cls_name = cls.__module__ + "." + cls.__qualname__
            except Exception:
                # target could be anything, so the above could fail
                cls_name = str(cls)
        assert callable(cls), f"_target_ {cls} does not define a callable object"
        try:
            # split args from kwargs and instantiate cls with normal sequence:
            # args, variable_args, kwargs
            if variable_args is not None:
                params = inspect.signature(cls).parameters
                try:
                    p_kind_list = [p.kind for p in params.values()]
                    i = p_kind_list.index(inspect._ParameterKind.VAR_POSITIONAL)
                except ValueError:
                    i = None
                arg_keys = list(params.keys())[:i]
                args = []
                for key in arg_keys:
                    args.append(cfg.pop(key))
                if variable_args is not None:
                    args.extend(variable_args)
                return cls(*args, **cfg)
            else:
                return cls(**cfg)
        except TypeError:
            import os

            logger = logging.getLogger(os.path.basename(os.getcwd()) + "." + __name__)
            logger.error(f"Error when instantiating {cls_name}!")
            raise
    return cfg  # return as-is if don't know what to do
