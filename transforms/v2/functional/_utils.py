from typing import Any

import torch
from util.datapoints import Datapoint


def is_simple_tensor(inpt: Any) -> bool:
    return isinstance(inpt, torch.Tensor) and not isinstance(inpt, Datapoint)
