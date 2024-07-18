import datetime
import logging
import os
import pickle
import time
import warnings
from collections import OrderedDict, defaultdict, deque

import torch
import torch.distributed as dist
from accelerate.logging import get_logger
from torch import nn

from terminaltables import AsciiTable


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.__dict__["meters"]:
            return self.__dict__["meters"][attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        logger = logging.getLogger(os.path.basename(os.getcwd()) + "." + __name__)
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                "[{0" + space_fmt + "}/{1}]",
                "eta: {eta}",
                "{meters}",
                "time: {time}",
                "data: {data}",
                "mem: {memory:.0f}",
                "max mem: {max_memory:.0f}",
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                "[{0" + space_fmt + "}/{1}]",
                "eta: {eta}",
                "{meters}",
                "time: {time}",
                "data: {data}",
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    logger.info(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.memory_allocated() / MB,
                            max_memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    logger.info(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("{} Total time: {} ({:.4f} s / it)".format(header, total_time_str, total_time / len(iterable)))


class HighestCheckpoint:
    def __init__(self, accelerator, model):
        self.accelerate = accelerator
        self.model = model
        self.meters = {}

    def update(self, **kwargs):
        logger = get_logger(os.path.basename(os.getcwd()) + "." + __name__)
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            if k not in self.meters:
                self.meters[k] = v
            if v >= self.meters[k]:
                self.meters.update({k: v})
                save_path = os.path.join(self.accelerate.project_dir, f"best_{k}.pth")
                model_state_dict = self.accelerate.get_state_dict(self.model, unwrap=True)
                self.accelerate.save(model_state_dict, save_path)
                logger.info(f"the best {k} model is saved to {save_path}")


class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """
    def __init__(self, model, decay, device="cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=True)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print("| distributed init (rank {}): {}".format(args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def filter_mismatched_weights(model_state_dict, weight_state_dict):
    mismatch_keys = {}
    for key in list(model_state_dict.keys()):
        if key in weight_state_dict:
            value_model = model_state_dict[key]
            value_state_dict = weight_state_dict[key]
            if value_model.shape != value_state_dict.shape:
                weight_state_dict[key] = value_model
                mismatch_keys[key] = [value_model.shape, value_state_dict.shape]
    return weight_state_dict, mismatch_keys


def load_checkpoint(file_name, map_location="cpu"):
    if isinstance(file_name, str):
        if file_name.startswith("http://") or file_name.startswith("https://"):
            return torch.hub.load_state_dict_from_url(file_name, map_location=map_location)
        elif os.path.exists(file_name):
            return torch.load(file_name, map_location=map_location)
        else:
            warnings.warn("Given string, only url and local path of weight are supported, skip loading.")
            return None
    elif isinstance(file_name, OrderedDict):
        return file_name

    return None


def load_state_dict(model: nn.Module, state_dict: OrderedDict):
    logger = logging.getLogger(os.path.basename(os.getcwd()) + "." + __name__)
    if state_dict is None:
        logger.warn("State dict is None, skip loading")
        return

    # load _classes_ for inference
    if "_classes_" in state_dict:
        dummy_classes = torch.zeros_like(state_dict["_classes_"])
        model.register_buffer("_classes_", dummy_classes)

    # initialize keys list
    matched_state_dict, mismatch_keys = filter_mismatched_weights(model.state_dict(), state_dict)
    incompatible_keys = model.load_state_dict(matched_state_dict, strict=False)
    missing_keys = incompatible_keys.missing_keys
    unexpected_keys = incompatible_keys.unexpected_keys
    if len(mismatch_keys) == 0 and len(missing_keys) == 0 and len(unexpected_keys) == 0:
        logger.info(incompatible_keys)
    else:
        logger.warning("The model and loaded state dict do not match exactly")
        if len(missing_keys) != 0:
            logger.warning(f"Missing keys: {', '.join(missing_keys)}\n")
        if len(unexpected_keys) != 0:
            logger.warning(f"Unexpected keys: {', '.join(unexpected_keys)}\n")
        if len(mismatch_keys) != 0:
            mismatch_tables = [["key name", "shape in model", "shape in state dict"]]

            def format_shape(shape):
                shape_str = "("
                shape_str += ", ".join([str(i) for i in shape])
                shape_str += ",)" if len(shape) == 1 else ")"
                return shape_str

            for key, (shape_model, shape_state_dict) in mismatch_keys.items():
                mismatch_tables.append([key, format_shape(shape_model), format_shape(shape_state_dict)])
            mismatch_tables = AsciiTable(mismatch_tables)
            mismatch_tables.inner_footing_row_border = True
            logger.warning(f"Size mismatch keys: {', '.join(mismatch_keys)}\n" + mismatch_tables.table + "\n")
