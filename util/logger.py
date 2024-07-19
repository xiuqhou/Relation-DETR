import atexit
import functools
import logging
import os
import re
import sys

from accelerate.logging import get_logger
from iopath.common.file_io import HTTPURLHandler, OneDrivePathHandler
from iopath.common.file_io import PathManager as PathManagerBase
from termcolor import colored

PathManager = PathManagerBase()
PathManager.register_handler(HTTPURLHandler())
PathManager.register_handler(OneDrivePathHandler())


def create_logger(output_dir=None, dist_rank=0):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    fmt = "[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s"
    color_fmt = colored("[%(asctime)s %(name)s]", "green")
    color_fmt += colored("(%(filename)s %(lineno)d)", "yellow")
    color_fmt += ": %(levelname)s %(message)s"

    # create console handlers for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(logging.Formatter(fmt=color_fmt, datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(console_handler)

        # create file handlers
        if output_dir:
            file_handler = logging.FileHandler(os.path.join(output_dir, "training.log"), mode="a")
            file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S"))
            logger.addHandler(file_handler)

    return logger


class _ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."
        self._abbrev_name = kwargs.pop("abbrev_name", "")
        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name + "."
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        log = super(_ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log


class ColorFilter(logging.Filter):
    def filter(self, record):
        message = record.getMessage()
        # matching colored patterns
        pattern = re.compile(r'\x1b\[[0-9;]*m')
        if pattern.search(message):
            record.msg = pattern.sub('', message)
        return True


@functools.lru_cache()  # so that calling setup_logger multiple times won't add many handlers
def setup_logger(
    output=None,
    distributed_rank=0,
    *,
    color=True,
    name="detection",
    abbrev_name=None,
    enable_propagation: bool = False,
    configure_stdout: bool = True,
):
    """Initialize the detection logger and set its verbosity level to "DEBUG"

    :param output: a file name or a directory to save log. If None, will not save log file.
        If ends with ".txt" or ".log", assumed to be a file name, defaults to None
    :param distributed_rank: rank number id in distributed training, defaults to 0
    :param color: whether to show colored logging information, defaults to True
    :param name: the root module name of this logger, defaults to "detection"
    :param abbrev_name: an abbreviation of the module, to avoid long names in logs.
        Set to "" to not log the root module in logs. By default, will abbreviate "detection"
        to "det" and leave other modules unchanged, defaults to None
    :param enable_propagation: whether to propogate logs to the parent logger, defaults to False
    :param configure_stdout: whether to configure logging to stdout, defaults to True
    """
    logger_adapter = get_logger(name, "DEBUG")
    logger = logger_adapter.logger
    logger.propagate = enable_propagation

    if abbrev_name is None:
        abbrev_name = name.replace(os.path.basename(os.getcwd()), "det")

    plain_formatter = logging.Formatter(
        "[%(asctime)s %(name)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    # stdout logging: master only
    if configure_stdout and distributed_rank == 0:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        if color:
            formatter = _ColorfulFormatter(
                colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                root_name=name,
                abbrev_name=str(abbrev_name),
            )
        else:
            formatter = plain_formatter
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # file logging: all workers
    if output is not None:
        if output.endswith(".txt") or output.endswith(".log"):
            filename = output
        else:
            filename = os.path.join(output, "log.log")
        if distributed_rank > 0:
            filename = filename.replace(".", "_rank{}".format(distributed_rank) + ".")
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        fh = logging.StreamHandler(_cached_log_stream(filename))
        fh.addFilter(ColorFilter())
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)

    return logger_adapter


# cache the opened file object, so that different calls to `setup_logger`
# with the same file name can safely write to the same file.
@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    # use 1K buffer if writing to cloud storage
    io = PathManager.open(filename, "a", buffering=_get_log_stream_buffer_size(filename))
    atexit.register(io.close)
    return io


def _get_log_stream_buffer_size(filename: str) -> int:
    if "://" not in filename:
        # Local file, no extra caching is necessary
        return -1
    # Remote file requires a larger cache to avoid many small writes.
    return 1024 * 1024
