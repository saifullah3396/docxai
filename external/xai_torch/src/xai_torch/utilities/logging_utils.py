"""
Defines logging related utility functions/classes.
"""
from __future__ import annotations

import logging
from typing import Mapping, Optional, TextIO

DEFAULT_LOGGER_NAME = "xai_torch"


def log_basic_info(args: Mapping, log_args=False):
    """
    Logs the basic information on training initialization.
    """

    import ignite
    import ignite.distributed as idist
    import torch

    logger = logging.getLogger(DEFAULT_LOGGER_NAME)
    logger.info(f"PyTorch version: {torch.__version__}{idist.get_rank()}")
    logger.info(f"Ignite version: {ignite.__version__}{idist.get_rank()}")

    if torch.cuda.is_available():
        # explicitly import cudnn as
        # torch.backends.cudnn can not be pickled with hvd spawning procs
        from torch.backends import cudnn

        logger.info(f"GPU Device: {torch.cuda.get_device_name(idist.get_local_rank())}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"CUDNN version: {cudnn.version()}")
    else:
        logger.info(f"Device: {idist.device()}")

    # print args
    logger.info("Initializing training script with the following arguments:")
    if idist.get_rank() == 0 and log_args:
        print(args)

    if idist.get_world_size() > 1:
        logger.info("Distributed setting:")
        logger.info(f"backend: {idist.backend()}")
        logger.info(f"world size: {idist.get_world_size()}")


def setup_logger(
    name: Optional[str] = "ignite",
    level: int = logging.INFO,
    stream: Optional[TextIO] = None,
    format: str = "%(asctime)s %(name)s %(levelname)s: %(message)s",
    filepath: Optional[str] = None,
    distributed_rank: Optional[int] = None,
    reset: bool = False,
) -> logging.Logger:

    import coloredlogs

    # check if the logger already exists
    existing = name is None or name in logging.root.manager.loggerDict

    # if existing, get the logger otherwise create a new one
    logger = logging.getLogger(name)

    if distributed_rank is None:
        import ignite.distributed as idist

        distributed_rank = idist.get_rank()

    # Remove previous handlers
    if distributed_rank > 0 or reset:

        if logger.hasHandlers():
            for h in list(logger.handlers):
                logger.removeHandler(h)

    if distributed_rank > 0:

        # Add null handler to avoid multiple parallel messages
        logger.addHandler(logging.NullHandler())

    # Keep the existing configuration if not reset
    if existing and not reset:
        return logger

    if distributed_rank == 0:
        logger.setLevel(level)

        formatter = logging.Formatter(format)

        ch = logging.StreamHandler(stream=stream)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        coloredlogs.install(logger=logger)

        if filepath is not None:
            fh = logging.FileHandler(filepath)
            fh.setLevel(level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    # don't propagate to ancestors
    # the problem here is to attach handlers to loggers
    # should we provide a default configuration less open ?
    if name is not None:
        logger.propagate = False

    return logger
