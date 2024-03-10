"""
The main script that serves as the entry-point for all kinds of training experiments.
"""


from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Tuple

from xai_torch.core.args import Arguments
from xai_torch.utilities.general import reset_random_seeds
from xai_torch.utilities.logging_utils import DEFAULT_LOGGER_NAME

if TYPE_CHECKING:
    from ignite.contrib.handlers.base_logger import BaseLogger


def initialize_analyzer(args: Arguments, seed: int = 0, deterministic: bool = False):
    pass

    import ignite.distributed as idist
    import torch

    from xai_torch.utilities.logging_utils import log_basic_info

    logger = logging.getLogger(DEFAULT_LOGGER_NAME)

    # log basic information
    log_basic_info(args)

    # initialize seed
    rank = idist.get_rank()

    logger.info(f"Global seed set to {seed + rank}")
    reset_random_seeds(seed + rank)

    # ensure that all operations are deterministic on GPU (if used) for reproducibility
    if deterministic:
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True


def generate_output_dir(args: Arguments, task_name: str):
    """
    Sets up the output dir for an experiment based on the arguments.
    """
    from pathlib import Path

    import ignite.distributed as idist

    # setup analyser output dir
    output_dir = (
        Path(args.analyzer_args.analyzer_output_dir)
        / task_name
        / args.data_args.dataset_name
        / Path(args.model_args.full_name)
    )

    # generate directories
    if not output_dir.exists() and idist.get_rank() == 0:
        output_dir.mkdir(parents=True)

    return output_dir


def setup_logging(args: Arguments, task_name: str) -> Tuple[str, BaseLogger]:
    import ignite.distributed as idist

    from xai_torch.core.training.tb_logger import XAITensorboardLogger

    rank = idist.get_rank()
    logger = logging.getLogger(DEFAULT_LOGGER_NAME)

    # get the root logging directory
    root_output_dir = generate_output_dir(args, task_name)
    logger.info(f"Setting output directory: {root_output_dir}")

    # Define a Tensorboard logger
    tb_logger = None
    if rank == 0:
        tb_logger = XAITensorboardLogger(log_dir=root_output_dir)

    return root_output_dir, tb_logger
