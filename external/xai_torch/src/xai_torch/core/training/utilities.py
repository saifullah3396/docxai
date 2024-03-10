from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Tuple

from xai_torch.core.args import Arguments, ArgumentsLoader
from xai_torch.utilities.general import reset_random_seeds
from xai_torch.utilities.logging_utils import DEFAULT_LOGGER_NAME

if TYPE_CHECKING:
    import optuna
    from ignite.contrib.handlers.base_logger import BaseLogger


def parse_args(
    arguments_class=Arguments, field_name: Optional[str] = None, trial: optuna.trial.Trial = None
) -> Tuple[Arguments, str]:
    # read arguments
    return ArgumentsLoader.parse(arguments_class=arguments_class, field_name=field_name, trial=trial)


def initialize_training(args: Arguments, seed: int = 0, deterministic: bool = False):
    pass

    import os

    import ignite.distributed as idist
    import torch

    from xai_torch.utilities.logging_utils import log_basic_info

    logger = logging.getLogger(DEFAULT_LOGGER_NAME)

    # log basic information
    log_basic_info(args)

    # initialize seed
    rank = idist.get_rank()

    seed = seed + rank
    logger.info(f"Global seed set to {seed}")
    reset_random_seeds(seed)

    # set seedon environment variable
    os.environ["DEFAULT_SEED"] = str(seed)

    # ensure that all operations are deterministic on GPU (if used) for reproducibility
    if deterministic:
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True


def generate_output_dir(
    root_output_dir: str,
    model_task: str,
    dataset_name: str,
    model_name: str,
    experiment_name: str,
    overwrite_output_dir: bool = False,
    logging_dir_suffix: str = "",
):
    """
    Sets up the output dir for an experiment based on the arguments.
    """
    from pathlib import Path

    import ignite.distributed as idist

    logger = logging.getLogger(DEFAULT_LOGGER_NAME)

    # generate root output dir = output_dir / model_task / model_name
    output_dir = Path(root_output_dir) / model_task / dataset_name / Path(model_name)

    # create a unique directory for each experiment
    if logging_dir_suffix != "":
        experiment = f"{experiment_name}/{logging_dir_suffix}"
    else:
        experiment = f"{experiment_name}"

    # append experiment name to output dir
    output_dir = output_dir / experiment

    # overwrite the experiment if required
    if overwrite_output_dir and idist.get_rank() == 0:
        import shutil

        logger.info("Overwriting output directory.")
        shutil.rmtree(output_dir, ignore_errors=True)

    # generate directories
    if not output_dir.exists() and idist.get_rank() == 0:
        output_dir.mkdir(parents=True)

    return output_dir


def setup_logging(
    root_output_dir: str,
    model_task: str,
    dataset_name: str,
    model_name: str,
    experiment_name: str,
    overwrite_output_dir: bool = False,
    logging_dir_suffix: str = "",
) -> Tuple[str, BaseLogger]:
    import ignite.distributed as idist

    rank = idist.get_rank()
    logger = logging.getLogger(DEFAULT_LOGGER_NAME)

    # get the root logging directory
    root_output_dir = generate_output_dir(
        root_output_dir=root_output_dir,
        model_task=model_task,
        dataset_name=dataset_name,
        model_name=model_name,
        experiment_name=experiment_name,
        overwrite_output_dir=overwrite_output_dir,
        logging_dir_suffix=logging_dir_suffix,
    )
    logger.info(f"Setting output directory: {root_output_dir}")

    # Define a Tensorboard logger
    tb_logger = None
    if rank == 0:
        from xai_torch.core.training.tb_logger import XAITensorboardLogger

        tb_logger = XAITensorboardLogger(log_dir=root_output_dir)

    return root_output_dir, tb_logger


def find_checkpoint_file(filename, checkpoint_dir: str, load_best: bool = False, resume=True):
    import glob
    import os
    from pathlib import Path

    if not checkpoint_dir.exists():
        return

    logger = logging.getLogger(DEFAULT_LOGGER_NAME)
    if filename is not None:
        if Path(filename).exists():
            return Path(filename)
        elif Path(checkpoint_dir / filename).exists():
            return Path(checkpoint_dir / filename)
        else:
            logger.warning(f"User provided checkpoint file filename={filename} not found.")

    list_checkpoints = glob.glob(str(checkpoint_dir) + "/*.pt")
    if len(list_checkpoints) > 0:
        if not load_best:
            list_checkpoints = [c for c in list_checkpoints if "best" not in c]
        else:
            list_checkpoints = [c for c in list_checkpoints if "best" in c]

        if len(list_checkpoints) > 0:
            latest_checkpoint = max(list_checkpoints, key=os.path.getctime)
            if resume:
                logger.info(
                    f"Checkpoint detected, resuming training from {latest_checkpoint}. To avoid this behavior, change "
                    "the `general_args.output_dir` or add `general_args.overwrite_output_dir` to train from scratch."
                )
            else:
                logger.info(f"Checkpoint detected, testing model using checkpoint {latest_checkpoint}.")
            return latest_checkpoint


def find_resume_checkpoint(resume_checkpoint_file: str, checkpoint_dir: str, load_best: bool = False):
    return find_checkpoint_file(
        filename=resume_checkpoint_file,
        checkpoint_dir=checkpoint_dir,
        load_best=load_best,
        resume=True,
    )


def find_test_checkpoint(test_checkpoint_file: str, checkpoint_dir: str, load_best: bool = False):
    return find_checkpoint_file(
        filename=test_checkpoint_file,
        checkpoint_dir=checkpoint_dir,
        load_best=load_best,
        resume=False,
    )
