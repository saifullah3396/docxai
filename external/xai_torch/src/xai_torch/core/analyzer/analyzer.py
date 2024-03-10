from __future__ import annotations

import logging

from omegaconf import DictConfig

import xai_torch.analyzer  # initialize tasks


class Analyzer:
    @classmethod
    def analyze(self, args):
        """
        Initializes the training of a model given dataset, and their configurations.
        """

        from xai_torch.core.analyzer.tasks.factory import AnalyzerTaskFactory
        from xai_torch.utilities.logging_utils import DEFAULT_LOGGER_NAME, setup_logger

        # make sure this is set to false always in evaluation
        args.general_args.overwrite_output_dir = False

        # setup logging
        logger = setup_logger(DEFAULT_LOGGER_NAME, distributed_rank=0, level=logging.INFO)

        if args.analyzer_args is None:
            raise ValueError("No analyzer arguments found in the config.")

        # setup task
        analyzer_tasks = AnalyzerTaskFactory.create(args)

        # run task on datamodule and models
        for task_name, task in analyzer_tasks.items():
            logger.info(f"Running task: {task_name}")

            # initialize task
            task.setup(task_name)

            # run task
            task.run()

            # task cleanup
            task.cleanup()

    @classmethod
    def run(cls, cfg: DictConfig):
        from omegaconf import OmegaConf

        from xai_torch.core.args import Arguments
        from xai_torch.utilities.dacite_wrapper import from_dict

        # initialize general configuration for script
        cfg = OmegaConf.to_object(cfg)
        args = from_dict(data_class=Arguments, data=cfg["args"])
        try:
            cls.analyze(args)
        except Exception as e:
            logging.exception(e)
            exit(1)
