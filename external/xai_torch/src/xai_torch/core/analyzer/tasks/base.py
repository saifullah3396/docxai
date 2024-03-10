"""
Defines the base DataAugmentation class for defining any kind of data augmentation.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from inspect import Arguments
from typing import TYPE_CHECKING

from xai_torch.core.analyzer.tasks.base_config import AnalyzerTaskConfig
from xai_torch.core.models.xai_model import XAIModel
from xai_torch.core.training.constants import TrainingStage
from xai_torch.utilities.dacite_wrapper import from_dict
from xai_torch.utilities.logging_utils import DEFAULT_LOGGER_NAME

if TYPE_CHECKING:
    from xai_torch.core.args import Arguments
    from xai_torch.core.data.data_modules.base import BaseDataModule


class AnalyzerTask(ABC):
    @dataclass
    class Config(AnalyzerTaskConfig):
        pass

    def __init__(self, args: Arguments, config: AnalyzerTaskConfig) -> None:
        self._args = args
        self._output_dir = None
        self._tb_logger = None
        self._trainer_base = None
        self._datamodule = None
        self._test_dataloader = None
        self._config = self._setup_config(config)

    @property
    def config(self):
        return self._config

    def _setup_config(self, config: AnalyzerTaskConfig):
        logger = logging.getLogger(DEFAULT_LOGGER_NAME)
        try:
            # initialize or validate the config
            if config is None:
                return self.Config()
            elif isinstance(config, dict):
                return from_dict(
                    data_class=self.Config,
                    data=config,
                )
        except Exception as e:
            logger.exception(f"Exception raised while initializing config for task: {self.__class__}: {e}")
            exit()

    def get_collate_fns(self):
        return None

    def _setup_analysis(self, task_name: str):
        import ignite.distributed as idist

        from xai_torch.core.analyzer.utilities import initialize_analyzer, setup_logging

        # initialize training
        initialize_analyzer(
            self._args, seed=self._args.general_args.seed, deterministic=self._args.general_args.deterministic
        )

        # initialize torch device (cpu or gpu)
        self._device = idist.device()

        # get device rank
        self._rank = idist.get_rank()

        # initialize logging directory and tensorboard logger
        self._output_dir, self._tb_logger = setup_logging(self._args, task_name)

    def _setup_trainer_base(self):
        from xai_torch.core.training.trainer import TrainerBase

        return TrainerBase

    def _setup_datamodule(self, stage: TrainingStage = TrainingStage.train) -> BaseDataModule:
        return self._trainer_base.setup_datamodule(
            self._args, rank=self._rank, stage=stage, override_collate_fns=self.get_collate_fns()
        )

    def _setup_test_engine(self, model: XAIModel):
        # setup training engine
        test_engine = self._trainer_base.setup_test_engine(
            args=self._args,
            model=model,
            test_dataloader=self._test_dataloader,
            output_dir=self._output_dir,
            tb_logger=self._tb_logger,
            device=self._device,
        )
        test_engine.logger = logging.getLogger(DEFAULT_LOGGER_NAME)

        return test_engine

    def setup(self, task_name: str):
        # setup base training functionality
        self._trainer_base = self._setup_trainer_base()

        # setup training
        self._setup_analysis(task_name)

        # setup datamodule
        self._datamodule = self._setup_datamodule(stage=TrainingStage.test)

        # setup dataloaders
        self._test_dataloader = self._datamodule.test_dataloader(
            self._args.data_args.data_loader_args.per_device_eval_batch_size,
            dataloader_num_workers=self._args.data_args.data_loader_args.dataloader_num_workers,
            pin_memory=self._args.data_args.data_loader_args.pin_memory,
        )

    def cleanup(self):
        # close tb logger
        self._tb_logger.close()

    @abstractmethod
    def run(self):
        pass
