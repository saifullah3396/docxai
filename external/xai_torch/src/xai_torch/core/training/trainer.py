from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from omegaconf import DictConfig

from xai_torch.core.args import Arguments
from xai_torch.core.factory.factory import BatchSamplerFactory
from xai_torch.core.training.constants import TrainingStage
from xai_torch.core.training.trainer_base import TrainerBase
from xai_torch.utilities.logging_utils import DEFAULT_LOGGER_NAME

if TYPE_CHECKING:
    pass

    from xai_torch.core.args import Arguments
    from xai_torch.core.data.data_modules.base import BaseDataModule


def log_eval_metrics(logger, epoch, elapsed, tag, metrics):
    metrics_output = "\n".join([f"\t{k}: {v}" for k, v in metrics.items()])
    logger.info(f"\nEpoch {epoch} - Evaluation time (seconds): {elapsed:.2f} - {tag} metrics:\n {metrics_output}")


class Trainer:
    def __init__(self, args: Arguments) -> None:
        self._args = args
        self._output_dir = None
        self._tb_logger = None
        self._trainer_base = None
        self._model = None
        self._opt_sch_handler = None
        self._datamodule = None
        self._train_dataloader = None
        self._val_dataloader = None
        self._test_dataloader = None

    @property
    def optimizers(self):
        return self._opt_sch_handler.optimizers

    @property
    def lr_schedulers(self):
        return self._opt_sch_handler.lr_schedulers

    @property
    def wd_schedulers(self):
        return self._opt_sch_handler.wd_schedulers

    @property
    def batches_per_epch(self):
        return self._trainer_base._get_batches_per_epoch(self._train_dataloader)

    @property
    def steps_per_epoch(self):
        return self._trainer_base._get_steps_per_epoch(self._args, self._train_dataloader)

    @property
    def total_training_steps(self):
        return self._trainer_base._get_total_training_steps(self._args, self._train_dataloader)

    @property
    def warmup_steps(self):
        return self._trainer_base._get_warmup_steps(self._args, self._train_dataloader)

    def _setup_training(self):
        import ignite.distributed as idist

        from xai_torch.core.training.utilities import initialize_training, setup_logging

        # initialize training
        initialize_training(
            self._args, seed=self._args.general_args.seed, deterministic=self._args.general_args.deterministic
        )

        # initialize torch device (cpu or gpu)
        self._device = idist.device()

        # get device rank
        self._rank = idist.get_rank()

        # initialize logging directory and tensorboard logger
        self._output_dir, self._tb_logger = setup_logging(
            root_output_dir=self._args.general_args.root_output_dir,
            model_task=self._args.model_args.model_task,
            dataset_name=self._args.data_args.dataset_name,
            model_name=self._args.model_args.full_name,
            experiment_name=self._args.training_args.experiment_name,
            overwrite_output_dir=self._args.general_args.overwrite_output_dir,
        )

    def _setup_trainer_base(self):
        return TrainerBase

    def _setup_datamodule(self, stage: TrainingStage = TrainingStage.train) -> BaseDataModule:
        return self._trainer_base.setup_datamodule(self._args, rank=self._rank, stage=stage)

    def _setup_model(self, summarize: bool = True, stage: TrainingStage = TrainingStage.train) -> BaseDataModule:
        return self._trainer_base.setup_model(
            self._args, self._tb_logger, summarize=summarize, stage=stage, gt_metadata=self._datamodule.gt_metadata
        )

    def _setup_training_engine(self):
        # setup training engine
        training_engine, _ = self._trainer_base.setup_training_engine(
            args=self._args,
            model=self._model,
            opt_sch_handler=self._opt_sch_handler,
            train_dataloader=self._train_dataloader,
            val_dataloader=self._val_dataloader,
            output_dir=self._output_dir,
            tb_logger=self._tb_logger,
            device=self._device,
            do_val=self._args.general_args.do_val,
        )
        training_engine.logger = logging.getLogger(DEFAULT_LOGGER_NAME)
        return training_engine

    def _setup_test_engine(self, checkpoint_type: str = "last"):
        # setup training engine
        test_engine = self._trainer_base.setup_test_engine(
            args=self._args,
            model=self._model,
            test_dataloader=self._test_dataloader,
            output_dir=self._output_dir,
            tb_logger=self._tb_logger,
            device=self._device,
            checkpoint_type=checkpoint_type,
        )
        test_engine.logger = logging.getLogger(DEFAULT_LOGGER_NAME)

        return test_engine

    def _setup_optimizers_schedulers(self):
        from xai_torch.core.training.optim_sch_handler import OptimizersSchedulersHandler

        opt_sch_handler = OptimizersSchedulersHandler(self._args, self._model, self)
        opt_sch_handler.setup()
        return opt_sch_handler

    def train(self, local_rank=0):
        """
        Initializes the training of a model given dataset, and their configurations.
        """

        from xai_torch.utilities.logging_utils import DEFAULT_LOGGER_NAME, setup_logger

        # setup logging
        logger = setup_logger(DEFAULT_LOGGER_NAME, distributed_rank=local_rank, level=logging.INFO)

        # setup training
        self._setup_training()

        # setup base training functionality
        self._trainer_base = self._setup_trainer_base()

        # setup datamodule
        self._datamodule = self._setup_datamodule()

        # setup batch sampler if needed
        batch_sampler_wrapper = BatchSamplerFactory.create(
            self._args.data_args.data_loader_args.train_batch_sampler_args.strategy,
            **self._args.data_args.data_loader_args.train_batch_sampler_args.config,
        )

        # setup dataloaders
        self._train_dataloader = self._datamodule.train_dataloader(
            self._args.data_args.data_loader_args.per_device_train_batch_size,
            dataloader_num_workers=self._args.data_args.data_loader_args.dataloader_num_workers,
            pin_memory=self._args.data_args.data_loader_args.pin_memory,
            shuffle_data=self._args.data_args.data_loader_args.shuffle_data,
            dataloader_drop_last=self._args.data_args.data_loader_args.dataloader_drop_last,
            batch_sampler_wrapper=batch_sampler_wrapper,
        )
        self._val_dataloader = self._datamodule.val_dataloader(
            self._args.data_args.data_loader_args.per_device_eval_batch_size,
            dataloader_num_workers=self._args.data_args.data_loader_args.dataloader_num_workers,
            pin_memory=self._args.data_args.data_loader_args.pin_memory,
        )

        # setup model
        self._model = self._setup_model(summarize=True, stage=TrainingStage.train)

        # initialize optimizers schedulers handler
        self._opt_sch_handler = self._setup_optimizers_schedulers()

        # setup training engine
        self._training_engine = self._setup_training_engine()

        resume_epoch = self._training_engine.state.epoch
        if (
            self._training_engine._is_done(self._training_engine.state)
            and resume_epoch >= self._args.training_args.max_epochs
        ):  # if we are resuming from last checkpoint and training is already finished
            logger.info(
                "Training has already been finished! Either increase the number of "
                f"epochs (current={self._args.training_args.max_epochs}) >= {resume_epoch} "
                "OR reset the training from start."
            )
            return

        # run training
        self._training_engine.run(self._train_dataloader, max_epochs=self._args.training_args.max_epochs)

        if self._rank == 0:
            # close tb logger
            self._tb_logger.close()

    def test(self, local_rank=0):
        """
        Initializes the training of a model given dataset, and their configurations.
        """

        from xai_torch.utilities.logging_utils import DEFAULT_LOGGER_NAME, setup_logger

        # make sure this is set to false always in evaluation
        self._args.general_args.overwrite_output_dir = False

        # setup logging
        logger = setup_logger(DEFAULT_LOGGER_NAME, distributed_rank=local_rank, level=logging.INFO)

        # setup training
        self._setup_training()

        # setup base training functionality
        self._trainer_base = self._setup_trainer_base()

        # setup datamodule
        self._datamodule = self._setup_datamodule(stage=TrainingStage.test)

        # setup dataloaders
        self._test_dataloader = self._datamodule.test_dataloader(
            self._args.data_args.data_loader_args.per_device_eval_batch_size,
            dataloader_num_workers=self._args.data_args.data_loader_args.dataloader_num_workers,
            pin_memory=self._args.data_args.data_loader_args.pin_memory,
        )

        # setup model
        self._model = self._setup_model(summarize=True, stage=TrainingStage.test)

        # setup test engines for different types of model checkpoints
        for checkpoint_type in ["last", "best", "ema"]:
            # setup training engine
            self._test_engine = self._setup_test_engine(checkpoint_type)

            # run tests
            self._test_engine.run(self._test_dataloader)

        # close tb logger
        self._tb_logger.close()

    @classmethod
    def train_parallel(cls, local_rank: int, args: Arguments):
        cls(args).train(local_rank)

    @classmethod
    def run_diagnostic(cls, local_rank: int, args: Arguments):
        import os

        import ignite.distributed as idist
        import torch

        prefix = f"{local_rank}) "
        print(f"{prefix}Rank={idist.get_rank()}")
        print(f"{prefix}torch version: {torch.version.__version__}")
        print(f"{prefix}torch git version: {torch.version.git_version}")

        if torch.cuda.is_available():
            print(f"{prefix}torch version cuda: {torch.version.cuda}")
            print(f"{prefix}number of cuda devices: {torch.cuda.device_count()}")

            for i in range(torch.cuda.device_count()):
                print(f"{prefix}\t- device {i}: {torch.cuda.get_device_properties(i)}")
        else:
            print("{prefix}no cuda available")

        if "SLURM_JOBID" in os.environ:
            for k in [
                "SLURM_PROCID",
                "SLURM_LOCALID",
                "SLURM_NTASKS",
                "SLURM_JOB_NODELIST",
                "MASTER_ADDR",
                "MASTER_PORT",
            ]:
                print(f"{k}: {os.environ[k]}")

    @classmethod
    def run(cls, cfg: DictConfig):
        import ignite.distributed as idist
        from omegaconf import OmegaConf

        from xai_torch.core.args import Arguments
        from xai_torch.utilities.dacite_wrapper import from_dict

        # initialize general configuration for script
        cfg = OmegaConf.to_object(cfg)
        args = from_dict(data_class=Arguments, data=cfg["args"])
        if args.general_args.n_devices > 1:
            if args.general_args.do_train:
                try:
                    with idist.Parallel(backend=args.general_args.backend) as parallel:
                        parallel.run(cls.train_parallel, args)
                except KeyboardInterrupt:
                    logging.info("Received ctrl-c interrupt. Stopping training...")
                except Exception as e:
                    logging.exception(e)

            if args.general_args.do_test:
                try:
                    cls(args).test()
                except Exception as e:
                    logging.exception(e)
                    exit(1)
        else:
            if args.general_args.do_train:
                try:
                    cls(args).train()
                except KeyboardInterrupt:
                    logging.info("Received ctrl-c interrupt. Stopping training...")
                except Exception as e:
                    logging.exception(e)
                    exit(1)

            if args.general_args.do_test:
                try:
                    cls(args).test()
                except Exception as e:
                    logging.exception(e)
                    exit(1)
