from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Mapping

import torch

from xai_torch.core.args import Arguments
from xai_torch.core.models.xai_model import XAIModel
from xai_torch.core.training.trainer import Trainer
from xai_torch.utilities.logging_utils import DEFAULT_LOGGER_NAME

if TYPE_CHECKING:
    import torch

    from xai_torch.core.args import Arguments
    from xai_torch.core.models.xai_model import XAIModel


def log_eval_metrics(logger, epoch, elapsed, tag, metrics):
    metrics_output = "\n".join([f"\t{k}: {v}" for k, v in metrics.items()])
    logger.info(f"\nEpoch {epoch} - Evaluation time (seconds): {elapsed:.2f} - {tag} metrics:\n {metrics_output}")


class OptimizersSchedulersHandler:
    def __init__(self, args: Arguments, model: XAIModel, trainer: Trainer) -> None:

        self._args = args
        self._model = model
        self._trainer = trainer
        self._optimizers = None
        self._lr_schedulers = None
        self._wd_schedulers = None
        self._logger = logging.getLogger(DEFAULT_LOGGER_NAME)

    @property
    def optimizers(self):
        return self._optimizers

    @property
    def lr_schedulers(self):
        return self._lr_schedulers

    @property
    def wd_schedulers(self):
        return self._wd_schedulers

    @property
    def param_groups(self):
        return self._model.torch_model.get_param_groups()

    def _setup_optimizers(self):
        from xai_torch.core.training.optim.factory import OptimizerFactory

        # get model parameter groups
        model_param_groups = self.param_groups

        # setup optimizers dictionary
        optimizers = {}
        for k, args in self._args.training_args.optimizers.items():
            if k not in model_param_groups.keys():
                raise ValueError(
                    f"Your optimizer configuration does not align the model optimizer "
                    f"parameter groups. {k} =/= {model_param_groups.keys()}"
                )

            # set optimizers
            if self._args.model_args.config.bypass_params_creation:
                optimizers[k] = OptimizerFactory.create(
                    args=args,
                    model_param_groups=model_param_groups[k],
                    bypass_params_creation=self._args.model_args.config.bypass_params_creation,
                )
            else:
                optimizers[k] = OptimizerFactory.create(
                    args=args,
                    model_param_groups=model_param_groups,
                    bypass_params_creation=self._args.model_args.config.bypass_params_creation,
                )

        return optimizers

    def _setup_lr_schedulers(self, optimizers: Mapping[str, torch.optim.Optimizer]):
        from xai_torch.core.training.sch.factory import LRSchedulerFactory

        # configure schedulers
        lr_schedulers = {}
        for k, sch in self._args.training_args.lr_schedulers.items():
            lr_schedulers[k] = LRSchedulerFactory.create(
                sch,
                optimizers[k],
                self._trainer.total_training_steps,
                self._trainer.warmup_steps,
                self._args.training_args.max_epochs,
            )
        return lr_schedulers

    def _setup_wd_schedulers(self, optimizers: Mapping[str, torch.optim.Optimizer]):
        from xai_torch.core.training.sch.factory import WdSchedulerFactory

        # configure schedulers
        wd_schedulers = {}
        for k, sch in self._args.training_args.wd_schedulers.items():
            wd_schedulers[k] = WdSchedulerFactory.create(
                sch,
                optimizers[k],
                self._args.training_args.max_epochs,
                self._trainer.steps_per_epoch,
            )
        return wd_schedulers

    def setup(self):
        import ignite.distributed as idist

        # setup optimizers
        self._optimizers = self._setup_optimizers()
        for k, opt in self._optimizers.items():
            self._optimizers[k] = idist.auto_optim(opt)

        # setup learning rate schedulers
        if self._args.training_args.lr_schedulers is not None:
            self._lr_schedulers = self._setup_lr_schedulers(
                self._optimizers,
            )

        # setup weight decay schedulers
        if self._args.training_args.wd_schedulers is not None:
            self._wd_schedulers = self._setup_wd_schedulers(
                self._optimizers,
            )

        # print information about optimizers and schedulers
        self._pretty_print_optimizers_schedulers()

    def _pretty_print_optimizers_schedulers(self):
        import ignite.distributed as idist

        from xai_torch.utilities.general import indent_string

        if idist.get_rank() == 0:
            # print information
            self._logger.info(f"Configured optimizers:")
            for k, v in self._optimizers.items():
                opt_str = indent_string(str(v), " " * 4)
                print(f"{k}:")
                print(f"{opt_str}")
            if self._lr_schedulers is not None:
                self._logger.info(f"Configured learning rate schedulers:")
                for k, v in self._lr_schedulers.items():
                    print(f"{k}:")
                    print(" " * 4, v)
            if self._wd_schedulers is not None:
                self._logger.info(f"Configured weight decay schedulers:")
                for k, v in self._wd_schedulers.items():
                    print(f"{k}: {v}")

    def get_checkpoint_map(self, checkpoint):
        # add optimizers to state
        if self._optimizers is not None:
            for k, opt in self._optimizers.items():
                checkpoint[f"opt_{k}"] = opt

        # add lr schedulers to state
        if self._lr_schedulers is not None:
            for k, sch in self._lr_schedulers.items():
                if sch is None:
                    continue
                checkpoint[f"lr_sch_{k}"] = sch

        # add wd schedulers to state
        if self._wd_schedulers is not None:
            for k, sch in self._wd_schedulers.items():
                if sch is None:
                    continue
                checkpoint[f"wd_sch_{k}"] = sch
