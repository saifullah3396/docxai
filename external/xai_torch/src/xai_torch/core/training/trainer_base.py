from __future__ import annotations

import logging
import math
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional, Sequence, Tuple, Union, cast

import torch
from ignite.contrib.handlers import TensorboardLogger
from ignite.engine import Engine
from torch.utils.data import DataLoader

from xai_torch.core.args import Arguments
from xai_torch.core.constants import DataKeys
from xai_torch.core.data.utilities.typing import TransformsDict
from xai_torch.core.factory.factory import DataAugmentationFactory, TrainValSamplerFactory
from xai_torch.core.models.xai_model import XAIModel
from xai_torch.core.training.constants import TrainingStage
from xai_torch.core.training.sch.schedulers import create_lr_scheduler_with_warmup
from xai_torch.utilities.logging_utils import DEFAULT_LOGGER_NAME

if TYPE_CHECKING:
    import torch

    from xai_torch.core.args import Arguments
    from xai_torch.core.data.data_modules.base import BaseDataModule
    from xai_torch.core.models.xai_model import XAIModel
    from xai_torch.core.training.optim_sch_handler import OptimizersSchedulersHandler


def log_eval_metrics(logger, epoch, elapsed, tag, metrics):
    metrics_output = "\n".join([f"\t{k}: {v}" for k, v in metrics.items()])
    logger.info(f"\nEpoch {epoch} - Evaluation time (seconds): {elapsed:.2f} - {tag} metrics:\n {metrics_output}")


class TrainerBase:
    @classmethod
    def initialize_training_engine(
        cls,
        args: Arguments,
        model: XAIModel,
        opt_sch_handler: OptimizersSchedulersHandler,
        device: Optional[Union[str, torch.device]] = torch.device("cpu"),
        scaler: Optional["torch.cuda.amp.GradScaler"] = None,
    ) -> Callable:

        if args.training_args.gradient_accumulation_steps <= 0:
            raise ValueError(
                "Gradient_accumulation_steps must be strictly positive. "
                "No gradient accumulation if the value set to one (default)."
            )

        from ignite.engine import Engine

        # get related arguments
        gradient_accumulation_steps = args.training_args.gradient_accumulation_steps
        non_blocking = args.training_args.non_blocking_tensor_conv

        if args.training_args.with_amp:
            try:
                from torch.cuda.amp import autocast
            except ImportError:
                raise ImportError("Please install torch>=1.6.0 to use amp_mode='amp'.")

            if scaler is None:
                from torch.cuda.amp.grad_scaler import GradScaler

                scaler = GradScaler(enabled=True)

        def training_step(engine: Engine, batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
            """
            Define the model training update step
            """

            import torch
            from ignite.utils import convert_tensor

            from xai_torch.core.constants import DataKeys

            # perform optimizers zero_grad() operation with gradient accumulation
            if (engine.state.iteration - 1) % gradient_accumulation_steps == 0:
                for opt in opt_sch_handler.optimizers.values():
                    opt.zero_grad()

            # setup model for training
            model.torch_model.train()

            # put batch to device
            batch = convert_tensor(batch, device=device, non_blocking=non_blocking)

            # forward pass
            model_output = model.torch_model.training_step(batch=batch)

            # make sure we get a dict from the model
            assert isinstance(model_output, dict), "Model must return an instance of dict."

            # get loss from the output dict
            loss = model_output[DataKeys.LOSS]

            # accumulate loss if required
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            # backward pass
            loss.backward()

            # perform optimizer update for correct gradient accumulation step
            if engine.state.iteration % gradient_accumulation_steps == 0:
                for opt in opt_sch_handler.optimizers.values():
                    opt.step()

            # if on the go training evaluation is required, detach data from the graph
            if args.training_args.eval_training:
                return_dict = {}
                for key, value in model_output.items():
                    if key == DataKeys.LOSS:
                        return_dict[key] = value.item()
                    elif isinstance(value, torch.Tensor):
                        return_dict[key] = value.detach()
                return return_dict

            return {DataKeys.LOSS: model_output[DataKeys.LOSS].item()}

        def training_step_with_amp(engine: Engine, batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
            """
            Define the model training update step
            """

            import torch
            from ignite.utils import convert_tensor

            from xai_torch.core.constants import DataKeys

            # perform optimizers zero_grad() operation with gradient accumulation
            if (engine.state.iteration - 1) % gradient_accumulation_steps == 0:
                for opt in opt_sch_handler.optimizers.values():
                    opt.zero_grad()

            # setup model for training
            model.torch_model.train()

            # put batch to device
            batch = convert_tensor(batch, device=device, non_blocking=non_blocking)

            with autocast(enabled=True):
                # forward pass
                model_output = model.torch_model.training_step(batch=batch)

                # make sure we get a dict from the model
                assert isinstance(model_output, dict), "Model must return an instance of dict."

                # get loss from the output dict
                loss = model_output[DataKeys.LOSS]

                # accumulate loss if required
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

            if scaler:
                scaler.scale(loss).backward()
                # perform optimizer update for correct gradient accumulation step
                if engine.state.iteration % gradient_accumulation_steps == 0:
                    for opt in opt_sch_handler.optimizers.values():
                        scaler.step(opt)
                        scaler.update()
            else:
                # backward pass
                loss.backward()

                # perform optimizer update for correct gradient accumulation step
                if engine.state.iteration % gradient_accumulation_steps == 0:
                    for opt in opt_sch_handler.optimizers.values():
                        opt.step()

            # if on the go training evaluation is required, detach data from the graph
            if args.training_args.eval_training:
                return_dict = {}
                for key, value in model_output.items():
                    if key == DataKeys.LOSS:
                        return_dict[key] = value.item()
                    elif isinstance(value, torch.Tensor):
                        return_dict[key] = value.detach()
                return return_dict

            return {DataKeys.LOSS: model_output[DataKeys.LOSS].item()}

        if args.training_args.with_amp:
            return Engine(training_step_with_amp)
        else:
            return Engine(training_step)

    @classmethod
    def initialize_validation_engine(
        cls,
        args: Arguments,
        model: XAIModel,
        device: Optional[Union[str, torch.device]] = torch.device("cpu"),
    ) -> Callable:
        from ignite.engine import Engine

        # get related arguments
        non_blocking = args.training_args.non_blocking_tensor_conv

        def validation_step(engine: Engine, batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
            """
            Define the model evaluation update step
            """

            import torch
            from ignite.utils import convert_tensor

            from xai_torch.core.training.constants import TrainingStage

            # ready model for evaluation
            model.torch_model.eval()
            with torch.no_grad():
                # put batch to device
                batch = convert_tensor(batch, device=device, non_blocking=non_blocking)

                # forward pass
                return model.torch_model.evaluation_step(batch, stage=TrainingStage.val)

        return Engine(validation_step)

    @classmethod
    def initialize_prediction_engine(
        cls,
        model: XAIModel,
        device: Optional[Union[str, torch.device]] = torch.device("cpu"),
    ) -> Callable:
        from ignite.engine import Engine

        def prediction_step(engine: Engine, batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
            """
            Define the model evaluation update step
            """

            import torch
            from ignite.utils import convert_tensor

            # ready model for evaluation
            model.torch_model.eval()
            with torch.no_grad():
                # put batch to device
                batch = convert_tensor(batch, device=device)

                # forward pass
                return model.torch_model.predict_step(batch)

        return Engine(prediction_step)

    @classmethod
    def initialize_test_engine(
        cls,
        args: Arguments,
        model: XAIModel,
        device: Optional[Union[str, torch.device]] = torch.device("cpu"),
    ) -> Callable:
        from ignite.engine import Engine

        # get related arguments
        non_blocking = args.training_args.non_blocking_tensor_conv

        def test_step(engine: Engine, batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
            """
            Define the model evaluation update step
            """

            import torch
            from ignite.utils import convert_tensor

            from xai_torch.core.training.constants import TrainingStage

            # ready model for evaluation
            model.torch_model.eval()
            with torch.no_grad():
                # put batch to device
                batch = convert_tensor(batch, device=device, non_blocking=non_blocking)

                # forward pass
                return model.torch_model.evaluation_step(batch, stage=TrainingStage.test)

        return Engine(test_step)

    @classmethod
    def _get_batches_per_epoch(cls, train_dataloader):
        return len(train_dataloader)

    @classmethod
    def _get_steps_per_epoch(cls, args, train_dataloader) -> int:
        """Total training steps inferred from datamodule and devices."""

        # batches = cls._get_batches_per_epoch(train_dataloader)
        # effective_accum = args.training_args.gradient_accumulation_steps * idist.get_world_size()
        # return batches // effective_accum
        return cls._get_batches_per_epoch(train_dataloader)

    @classmethod
    def _get_total_training_steps(cls, args, train_dataloader) -> int:
        """Total number of training steps inferred from datamodule and devices."""

        return cls._get_steps_per_epoch(args, train_dataloader) * args.training_args.max_epochs

    @classmethod
    def _get_warmup_steps(cls, args, train_dataloader):
        """Total number of warmup steps to be used."""

        return (
            args.training_args.warmup_steps
            if args.training_args.warmup_steps > 0
            else math.ceil(cls._get_total_training_steps(args, train_dataloader) * args.training_args.warmup_ratio)
        )

    @classmethod
    def configure_nan_callback(cls, args: Arguments, training_engine: Engine):
        from ignite.engine import Events

        # setup nan termination callback if required
        if args.training_args.stop_on_nan:
            from ignite.handlers import TerminateOnNan

            training_engine.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

    @classmethod
    def configure_cuda_cache_callback(cls, args: Arguments, training_engine: Engine):
        import torch
        from ignite.engine import Events

        # add cuda cache clear callback if required
        if torch.cuda.is_available() and args.training_args.clear_cuda_cache:
            from xai_torch.utilities.general import empty_cuda_cache

            training_engine.add_event_handler(Events.EPOCH_COMPLETED, empty_cuda_cache)

    @classmethod
    def configure_gpu_stats_callback(cls, args: Arguments, training_engine: Engine):
        import ignite.distributed as idist
        import torch

        # add gpu stats callback if required
        if idist.device() != torch.device("cpu") and args.training_args.log_gpu_stats:
            from ignite.contrib.metrics import GpuInfo
            from ignite.engine import Events

            GpuInfo().attach(
                training_engine,
                name="gpu",
                event_name=Events.ITERATION_COMPLETED(every=args.training_args.logging_steps),
            )

    @classmethod
    def configure_model_ema_callback(cls, args: Arguments, training_engine: Engine, model: XAIModel):
        if model.ema_handler is not None:
            logger = logging.getLogger(DEFAULT_LOGGER_NAME)
            logger.info(f"Attaching EMAHandler with following configuration: {args.training_args.model_ema_args}")

            from ignite.engine import Events

            model.ema_handler.attach(
                training_engine,
                name="ema_momentum",
                event=Events.ITERATION_COMPLETED(every=args.training_args.model_ema_args.update_every),
            )

    @classmethod
    def configure_early_stopping_callback(cls, args: Arguments, training_engine: Engine, validation_engine: Engine):
        # add gpu stats callback if required
        if args.training_args.early_stopping_args.monitored_metric != "":
            from ignite.engine import Events
            from ignite.handlers import Checkpoint, EarlyStopping

            cfg = args.training_args.early_stopping_args
            es_handler = EarlyStopping(
                patience=cfg.patience,
                score_function=Checkpoint.get_default_score_fn(cfg.monitored_metric, -1 if cfg.mode == "min" else 1.0),
                trainer=training_engine,
            )
            validation_engine.add_event_handler(Events.COMPLETED, es_handler)

    @classmethod
    def configure_train_sampler(cls, args: Arguments, training_engine: Engine, train_dataloader: DataLoader):
        import ignite.distributed as idist
        from torch.utils.data.distributed import DistributedSampler

        if idist.get_world_size() > 1:
            from ignite.engine import Engine, Events

            train_sampler = train_dataloader.sampler
            if not isinstance(train_sampler, DistributedSampler):
                raise TypeError("Train sampler should be torch DistributedSampler and have `set_epoch` method")

            @training_engine.on(Events.EPOCH_STARTED)
            def distrib_set_epoch(engine: Engine) -> None:
                cast(DistributedSampler, train_sampler).set_epoch(engine.state.epoch - 1)

        else:
            # check whether the correct training sample is being used
            if train_dataloader.sampler is not None and isinstance(train_dataloader.sampler, DistributedSampler):
                logger = logging.getLogger(DEFAULT_LOGGER_NAME)

                logger.warning(
                    "Argument train_sampler is a distributed sampler,"
                    " but either there is no distributed setting or world size is < 2. "
                    "Train sampler argument will be ignored",
                    UserWarning,
                )

    @classmethod
    def configure_tb_logger(
        cls,
        args: Arguments,
        training_engine: Engine,
        validation_engine: Engine,
        model: XAIModel,
        opt_sch_handler: OptimizersSchedulersHandler,
        tb_logger: TensorboardLogger,
    ):
        class_labels = None
        if hasattr(model._torch_model, "class_labels"):
            class_labels = model._torch_model.class_labels

        # setup tensorboard logging if required
        if args.training_args.log_to_tb is not None:
            from ignite.contrib.handlers import global_step_from_engine
            from ignite.engine import Events

            # attach handler to plot trainer's loss every 'logging_steps' iterations
            tb_logger.attach_output_handler(
                training_engine,
                event_name=Events.ITERATION_COMPLETED(every=args.training_args.logging_steps),
                tag=f"step",
                metric_names="all",
                class_labels=class_labels,
            )

            # attach the logger to the trainer to log optimizer's parameters, e.g. learning rate at every
            # 'logging_steps' iteration
            for param_name in ["lr", "weight_decay"]:
                for k, opt in opt_sch_handler.optimizers.items():
                    tb_logger.attach_opt_params_handler(
                        training_engine,
                        event_name=Events.ITERATION_STARTED(every=args.training_args.logging_steps),
                        optimizer=opt,
                        param_name=param_name,
                        tag=f"step/opt/{k}",
                    )

            if validation_engine is not None:
                # attach tb logger to validation engine
                tb_logger.attach_output_handler(
                    validation_engine,
                    event_name=Events.EPOCH_COMPLETED,
                    metric_names="all",
                    tag="epoch",
                    global_step_transform=global_step_from_engine(training_engine),
                    class_labels=class_labels,
                )

    @classmethod
    def configure_test_tb_logger(
        cls, args: Arguments, test_engine: Engine, model: XAIModel, tb_logger: TensorboardLogger, tag: str = "epoch"
    ):
        from ignite.engine import Events

        class_labels = None
        if hasattr(model._torch_model, "class_labels"):
            class_labels = model._torch_model.class_labels

        # attach tb logger to validation engine
        tb_logger.attach_output_handler(
            test_engine,
            event_name=Events.EPOCH_COMPLETED,
            metric_names="all",
            tag=tag,
            class_labels=class_labels,
        )

    @classmethod
    def configure_metrics(
        cls, args: Arguments, engine: Engine, model: XAIModel, stage: TrainingStage, prefix: str = ""
    ):
        if stage == TrainingStage.train and not args.training_args.eval_training:
            return

        if model.torch_model.metrics is not None:
            for k, metric in model.torch_model.metrics.items():
                metric().attach(engine, f"{stage}/{k}" if prefix == "" else f"{prefix}/{stage}/{k}")

    @classmethod
    def configure_lr_schedulers(
        cls,
        args: Arguments,
        training_engine: Engine,
        train_dataloader: DataLoader,
        opt_sch_handler: OptimizersSchedulersHandler,
        validation_engine: Engine = None,
    ):
        # setup learning rate schedulers as required in the arguments
        if opt_sch_handler.lr_schedulers is None:
            return

        from ignite.engine import Events
        from ignite.handlers import LRScheduler, ParamScheduler, ReduceLROnPlateauScheduler
        from torch.optim.lr_scheduler import StepLR

        for _, inner_sch in opt_sch_handler.lr_schedulers.items():
            if inner_sch is None:
                continue

            warmup_duration = cls._get_warmup_steps(args, train_dataloader)
            if warmup_duration > 0:
                if isinstance(inner_sch, StepLR):
                    sch = create_lr_scheduler_with_warmup(
                        inner_sch, warmup_start_value=0.0, warmup_duration=warmup_duration
                    )

                    # we want warmup on steps and step_lr on epochs, so we create two events first for steps
                    # and then for epochs
                    # Trigger scheduler on iteration_started events before reaching warmup_duration
                    combined_events = Events.ITERATION_STARTED(
                        event_filter=lambda _, __: training_engine.state.iteration <= warmup_duration
                    )

                    # Trigger scheduler on epoch_started events after the warm-up. Epochs are 1-based, thus we do 1 +
                    combined_events |= Events.EPOCH_STARTED(
                        event_filter=lambda _, __: training_engine.state.epoch
                        > 1 + warmup_duration / cls._get_steps_per_epoch(args, train_dataloader)
                    )

                    training_engine.add_event_handler(combined_events, sch)
                elif isinstance(inner_sch, ReduceLROnPlateauScheduler):
                    # we want warmup on steps and step_lr on epochs, so we create two events first for steps
                    # and then for epochs
                    sch = create_lr_scheduler_with_warmup(
                        inner_sch,
                        warmup_start_value=0.0,
                        warmup_duration=warmup_duration,
                    )
                    training_engine.add_event_handler(
                        Events.ITERATION_STARTED(
                            event_filter=lambda _, __: training_engine.state.iteration <= warmup_duration
                        ),
                        sch.schedulers[0],
                    )

                    # Trigger scheduler on epoch_started events after the warm-up. Epochs are 1-based, thus we do 1 +
                    combined_events = Events.COMPLETED | Events.COMPLETED(
                        event_filter=lambda _, __: training_engine.state.epoch
                        > 1 + warmup_duration / cls._get_steps_per_epoch(args, train_dataloader)
                    )

                    validation_engine.add_event_handler(combined_events, inner_sch)
                else:
                    sch = create_lr_scheduler_with_warmup(
                        inner_sch, warmup_start_value=0.0, warmup_duration=warmup_duration
                    )
                    training_engine.add_event_handler(Events.ITERATION_STARTED, sch)
            else:
                if not isinstance(inner_sch, ParamScheduler):
                    # convert scheduler to ignite scheduler
                    sch = LRScheduler(inner_sch)
                else:
                    sch = inner_sch

                if isinstance(inner_sch, StepLR):
                    training_engine.add_event_handler(Events.EPOCH_STARTED, sch)
                elif isinstance(inner_sch, ReduceLROnPlateauScheduler):
                    validation_engine.add_event_handler(Events.COMPLETED, sch)
                else:
                    training_engine.add_event_handler(Events.ITERATION_STARTED, sch)

    @classmethod
    def configure_wd_schedulers(
        cls,
        args: Arguments,
        training_engine: Engine,
        opt_sch_handler: OptimizersSchedulersHandler,
        validation_engine: Engine = None,
    ):
        # setup learning rate schedulers as required in the arguments
        if opt_sch_handler.wd_schedulers is None or all(sch is None for sch in opt_sch_handler.wd_schedulers.values()):
            return
        from ignite.engine import Events

        # handle weight decay
        def update_weight_decays():
            for key, opt in opt_sch_handler.optimizers.items():
                opt_wd_schs = opt_sch_handler.wd_schedulers[key]
                if opt_wd_schs.d is None:
                    continue
                for pg_idx, pg in enumerate(opt.param_groups):
                    group_name = pg["name"]
                    if group_name in opt_wd_schs.d:
                        pg["weight_decay"] = opt_wd_schs.d[group_name].step()

        training_engine.add_event_handler(Events.ITERATION_STARTED, update_weight_decays)

    @classmethod
    def configure_model_checkpoints(
        cls,
        args: Arguments,
        training_engine: Engine,
        model: XAIModel,
        opt_sch_handler: OptimizersSchedulersHandler,
        output_dir: str,
        validation_engine: Optional[Engine] = None,
        do_val: bool = True,
    ):
        # setup checkpoint saving if required
        if args.training_args.enable_checkpointing:
            from ignite.engine import Events
            from ignite.handlers.checkpoint import BaseSaveHandler, DiskSaver

            checkpoint_map = {"training_engine": training_engine}
            model.get_checkpoint_map(checkpoint_map)

            # add optimizers and lr/wd scheduler states to checkpoint_map
            opt_sch_handler.get_checkpoint_map(checkpoint_map)

            # if only to save weights, remove all other keys
            if args.training_args.model_checkpoint_config.save_weights_only:
                for k in list(checkpoint_map.keys()):
                    if k not in ["training_engine", "model"]:
                        checkpoint_map.pop(k)

            model_checkpoint_config = args.training_args.model_checkpoint_config
            checkpoint_dir = output_dir / model_checkpoint_config.dir
            save_handler = DiskSaver(
                checkpoint_dir,
                require_empty=False,
            )

            if model_checkpoint_config.save_per_epoch:
                from ignite.handlers import Checkpoint

                checkpoint_handler = Checkpoint(
                    checkpoint_map,
                    cast(Union[Callable, BaseSaveHandler], save_handler),
                    filename_prefix=model_checkpoint_config.name_prefix,
                    global_step_transform=lambda *_: training_engine.state.epoch,
                )
                training_engine.add_event_handler(
                    Events.EPOCH_COMPLETED(every=model_checkpoint_config.save_every_iters), checkpoint_handler
                )
            else:
                checkpoint_handler = Checkpoint(
                    checkpoint_map,
                    cast(Union[Callable, BaseSaveHandler], save_handler),
                    filename_prefix=model_checkpoint_config.name_prefix,
                    n_saved=model_checkpoint_config.n_saved,
                )
                training_engine.add_event_handler(
                    Events.ITERATION_COMPLETED(every=model_checkpoint_config.save_every_iters), checkpoint_handler
                )

        if args.training_args.resume_from_checkpoint:
            import logging

            import torch

            from xai_torch.core.training.utilities import find_resume_checkpoint

            logger = logging.getLogger(DEFAULT_LOGGER_NAME)

            resume_checkpoint_path = find_resume_checkpoint(
                args.training_args.resume_checkpoint_file,
                checkpoint_dir,
                args.training_args.load_best_checkpoint_resume,
            )
            if resume_checkpoint_path is not None:
                resume_checkpoint = torch.load(resume_checkpoint_path, map_location="cpu")
                for k in list(checkpoint_map.keys()):
                    if k not in list(resume_checkpoint.keys()):
                        logger.warning(f"Object {k} not found in the resume checkpoint_map.")
                        del checkpoint_map[k]

                Checkpoint.load_objects(to_load=checkpoint_map, checkpoint=resume_checkpoint)

        if validation_engine is not None and do_val and model_checkpoint_config.monitored_metric is not None:
            from ignite.contrib.handlers import global_step_from_engine

            best_model_saver = Checkpoint(
                checkpoint_map,
                save_handler=DiskSaver(
                    checkpoint_dir,
                    require_empty=False,
                ),
                filename_prefix="best",
                # filename_pattern="{filename_prefix}_{name}.{ext}",
                n_saved=model_checkpoint_config.n_best_saved,
                global_step_transform=global_step_from_engine(training_engine),
                score_name=model_checkpoint_config.monitored_metric.replace("/", "-"),
                score_function=Checkpoint.get_default_score_fn(
                    model_checkpoint_config.monitored_metric, -1 if model_checkpoint_config.mode == "min" else 1.0
                ),
            )
            validation_engine.add_event_handler(
                Events.COMPLETED,
                best_model_saver,
            )

    @classmethod
    def configure_running_avg_logging(cls, args: Arguments, engine: Engine, stage: TrainingStage):
        from ignite.metrics import RunningAverage

        def output_transform(x: Any, index: int, name: str) -> Any:
            import numbers

            import torch

            if isinstance(x, Mapping):
                return x[name]
            elif isinstance(x, Sequence):
                return x[index]
            elif isinstance(x, (torch.Tensor, numbers.Number)):
                return x
            else:
                raise TypeError(
                    "Unhandled type of update_function's output. "
                    f"It should either mapping or sequence, but given {type(x)}"
                )

        # add loss as a running average metric
        for i, n in enumerate([DataKeys.LOSS]):
            RunningAverage(output_transform=partial(output_transform, index=i, name=n), epoch_bound=False).attach(
                engine, f"{stage}/{n}"
            )

    @classmethod
    def configure_progress_bars(
        cls,
        args: Arguments,
        engine: Engine,
        stage=TrainingStage.train,
        opt_sch_handler: OptimizersSchedulersHandler = None,
    ):
        from ignite.engine import Events

        from xai_torch.core.training.progress_bar import XAIProgressBar

        if stage == TrainingStage.train:
            if opt_sch_handler is None:
                raise ValueError("opt_sch_handler is required for TrainingStage=Train.")

            XAIProgressBar(persist=False, desc="Training").attach(
                engine,
                metric_names="all",
                optimizers=opt_sch_handler.optimizers,
                optimizer_params=["lr"],
                event_name=Events.ITERATION_COMPLETED(every=args.training_args.logging_steps),
            )

            XAIProgressBar(persist=True, desc="Training", ncols=0).attach(
                engine,
                metric_names="all",
                optimizers=opt_sch_handler.optimizers,
                optimizer_params=["lr"],
                event_name=Events.EPOCH_STARTED,
                closing_event_name=Events.COMPLETED,
            )
        elif stage == TrainingStage.val:
            XAIProgressBar(desc="Validation", persist=False).attach(engine)
        elif stage == TrainingStage.test:
            XAIProgressBar(desc="Testing", persist=False).attach(engine)

    @classmethod
    def attach_validator(
        cls,
        args: Arguments,
        training_engine: Engine,
        validation_engine: Engine,
        val_dataloader,
    ):
        from ignite.engine import Events

        logger = logging.getLogger(DEFAULT_LOGGER_NAME)

        def validate(engine):
            epoch = training_engine.state.epoch
            state = validation_engine.run(val_dataloader)
            log_eval_metrics(logger, epoch, state.times["COMPLETED"], TrainingStage.val, state.metrics)

        training_engine.add_event_handler(
            Events.EPOCH_COMPLETED(every=args.training_args.eval_every_n_epochs)
            | Events.COMPLETED,  # | Events.STARTED,
            validate,
        )

    @classmethod
    def configure_training_engine(
        cls,
        args: Arguments,
        training_engine: Engine,
        model: XAIModel,
        opt_sch_handler: OptimizersSchedulersHandler,
        output_dir: str,
        validation_engine: Optional[Engine] = None,
        tb_logger: Optional[TensorboardLogger] = None,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloader: Optional[DataLoader] = None,
        do_val: bool = True,
    ) -> None:
        import ignite.distributed as idist

        # configure training engine
        cls.configure_train_sampler(args=args, training_engine=training_engine, train_dataloader=train_dataloader)
        cls.configure_nan_callback(args=args, training_engine=training_engine)
        cls.configure_cuda_cache_callback(args=args, training_engine=training_engine)
        cls.configure_gpu_stats_callback(args=args, training_engine=training_engine)
        cls.configure_model_ema_callback(args=args, training_engine=training_engine, model=model)
        cls.configure_metrics(args=args, engine=training_engine, model=model, stage=TrainingStage.train)
        cls.configure_wd_schedulers(args=args, training_engine=training_engine, opt_sch_handler=opt_sch_handler)
        if validation_engine is None:
            cls.configure_lr_schedulers(
                args=args,
                training_engine=training_engine,
                train_dataloader=train_dataloader,
                opt_sch_handler=opt_sch_handler,
            )
        cls.configure_model_checkpoints(
            args=args,
            training_engine=training_engine,
            model=model,
            opt_sch_handler=opt_sch_handler,
            output_dir=output_dir,
            validation_engine=validation_engine,
            do_val=do_val,
        )
        cls.configure_running_avg_logging(args=args, engine=training_engine, stage=TrainingStage.train)
        if idist.get_rank() == 0:
            cls.configure_progress_bars(
                args=args, engine=training_engine, opt_sch_handler=opt_sch_handler, stage=TrainingStage.train
            )

        # configure validation engine
        if validation_engine is not None:
            if val_dataloader is None:
                raise ValueError("val_loader is required for using validation_engine.")
            elif val_dataloader.dataset is None:
                logger = logging.getLogger(DEFAULT_LOGGER_NAME)
                logger.warning("No validation dataset found.")
            else:
                cls.configure_metrics(args=args, engine=validation_engine, model=model, stage=TrainingStage.val)
                cls.configure_lr_schedulers(
                    args=args,
                    training_engine=training_engine,
                    train_dataloader=train_dataloader,
                    opt_sch_handler=opt_sch_handler,
                    validation_engine=validation_engine,
                )
                cls.configure_progress_bars(
                    args=args, engine=validation_engine, opt_sch_handler=opt_sch_handler, stage=TrainingStage.val
                )
                cls.configure_running_avg_logging(args=args, engine=validation_engine, stage=TrainingStage.val)
                cls.configure_early_stopping_callback(
                    args=args, training_engine=training_engine, validation_engine=validation_engine
                )
                cls.attach_validator(
                    args=args,
                    training_engine=training_engine,
                    validation_engine=validation_engine,
                    val_dataloader=val_dataloader,
                )

        if idist.get_rank() == 0 and tb_logger is not None:
            # configure tensorboard
            cls.configure_tb_logger(
                args=args,
                training_engine=training_engine,
                validation_engine=validation_engine,
                model=model,
                opt_sch_handler=opt_sch_handler,
                tb_logger=tb_logger,
            )

    @classmethod
    def configure_test_engine(
        cls,
        args: Arguments,
        test_engine: Engine,
        model: XAIModel,
        output_dir: str,
        tb_logger: Optional[TensorboardLogger] = None,
        checkpoint_type: str = "last",
        load_checkpoint: bool = True,
    ) -> None:
        import ignite.distributed as idist
        import torch
        from ignite.engine import Events
        from ignite.handlers import Checkpoint

        from xai_torch.core.training.utilities import find_test_checkpoint

        # configure model checkpoint_map
        model_checkpoint_config = args.training_args.model_checkpoint_config
        checkpoint_dir = output_dir / model_checkpoint_config.dir
        if load_checkpoint:
            if checkpoint_type in ["last", "ema"]:
                checkpoint = find_test_checkpoint(
                    args.training_args.test_checkpoint_file, checkpoint_dir, load_best=False
                )
            if checkpoint_type == "best":
                checkpoint = find_test_checkpoint(
                    args.training_args.test_checkpoint_file, checkpoint_dir, load_best=True
                )
            if checkpoint is not None:
                checkpoint_map = {}
                model.get_checkpoint_map(checkpoint_map)
                test_checkpoint = torch.load(checkpoint, map_location="cpu")
                Checkpoint.load_objects(to_load=checkpoint_map, checkpoint=test_checkpoint)

        if checkpoint_type == "ema":
            model.activate_ema = True
        else:
            model.activate_ema = False

        # configure test engine
        cls.configure_metrics(
            args=args, engine=test_engine, model=model, stage=TrainingStage.test, prefix=checkpoint_type
        )
        if idist.get_rank() == 0:
            cls.configure_progress_bars(args=args, engine=test_engine, stage=TrainingStage.test)
        cls.configure_running_avg_logging(args=args, engine=test_engine, stage=TrainingStage.test)

        if idist.get_rank() == 0:
            logger = logging.getLogger(DEFAULT_LOGGER_NAME)

            def log_test_metrics(engine):
                state = engine.state
                log_eval_metrics(
                    logger,
                    state.epoch,
                    state.times["COMPLETED"],
                    TrainingStage.test,
                    state.metrics,
                )

            test_engine.add_event_handler(
                Events.EPOCH_COMPLETED,
                log_test_metrics,
            )

        if idist.get_rank() == 0 and tb_logger is not None:
            # configure tensorboard
            cls.configure_test_tb_logger(args=args, test_engine=test_engine, model=model, tb_logger=tb_logger)

    @classmethod
    def setup_datamodule(
        cls, args: Arguments, rank: int = 0, stage: TrainingStage = TrainingStage.train, override_collate_fns=None
    ) -> BaseDataModule:
        """
        Initializes the datamodule for training.
        """
        import ignite.distributed as idist

        from xai_torch.core.factory.factory import DataCacherFactory, DataModuleFactory
        from xai_torch.core.models.factory import ModelFactory

        # get data collator required for the model
        model_class = ModelFactory.get_model_class(args.model_args)
        collate_fns = override_collate_fns if override_collate_fns is not None else model_class.get_data_collators(args)

        # define data transforms according to the configuration
        transforms = TransformsDict()
        if args.data_args.train_aug_args is not None:
            transforms.train = DataAugmentationFactory.create(
                args.data_args.train_aug_args.strategy,
                args.data_args.train_aug_args.config,
                args.data_args.train_aug_args.keys,
            )

        if args.data_args.eval_aug_args is not None:
            eval_augs = DataAugmentationFactory.create(
                args.data_args.eval_aug_args.strategy,
                args.data_args.eval_aug_args.config,
                args.data_args.eval_aug_args.keys,
            )
            transforms.val = eval_augs
            transforms.test = eval_augs

        # setup train/val sampler
        train_val_sampler = TrainValSamplerFactory.create(
            args.data_args.train_val_sampling_args.strategy, **args.data_args.train_val_sampling_args.config
        )

        # setup data cacher
        data_cacher_wrapper = DataCacherFactory.create(
            args.data_args.data_cacher_args.strategy, **args.data_args.data_cacher_args.config
        )

        # see if tokenization is required
        tokenizer = None
        if args.data_args.data_tokenizer_args:
            from xai_torch.core.factory.factory import TokenizerFactory

            tokenizer = TokenizerFactory.create(args.data_args.data_tokenizer_args)

        # initialize data module generator function
        datamodule = DataModuleFactory.create(
            args.data_args.dataset_name,
            args.data_args.dataset_dir,
            collate_fns=collate_fns,
            transforms=transforms,
            train_val_sampler=train_val_sampler,
            data_cacher_wrapper=data_cacher_wrapper,
            tokenizer=tokenizer,
            show_transforms=args.data_args.show_transforms,
        )

        # only download dataset on rank 0, all other ranks wait here for rank 0 to load the datasets
        if rank > 0:
            idist.barrier()

        # we manually prepare data and call setup here so dataset related properties can be initalized.
        datamodule.setup(
            quiet=False,
            stage=stage,
            do_train=args.general_args.do_train,
            max_train_samples=args.data_args.data_loader_args.max_train_samples,
            max_val_samples=args.data_args.data_loader_args.max_val_samples,
            max_test_samples=args.data_args.data_loader_args.max_test_samples,
            use_test_set_for_val=args.data_args.data_loader_args.use_test_set_for_val,
        )

        if rank == 0:
            idist.barrier()

        # Todo: Fix with new code
        # call prepare data on start so that initial variables are set correctly
        # if args.general_args.debug_data:
        #     datamodule.setup()
        #     for stage in list(TrainingStage):
        #         if stage == TrainingStage.predict:
        #             continue
        #         logger.info(f"Visualizing data batch for training stage = [{stage}]")
        #         image_grid = datamodule.show_batch(stage=stage, show=False)
        #         for pl_logger in pl_loggers:
        #             if isinstance(pl_logger, TensorBoardLogger):
        #                 writer = pl_logger.experiment
        #                 writer.add_image(f"Images for stage = {stage.value}", image_grid)

        return datamodule

    @classmethod
    def setup_model(
        cls,
        args: Arguments,
        tb_logger: TensorboardLogger,
        summarize: bool = False,
        stage: TrainingStage = TrainingStage.train,
        gt_metadata: Optional[dict] = None,
        checkpoint: Optional[str] = None,
        strict: bool = False,
    ) -> XAIModel:
        """
        Initializes the model for training.
        """
        from xai_torch.core.models.factory import ModelFactory

        # setup model
        model = ModelFactory.create(args, checkpoint=checkpoint, tb_logger=tb_logger, gt_metadata=gt_metadata, strict=strict)
        model.setup(stage=stage)

        # generate model summary
        if summarize:
            model.summarize()

        return model

    @classmethod
    def setup_training_engine(
        cls, args, model, opt_sch_handler, train_dataloader, val_dataloader, output_dir, tb_logger, device, do_val=True
    ):
        # setup training engine
        training_engine = cls.initialize_training_engine(
            args=args, model=model, opt_sch_handler=opt_sch_handler, device=device
        )

        validation_engine = None
        if do_val:
            # setup validation engine
            validation_engine = cls.initialize_validation_engine(args=args, model=model, device=device)

        # configure training and validation engines
        cls.configure_training_engine(
            args=args,
            training_engine=training_engine,
            model=model,
            opt_sch_handler=opt_sch_handler,
            output_dir=output_dir,
            tb_logger=tb_logger,
            train_dataloader=train_dataloader,
            validation_engine=validation_engine,
            val_dataloader=val_dataloader,
            do_val=do_val,
        )

        # add training hooks from the model
        model.add_training_hooks(training_engine)

        return training_engine, validation_engine

    @classmethod
    def setup_test_engine(
        cls, args, model, test_dataloader, output_dir, tb_logger, device, checkpoint_type: str = "last"
    ):
        # setup training engine
        test_engine = cls.initialize_test_engine(args=args, model=model, device=device)

        # configure training and validation engines
        cls.configure_test_engine(
            args=args,
            test_engine=test_engine,
            model=model,
            output_dir=output_dir,
            tb_logger=tb_logger,
            checkpoint_type=checkpoint_type,
        )

        return test_engine
