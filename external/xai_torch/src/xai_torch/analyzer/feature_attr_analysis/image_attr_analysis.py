"""
Defines the feature attribution generation task.
"""

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import ignite.distributed as idist
import torch
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Events
from tensorboardX import SummaryWriter
from torch.utils.data import Subset
from xai_torch.analyzer.feature_attr_analysis.attribution_wrappers import (
    ATTRIBUTIONS_REGISTRY,
    CaptumAttrWrapperBase,
    DeepShap,
)
from xai_torch.analyzer.feature_attr_analysis.evaluators import (
    AOPCEvaluator,
    AttributionEvaluator,
    ContinuityEvaluator,
    DataSaverHandler,
    EvaluatorEvents,
    EvaluatorKeys,
    FeaturePerturbationEvaluator,
    InfidelityEvaluator,
    SensitivityEvaluator,
)
from xai_torch.analyzer.feature_attr_analysis.utilities import (
    fix_model_inplace,
    generate_image_baselines,
    wrap_model_output,
)
from xai_torch.analyzer.feature_perturbers.image_feature_perturbor import ImageFeaturePerturber
from xai_torch.core.analyzer.decorators import register_analyzer_task
from xai_torch.core.analyzer.tasks.base import AnalyzerTask
from xai_torch.core.analyzer.tasks.base_config import AnalyzerTaskConfig
from xai_torch.core.constants import DataKeys
from xai_torch.core.data.data_modules.base import BaseDataModule
from xai_torch.core.models.utilities.data_collators import PassThroughCollator
from xai_torch.core.models.xai_model import XAIModel
from xai_torch.core.training.constants import TrainingStage
from xai_torch.utilities.logging_utils import DEFAULT_LOGGER_NAME


@register_analyzer_task("image_attr_analysis")
class ImageAttrAnalysis(AnalyzerTask):
    @dataclass
    class Config(AnalyzerTaskConfig):
        methods: Mapping[str, dict] = field(
            default=None,
        )
        evaluators: Mapping[str, dict] = field(
            default_factory=lambda: {
                EvaluatorKeys.ATTR_MAP: {},
                EvaluatorKeys.CONTINUITY: {},
                EvaluatorKeys.INFIDELITY: {},
                EvaluatorKeys.SENSITIVITY: {},
                EvaluatorKeys.FEATURE_PERTURBATION: {},
            }
        )
        target_baselines_per_label: int = 10
        random_runs: int = 10
        start_idx: int = None
        end_idx: int = None

        # def __post_init__(self):
        #     # if self.methods is None or len(self.methods.keys()) == 0:
        #     #     raise AttributeError(
        #     #         f"Config must define target attribution methods along with their config under 'methods' parameter. "
        #     #         f"Choices: [{ATTRIBUTIONS_REGISTRY.keys()}]"
        #     #     )

        #     # for method in self.methods.keys():
        #     #     if method not in ATTRIBUTIONS_REGISTRY.keys():
        #     #         raise AttributeError(f"Attribution method [{method}] is not supported")

    def _setup_trainer_base(self):
        from xai_torch.analyzer.feature_attr_analysis.trainer_base import FeatureAttrTrainerBase

        return FeatureAttrTrainerBase

    def _setup_datamodule(self, stage: TrainingStage = TrainingStage.train) -> BaseDataModule:
        datamodule = self._trainer_base.setup_datamodule(
            self._args, rank=self._rank, stage=stage, override_collate_fns=self.get_collate_fns()
        )

        test_collate_fn = datamodule._collate_fns.test
        datamodule._collate_fns.test = PassThroughCollator()
        self._test_collate_fn = test_collate_fn

        return datamodule

    def setup(self, task_name: str):
        # setup base training functionality
        self._trainer_base = self._setup_trainer_base()

        # setup training
        self._setup_analysis(task_name)

        # setup datamodule
        self._datamodule = self._setup_datamodule(stage=None)

        # remove transforms and store locally
        if isinstance(self._datamodule.test_dataset, Subset):
            transforms = self._datamodule.test_dataset.dataset.transforms
            self._datamodule.test_dataset.dataset.transforms = None
        else:
            transforms = self._datamodule.test_dataset.transforms
            self._datamodule.test_dataset.transforms = None
        self._transforms = [transforms] if not isinstance(transforms, list) else transforms

        # generate baselines as they are needed for some attributions
        self._baselines = generate_image_baselines(
            self._datamodule,
            self._args.data_args,
            target_baselines_per_label=self.config.target_baselines_per_label,
        )

    def get_collate_fns(self):
        import torch
        from xai_torch.core.data.utilities.typing import CollateFnDict
        from xai_torch.core.models.utilities.data_collators import BatchToTensorDataCollator

        # train
        data_key_type_map = {
            DataKeys.INDEX: torch.int,
            DataKeys.IMAGE: torch.float,
            DataKeys.LABEL: torch.long,
        }
        collate_fn = BatchToTensorDataCollator(
            data_key_type_map=data_key_type_map,
        )

        # test
        test_data_key_type_map = {
            DataKeys.IMAGE: torch.float,
            DataKeys.LABEL: torch.long,
        }
        test_collate_fn = BatchToTensorDataCollator(
            data_key_type_map=test_data_key_type_map,
        )

        # initialize the data collators for bert grid based word classification
        return CollateFnDict(train=collate_fn, val=collate_fn, test=test_collate_fn)

    def _configure_attr_computation_engine(
        self,
        model: XAIModel,
        method: str,
        attr: CaptumAttrWrapperBase,
        model_ckpt_name: str,
        summary_writer: SummaryWriter,
        max_data_required: int,
    ):
        if self.config.start_idx is not None and self.config.end_idx is not None:
            # setup the file where all outputs are stored per image and read it if it exists
            output_file = (
                self._output_dir / method / f"{model_ckpt_name}_{self.config.start_idx}_{self.config.end_idx}.h5"
            )
        else:
            output_file = self._output_dir / method / f"{model_ckpt_name}.h5"

        # define compute step
        attr_computation_engine = self._trainer_base.initialize_prediction_engine(
            model, self._transforms, self._test_collate_fn, device=idist.device(), orig_resize=(1024, 1024)
        )
        attr_computation_engine.register_events(*EvaluatorEvents)

        # add evaluators based on required evaluators
        attached_evaluators = {}
        all_evaluations_computed = True
        for evaluator_name, kwargs in self.config.evaluators.items():
            if evaluator_name == EvaluatorKeys.ATTR_MAP:
                import copy
                kwargs_copy = copy.deepcopy(kwargs)

                # add a handler for computing attribution maps
                if 'attr_dir' in kwargs:
                    attr_output_file = Path(kwargs_copy.pop('attr_dir')) / method / f"{model_ckpt_name}.h5"
                    save_eval=False
                else:
                    attr_output_file = output_file
                    save_eval=True
                # add a handler for computing attribution maps
                evaluator = AttributionEvaluator(
                    attr_output_file, summary_writer, attr, self._baselines, visualization_tag=model_ckpt_name, save_eval=save_eval, **kwargs_copy
                )
                evaluator.attach(attr_computation_engine, name=EvaluatorKeys.ATTR_MAP, event=Events.ITERATION_COMPLETED)
            elif evaluator_name == EvaluatorKeys.CONTINUITY:
                # add a handler for computing attribution maps
                evaluator = ContinuityEvaluator(output_file, summary_writer, EvaluatorKeys.ATTR_MAP, **kwargs)
                evaluator.attach(
                    attr_computation_engine, name=EvaluatorKeys.CONTINUITY, event=EvaluatorEvents.ATTR_MAP_COMPUTED
                )
            elif evaluator_name == EvaluatorKeys.INFIDELITY:
                # add a handler for computing attribution maps
                evaluator = InfidelityEvaluator(
                    output_file, summary_writer, model.torch_model, EvaluatorKeys.ATTR_MAP, **kwargs
                )
                evaluator.attach(
                    attr_computation_engine, name=EvaluatorKeys.INFIDELITY, event=EvaluatorEvents.ATTR_MAP_COMPUTED
                )
            elif evaluator_name == EvaluatorKeys.SENSITIVITY:
                # add a handler for computing attribution maps
                evaluator = SensitivityEvaluator(output_file, summary_writer, attr, self._baselines, **kwargs)
                evaluator.attach(
                    attr_computation_engine, name=EvaluatorKeys.SENSITIVITY, event=Events.ITERATION_COMPLETED
                )
            elif evaluator_name == EvaluatorKeys.FEATURE_PERTURBATION:
                # create the image perturber
                kwargs = copy.copy(kwargs)
                perturbation_kwargs = kwargs.pop("perturbation_kwargs")
                image_feature_perturber = ImageFeaturePerturber(
                    collate_fn=self._test_collate_fn, transforms=self._transforms, **perturbation_kwargs
                )

                # add a handler for computing attribution maps
                kwargs["is_random_run"] = False
                evaluator = FeaturePerturbationEvaluator(  # here we sent the original unwrapped model
                    output_file,
                    summary_writer,
                    model.torch_model._model,
                    EvaluatorKeys.ATTR_MAP,
                    feature_perturber=image_feature_perturber,
                    **kwargs,
                )
                evaluator.attach(
                    attr_computation_engine,
                    name=EvaluatorKeys.FEATURE_PERTURBATION,
                    event=EvaluatorEvents.ATTR_MAP_COMPUTED,
                )
            elif evaluator_name == EvaluatorKeys.AOPC:
                # add a handler for computing attribution maps
                evaluator = AOPCEvaluator(output_file, summary_writer, **kwargs)
                evaluator.attach(
                    attr_computation_engine,
                    name=EvaluatorKeys.AOPC,
                    event=EvaluatorEvents.FEATURE_PERTURBATION_COMPUTED,
                )
            attached_evaluators[evaluator_name] = evaluator
            if not evaluator.finished(evaluator_name, max_data_required):
                all_evaluations_computed = False

        if all_evaluations_computed:
            return None

        # add a handler for computing attribution maps
        attr_computation_engine.add_event_handler(
            Events.ITERATION_COMPLETED,
            DataSaverHandler(output_file, attached_evaluators=attached_evaluators),
        )

        # attach progress bar
        ProgressBar().attach(attr_computation_engine)

        return attr_computation_engine

    def _configure_random_computation_engine(
        self,
        model: XAIModel,
        model_ckpt_name: str,
        summary_writer: SummaryWriter,
        output_file: Path,
    ):
        if EvaluatorKeys.FEATURE_PERTURBATION not in self.config.evaluators.keys():
            return

        # define compute step
        attr_computation_engine = self._trainer_base.initialize_prediction_engine(
            model, self._transforms, self._test_collate_fn, device=idist.device(),  orig_resize=(1024, 1024)
        )
        attr_computation_engine.register_events(*EvaluatorEvents)

        # add evaluators based on required evaluators
        attached_evaluators = {}
        for evaluator_name, kwargs in self.config.evaluators.items():
            evaluator = None
            if evaluator_name == EvaluatorKeys.FEATURE_PERTURBATION:
                # create the image perturber
                kwargs = copy.copy(kwargs)
                perturbation_kwargs = kwargs.pop("perturbation_kwargs")
                image_feature_perturber = ImageFeaturePerturber(
                    collate_fn=self._test_collate_fn, transforms=self._transforms, **perturbation_kwargs
                )

                # add a handler for computing attribution maps
                kwargs["is_random_run"] = True
                evaluator = FeaturePerturbationEvaluator(  # here we sent the original unwrapped model
                    output_file,
                    summary_writer,
                    model.torch_model._model,
                    EvaluatorKeys.ATTR_MAP,
                    feature_perturber=image_feature_perturber,
                    **kwargs,
                )
                evaluator.attach(
                    attr_computation_engine,
                    name=EvaluatorKeys.FEATURE_PERTURBATION,
                    event=Events.ITERATION_COMPLETED,
                )
            elif evaluator_name == EvaluatorKeys.AOPC:
                # add a handler for computing attribution maps
                evaluator = AOPCEvaluator(output_file, summary_writer, **kwargs)
                evaluator.attach(
                    attr_computation_engine,
                    name=EvaluatorKeys.AOPC,
                    event=EvaluatorEvents.FEATURE_PERTURBATION_COMPUTED,
                )

            if evaluator is not None:
                attached_evaluators[evaluator_name] = evaluator

        # add a handler for computing attribution maps
        attr_computation_engine.add_event_handler(
            Events.ITERATION_COMPLETED,
            DataSaverHandler(output_file, attached_evaluators=attached_evaluators),
        )

        # attach progress bar
        ProgressBar().attach(attr_computation_engine)

        return attr_computation_engine

    def _get_attr(self, model: XAIModel, method: str, method_kwargs: dict):
        # get wrapped attribution method from captum
        attr = ATTRIBUTIONS_REGISTRY.get(method, None)
        if attr is None:
            raise ValueError(f"Feature attribution method [{method}] not supported.")

        # we initialize the method for each model checkpoint
        if attr == DeepShap:
            # allocate gpu
            if torch.cuda.is_available():
                model.model_to_device()
                self._baselines = self._baselines.cuda()

            # init attribution method
            attr = attr(model.torch_model, background_samples=self._baselines, **method_kwargs)

            # deallocate gpu
            model.model_to_device("cpu")
            self._baselines = self._baselines.cpu()
        else:
            # init attribution method
            attr = attr(model.torch_model, **method_kwargs)

        return attr

    def _perform_attr_analysis(self, model_ckpt_name: str, model: XAIModel, method: str, **method_kwargs):
        logger = logging.getLogger(DEFAULT_LOGGER_NAME)

        # log info
        logger.info(f"Applying method [{method}] to the dataset for model [{model_ckpt_name}].")

        # define summary writer for this method
        summary_writer = SummaryWriter(log_dir=self._output_dir / method)

        # get attrs
        attr = self._get_attr(model=model, method=method, method_kwargs=method_kwargs)

        # setup dataloaders
        if self.config.start_idx is not None and self.config.end_idx is not None:
            test_dataloader = self._datamodule.test_dataloader_indices(
                self.config.start_idx,
                self.config.end_idx,
                attr.batch_size,
                dataloader_num_workers=self._args.data_args.data_loader_args.dataloader_num_workers,
                pin_memory=self._args.data_args.data_loader_args.pin_memory,
            )
        else:
            test_dataloader = self._datamodule.test_dataloader(
                attr.batch_size,
                dataloader_num_workers=self._args.data_args.data_loader_args.dataloader_num_workers,
                pin_memory=self._args.data_args.data_loader_args.pin_memory,
            )

        # perform attribution analysis
        max_data_required = len(test_dataloader.dataset)
        attr_computation_engine = self._configure_attr_computation_engine(
            model, method, attr, model_ckpt_name, summary_writer, max_data_required
        )

        # run engine
        if attr_computation_engine is not None:
            attr_computation_engine.run(test_dataloader)

        summary_writer.close()

    def _perform_random_runs_analysis(self, model_ckpt_name: str, model: XAIModel):
        # setup the file where all outputs are stored per image and read it if it exists
        output_file = self._output_dir / "random" / f"{model_ckpt_name}.h5"
        if output_file.exists():
            return

        logger = logging.getLogger(DEFAULT_LOGGER_NAME)

        # log info
        logger.info(f"Applying random runs now to the dataset for model [{model_ckpt_name}].")

        # define summary writer for this method
        summary_writer = SummaryWriter(log_dir=self._output_dir / "random")

        # setup dataloaders
        test_dataloader = self._datamodule.test_dataloader(
            self._args.data_args.data_loader_args.per_device_eval_batch_size,
            dataloader_num_workers=self._args.data_args.data_loader_args.dataloader_num_workers,
            pin_memory=self._args.data_args.data_loader_args.pin_memory,
        )

        # perform random runs now
        attr_computation_engine = self._configure_random_computation_engine(
            model, model_ckpt_name, summary_writer, output_file=output_file
        )

        # run engine
        print('test_dataloader', len(test_dataloader))
        attr_computation_engine.run(test_dataloader)

        summary_writer.close()

    def run(self):
        logger = logging.getLogger(DEFAULT_LOGGER_NAME)
        try:
            # for each model checkpoint perform the attributions analysis
            for model_ckpt_name, checkpoint in self._args.analyzer_args.model_checkpoints:
                logger.info(f"Running task on model [{model_ckpt_name}]")

                # initialize the model
                model = self._trainer_base.setup_model(
                    self._args,
                    self._tb_logger,
                    summarize=False,
                    stage=TrainingStage.test,
                    gt_metadata=self._datamodule.gt_metadata,
                    checkpoint=checkpoint,
                )

                # fix model inplace relus
                fix_model_inplace(model)

                # get base model
                if self.config.methods is not None:
                    torch_model_unwrapped = model._torch_model

                    # run the analysis through all the attribution methods one by one
                    for method, method_kwargs in self.config.methods.items():
                        # we copy the model here so individual attribution methods do not get into hook related conflicts
                        model._torch_model = wrap_model_output(copy.deepcopy(torch_model_unwrapped))

                        self._perform_attr_analysis(model_ckpt_name, model, method, **method_kwargs)

                    model._torch_model = wrap_model_output(copy.deepcopy(torch_model_unwrapped))
                    self._perform_random_runs_analysis(model_ckpt_name, model)
                else:
                    model._torch_model = wrap_model_output(torch_model_unwrapped)
                    self._perform_random_runs_analysis(model_ckpt_name, model)


        except Exception as e:
            logger.exception(f"Exception raised while generating feature attribution maps: {e}")
