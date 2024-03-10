"""
Defines the feature attribution generation task.
"""

import logging
from dataclasses import dataclass
from typing import List

from xai_torch.core.analyzer.decorators import register_analyzer_task
from xai_torch.core.analyzer.tasks.base import AnalyzerTask
from xai_torch.core.analyzer.tasks.base_config import AnalyzerTaskConfig
from xai_torch.core.training.constants import TrainingStage
from xai_torch.core.training.tb_logger import XAITensorboardLogger
from xai_torch.utilities.logging_utils import DEFAULT_LOGGER_NAME


@register_analyzer_task("generate_metrics")
class GenerateMetrics(AnalyzerTask):
    @dataclass
    class Config(AnalyzerTaskConfig):
        metrics: List[str]

    def setup(self, task_name: str):
        super().setup(task_name=task_name)

        self._metrics = self._setup_metrics()

    def _setup_metrics(self):
        from ignite import metrics

        from xai_torch.core.constants import DataKeys, MetricKeys

        logger = logging.getLogger(DEFAULT_LOGGER_NAME)

        if self._args.model_args.model_task == "image_classification":

            def default_output_transform(output):
                return output[DataKeys.LOGITS], output[DataKeys.LABEL]

            num_classes = self._datamodule.num_labels
            supported_metrics = {
                MetricKeys.ACCURACY: lambda: metrics.Accuracy(output_transform=default_output_transform),
                MetricKeys.CONFUSION_MATRIX: lambda: metrics.ConfusionMatrix(
                    output_transform=default_output_transform, num_classes=num_classes, average="recall"
                ),
            }

            target_metrics = {}
            for metric in self.config.metrics:
                if metric in supported_metrics:
                    target_metrics[metric] = supported_metrics[metric]
                else:
                    logger.warning("Requested metric [{metric}] is not supported.")

            return target_metrics
        else:
            raise ValueError(f"GenerateMetrics is not supported for model task [{self._args.model_args.model_task}].")

    def run(self):
        logger = logging.getLogger(DEFAULT_LOGGER_NAME)
        for model_name, checkpoint in self._args.analyzer_args.model_checkpoints:
            logger.info(f"Testing model [{model_name}]")
            self._tb_logger = XAITensorboardLogger(self._output_dir / model_name)

            model = self._trainer_base.setup_model(
                self._args,
                self._tb_logger,
                summarize=True,
                stage=TrainingStage.test,
                gt_metadata=self._datamodule.gt_metadata,
                checkpoint=checkpoint,
                strict=True,
            )

            model._torch_model._metrics = self._metrics

            test_engine = self._setup_test_engine(model)
            test_engine.run(self._test_dataloader)

            self._tb_logger.close()
