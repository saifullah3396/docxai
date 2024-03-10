""" PyTorch module that defines the base model for training/testing etc. """

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from xai_torch.core.models.utilities.checkpoints import prepend_keys

if TYPE_CHECKING:
    from xai_torch.core.args import Arguments
    from xai_torch.core.data.utilities.typing import CollateFnDict

from xai_torch.core.constants import DataKeys, MetricKeys
from xai_torch.core.models.base import BaseModule
from xai_torch.core.training.constants import TrainingStage


class BaseModuleForImageClassification(BaseModule):
    def __init__(
        self,
        args: Arguments,
        class_labels: List[str],
        **kwargs,
    ):
        super().__init__(args, **kwargs)

        self._logger.info(f"Initialized the model with data labels: {class_labels}")
        self._class_labels = class_labels

    @property
    def num_labels(self):
        return len(self._class_labels)

    def _init_metrics(self):
        from ignite.metrics import Accuracy

        def acc_output_transform(output):
            return output[DataKeys.LOGITS], output[DataKeys.LABEL]

        return {
            MetricKeys.ACCURACY: lambda: Accuracy(
                output_transform=acc_output_transform,
            )
        }

    def _build_model(self):
        import torch
        from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

        self.model = self.load_model(
            model_name=self.model_name,
            num_labels=self.num_labels,
            pretrained=self.args.model_args.pretrained,
            cache_dir=self.model_args.cache_dir,
            config=self.config,
        )
        self.loss_fn_train = torch.nn.CrossEntropyLoss()
        self.loss_fn_eval = torch.nn.CrossEntropyLoss()

        # setup mixup function
        if self.training_args.cutmixup_args is not None:
            self.mixup_fn = self.training_args.cutmixup_args.get_fn(
                num_classes=self.num_labels, smoothing=self.training_args.smoothing
            )

        # setup loss accordingly if mixup, or label smoothing is required
        if self.mixup_fn is not None:
            # smoothing is handled with mixup label transform
            self.loss_fn_train = SoftTargetCrossEntropy()
        elif self.training_args.smoothing > 0.0:
            self.loss_fn_train = LabelSmoothingCrossEntropy(smoothing=self.training_args.smoothing)
        else:
            self.loss_fn_train = torch.nn.CrossEntropyLoss()
        self.loss_fn_eval = torch.nn.CrossEntropyLoss()

    def forward(self, image, label=None, stage=TrainingStage.predict):
        if stage in [TrainingStage.train, TrainingStage.test, TrainingStage.val] and label is None:
            raise ValueError(f"Label must be passed for stage={stage}")

        # apply mixup if required
        if stage == TrainingStage.train and self.mixup_fn is not None:
            image, label = self.mixup_fn(image, label)

        # compute logits
        logits = self.model(image)
        if stage is TrainingStage.predict:
            if self.config.return_dict:
                return {
                    DataKeys.LOGITS: logits,
                }
            else:
                return logits
        else:
            if stage is TrainingStage.train:
                loss = self.loss_fn_train(logits, label)
            else:
                loss = self.loss_fn_eval(logits, label)
            if self.config.return_dict:
                return {
                    DataKeys.LOSS: loss,
                    DataKeys.LOGITS: logits,
                    DataKeys.LABEL: label,
                }
            else:
                return (loss, logits, label)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any], strict: bool = True):
        state_dict_key = "state_dict"
        checkpoint = prepend_keys(checkpoint, state_dict_key, ["model"])
        return super().on_load_checkpoint(checkpoint, strict)

    @classmethod
    def load_model(
        cls,
        model_name,
        num_labels=None,
        use_timm=True,
        pretrained=True,
        config=None,
        cache_dir=None,
        **kwargs,
    ):
        from xai_torch.core.models.utilities.general import load_model_from_online_repository

        return load_model_from_online_repository(
            model_name=model_name,
            num_labels=num_labels,
            use_timm=use_timm,
            pretrained=pretrained,
            **kwargs,
        )

    @classmethod
    def get_data_collators(
        cls, args: Optional[Arguments] = None, data_key_type_map: Optional[dict] = None
    ) -> CollateFnDict:
        import torch
        from xai_torch.core.data.utilities.typing import CollateFnDict
        from xai_torch.core.models.utilities.data_collators import BatchToTensorDataCollator

        if data_key_type_map is None:
            data_key_type_map = {
                DataKeys.IMAGE: torch.float,
                DataKeys.LABEL: torch.long,
            }
        else:
            data_key_type_map[DataKeys.IMAGE] = torch.float
            data_key_type_map[DataKeys.LABEL] = torch.long

        collate_fn = BatchToTensorDataCollator(
            data_key_type_map=data_key_type_map,
        )

        # initialize the data collators for bert grid based word classification
        return CollateFnDict(train=collate_fn, val=collate_fn, test=collate_fn)
