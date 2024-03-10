""" Base Model class for the any model from Timm Repository. """


from dataclasses import dataclass, field

from xai_torch.core.constants import DataKeys
from xai_torch.core.models.base_config import ModelConfig
from xai_torch.core.models.decorators import register_model
from xai_torch.core.models.image_base import BaseModuleForImageClassification
from xai_torch.core.models.utilities.general import load_model_from_online_repository
from xai_torch.core.training.constants import TrainingStage


@register_model(reg_name="hf_model", task="image_classification")
class HuggingfaceModelForImageClassification(BaseModuleForImageClassification):
    @dataclass
    class Config(ModelConfig):
        kwargs: dict = field(default_factory={})

    @property
    def model_name(self):
        return self.config.model_type

    def forward(self, image, label=None, stage=TrainingStage.predict):
        if stage in [TrainingStage.train, TrainingStage.test, TrainingStage.val] and label is None:
            raise ValueError(f"Label must be passed for stage={stage}")

        # apply mixup if required
        if stage == TrainingStage.train and self.mixup_fn is not None:
            image, label = self.mixup_fn(image, label)

        # compute logits
        logits = self.model(image).logits
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

    @classmethod
    def load_model(cls, model_name, num_labels, pretrained=True, cache_dir=None, config=None):
        from torch import nn
        from transformers import AutoConfig, AutoModelForImageClassification

        hf_config = AutoConfig.from_pretrained(
            config.model_type,
            cache_dir=cache_dir,
        )
        model = AutoModelForImageClassification.from_pretrained(
            config.model_type,
            cache_dir=cache_dir,
            config=hf_config,
        )
        model.classifier = nn.Linear(model.classifier.in_features, num_labels)
        return model
