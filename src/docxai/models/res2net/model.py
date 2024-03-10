""" Lightning modules for the standard ResNet model. """


from dataclasses import dataclass
from functools import cached_property

from torch import nn
from xai_torch.core.models.base_config import ModelConfig
from xai_torch.core.models.decorators import register_model
from xai_torch.core.models.image_base import BaseModuleForImageClassification

from docxai.models.res2net.definition import MODELS


@register_model(reg_name="res2net", task="image_classification")
class Res2NetForImageClassification(BaseModuleForImageClassification):
    @dataclass
    class Config(ModelConfig):
        pass

    @property
    def model_name(self):
        return f"{self.model_args.name}{self.config.model_type}" if self.config.model_type != "" else self.model_args.name

    @classmethod
    def load_model(cls, model_name, num_labels, use_timm=False, pretrained=True, config=None, cache_dir=''):
        model = MODELS[model_name](pretrained=pretrained)
        if num_labels is not None:
            return Res2NetForImageClassification.update_classifier_for_labels(model, num_labels=num_labels)
        else:
            return model

    @classmethod
    def update_classifier_for_labels(cls, model, num_labels):
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_labels)
        return model
