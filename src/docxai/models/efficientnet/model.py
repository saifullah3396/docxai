""" Lightning modules for the standard VGG model. """


from dataclasses import dataclass
from functools import cached_property

from efficientnet_pytorch import EfficientNet
from xai_torch.core.models.base_config import ModelConfig
from xai_torch.core.models.decorators import register_model
from xai_torch.core.models.image_base import BaseModuleForImageClassification


@register_model(reg_name="efficientnet", task="image_classification")
class EfficientNetForImageClassification(BaseModuleForImageClassification):
    @dataclass
    class Config(ModelConfig):
        dropout_rate: float = 0.5

    @property
    def model_name(self):
        return f"{self.model_args.name}-{self.config.model_type}" if self.config.model_type != "" else self.model_args.name

    @classmethod
    def load_model(cls, model_name, num_labels, use_timm=False, pretrained=True, config=None, cache_dir=''):
        if pretrained:
            return EfficientNet.from_pretrained(
                model_name,
                num_classes=num_labels,
                dropout_rate=config.dropout_rate,
            )
        else:
            return EfficientNet.from_name(
                model_name,
                num_classes=num_labels,
                dropout_rate=config.dropout_rate,
            )
