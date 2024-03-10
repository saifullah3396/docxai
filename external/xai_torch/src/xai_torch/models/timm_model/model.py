""" Base Model class for the any model from Timm Repository. """


from dataclasses import dataclass, field

from xai_torch.core.models.base_config import ModelConfig
from xai_torch.core.models.decorators import register_model
from xai_torch.core.models.image_base import BaseModuleForImageClassification
from xai_torch.core.models.utilities.general import load_model_from_online_repository


@register_model(reg_name="timm_model", task="image_classification")
class TimmModelForImageClassification(BaseModuleForImageClassification):
    @dataclass
    class Config(ModelConfig):
        kwargs: dict = field(default_factory=lambda: {})

    @property
    def model_name(self):
        return self.config.model_type

    @classmethod
    def load_model(cls, model_name, num_labels, pretrained=True, config=None, cache_dir=None):
        return load_model_from_online_repository(
            model_name=model_name,
            num_labels=num_labels,
            use_timm=True,
            pretrained=pretrained,
            **config.kwargs,
        )
