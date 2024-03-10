""" Base Model class for the any model from Timm Repository. """


from dataclasses import dataclass, field

from xai_torch.core.models.base_config import ModelConfig
from xai_torch.core.models.decorators import register_model
from xai_torch.core.models.image_base import BaseModuleForImageClassification
from xai_torch.core.models.utilities.general import load_model_from_online_repository


@register_model(reg_name="tv_model", task="image_classification")
class TorchvisionModelForImageClassification(BaseModuleForImageClassification):
    @dataclass
    class Config(ModelConfig):
        model_type: str = "alexnet"
        kwargs: dict = field(default_factory={})

    @property
    def model_name(self):
        return self.config.model_type

    @classmethod
    def replace_classification_head(cls, model_name, model, num_labels):
        import torch

        if model_name == "alexnet":
            model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_labels)
        elif model_name == "vgg16":
            model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_labels)
        elif model_name == "resnet50":
            model.fc = torch.nn.Linear(model.fc.in_features, num_labels)
        elif model_name == "inception_v3":
            model.fc = torch.nn.Linear(model.fc.in_features, num_labels)
        elif model_name == "googlenet":
            model.fc = torch.nn.Linear(model.fc.in_features, num_labels)
        else:
            raise ValueError(f"No classification head replacer defined for the model {model_name}.")

        return model

    @classmethod
    def load_model(cls, model_name, num_labels, pretrained=True, config=None, cache_dir=None):
        import torch

        model = load_model_from_online_repository(
            model_name=model_name,
            num_labels=num_labels,
            use_timm=False,
            pretrained=pretrained,
            **config.kwargs,
        )
        model = cls.replace_classification_head(model_name, model, num_labels)
        return model
