""" Base Model class for the any model from Timm Repository. """


from dataclasses import dataclass, field

import torch.nn.functional as F
from torch import nn
from xai_torch.core.models.base_config import ModelConfig
from xai_torch.core.models.decorators import register_model
from xai_torch.core.models.image_base import BaseModuleForImageClassification


class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


@register_model(reg_name="test_model", task="image_classification")
class TestModelForImageClassification(BaseModuleForImageClassification):
    @dataclass
    class Config(ModelConfig):
        kwargs: dict = field(default_factory={})

    @classmethod
    def load_model(
        cls,
        model_name,
        num_labels=None,
        use_timm=True,
        pretrained=True,
        config=None,
        **kwargs,
    ):
        return TestModel()
