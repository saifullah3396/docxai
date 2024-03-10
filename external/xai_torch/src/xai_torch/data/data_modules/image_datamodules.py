"""
Defines the DataModule for CIFAR10 dataset.
"""

from functools import cached_property

from xai_torch.core.data.data_modules.image_datamodule import ImageDataModule
from xai_torch.core.factory.decorators import register_datamodule


@register_datamodule(reg_name="cifar10")
class CIFAR10(ImageDataModule):
    @cached_property
    def dataset_class(self):
        from xai_torch.data.datasets.cifar10 import CIFAR10Dataset

        return CIFAR10Dataset


@register_datamodule(reg_name="mnist")
class MNIST(ImageDataModule):
    @cached_property
    def dataset_class(self):
        from xai_torch.data.datasets.mnist import MNISTDataset

        return MNISTDataset
