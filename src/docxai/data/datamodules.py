"""
Defines the DataModule for CIFAR10 dataset.
"""

from functools import cached_property

from xai_torch.core.data.data_modules.image_datamodule import ImageDataModule
from xai_torch.core.factory.decorators import register_datamodule

from docxai.data.datasets.rvlcdip import RVLCDIPDataset
from docxai.data.datasets.tobacco3482 import Tobacco3482Dataset


@register_datamodule(reg_name="rvlcdip")
class RVLCDIP(ImageDataModule):
    @cached_property
    def dataset_class(self):
        return RVLCDIPDataset


@register_datamodule(reg_name="tobacco3482")
class Tobacco3482(ImageDataModule):
    @cached_property
    def dataset_class(self):
        return Tobacco3482Dataset
