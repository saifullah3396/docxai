"""
Defines the base DataAugmentation class for defining any kind of data augmentation.
"""

import typing
from dataclasses import dataclass

from xai_torch.core.data.augmentations.base import DataAugmentation
from xai_torch.core.factory.decorators import register_augmentation


@register_augmentation()
@dataclass
class DictTransform(DataAugmentation):
    """
    Applies the transformation on given keys for dictionary outputs

    Args:
        keys (list): List of keys
        transform (callable): Transformation to be applied
    """

    keys: list
    transform: typing.Callable

    def __post_init__(self):
        return super().__post_init__()

    def _initialize_aug(self):
        def aug(sample):
            for key in self.keys:
                if key in sample:
                    sample[key] = self.transform(sample[key])

            return sample

        return aug
