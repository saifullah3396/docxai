"""
Defines the base DataAugmentation class for defining any kind of data augmentation.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass

from xai_torch.utilities.abstract_dataclass import AbstractDataclass


@dataclass
class DataAugmentation(AbstractDataclass):
    @abstractmethod
    def _initialize_aug(self):
        pass

    def __post_init__(self):
        self._aug = self._initialize_aug()

    def __call__(self, *args, **kwargs):
        return self._aug(*args, **kwargs)
