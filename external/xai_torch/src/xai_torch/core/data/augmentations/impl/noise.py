from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from xai_torch.core.data.augmentations.base import DataAugmentation
from xai_torch.core.factory.decorators import register_augmentation

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


@register_augmentation()
@dataclass
class GaussianNoiseRGB(DataAugmentation):
    """
    Applies RGB Gaussian noise to a numpy image.
    """

    magnitude: float

    def __post_init__(self):
        return super().__post_init__()

    def _initialize_aug(self):
        import cv2
        import numpy as np

        def aug(image: ArrayLike):
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            return np.clip(image + np.random.normal(size=image.shape, scale=self.magnitude), 0, 1)

        return aug


@register_augmentation()
@dataclass
class ShotNoiseRGB(DataAugmentation):
    """
    Applies shot noise to a numpy image.
    """

    magnitude: float

    def __post_init__(self):
        return super().__post_init__()

    def _initialize_aug(self):
        import cv2
        import numpy as np

        def aug(image: ArrayLike):
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            return np.clip(np.random.poisson(image * self.magnitude) / float(self.magnitude), 0, 1)

        return aug


@register_augmentation()
@dataclass
class FibrousNoise(DataAugmentation):
    """
    Applies fibrous noise to a numpy image.
    """

    blur: float = 1.0
    blotches: float = 5e-5

    def __post_init__(self):
        return super().__post_init__()

    def _initialize_aug(self):
        import ocrodeg

        def aug(image: ArrayLike):
            return ocrodeg.printlike_fibrous(image, blur=self.blur, blotches=self.blotches)

        return aug


@register_augmentation()
@dataclass
class MultiscaleNoise(DataAugmentation):
    """
    Applies multiscale noise to a numpy image.
    """

    blur: float = 1.0
    blotches: float = 5e-5

    def __post_init__(self):
        return super().__post_init__()

    def _initialize_aug(self):
        import ocrodeg

        def aug(image: ArrayLike):
            return ocrodeg.printlike_multiscale(image, blur=self.blur, blotches=self.blotches)

        return aug
