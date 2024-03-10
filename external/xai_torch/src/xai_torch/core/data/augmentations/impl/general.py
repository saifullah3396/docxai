from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from xai_torch.core.data.augmentations.base import DataAugmentation
from xai_torch.core.factory.decorators import register_augmentation

if TYPE_CHECKING:
    import torch
    from numpy.typing import ArrayLike


@register_augmentation()
@dataclass
class Brightness(DataAugmentation):
    """
    Increases/decreases brightness of a numpy image based on the beta parameter.
    """

    beta: float = 0.5

    def __post_init__(self):
        super().__post_init__()

    def _initialize_aug(self):
        import numpy as np

        def aug(image: ArrayLike):
            return np.clip(image + self.beta, 0, 1)

        return aug


@register_augmentation()
@dataclass
class Contrast(DataAugmentation):
    """
    Increases/decreases contrast of a numpy image based on the alpha parameter.
    """

    alpha: float = 0.5

    def __post_init__(self):
        super().__post_init__()

    def _initialize_aug(self):
        import numpy as np

        def aug(image: ArrayLike):
            channel_means = np.mean(image, axis=(0, 1))
            return np.clip((image - channel_means) * self.alpha + channel_means, 0, 1)

        return aug


@register_augmentation()
@dataclass
class GrayScaleToRGB(DataAugmentation):
    """
    Converts a gray-scale torch image to rgb image.
    """

    def __post_init__(self):
        super().__post_init__()

    def _initialize_aug(self):
        def aug(image: torch.Tensor):
            if image.shape[0] == 1:
                return image.repeat(3, 1, 1)
            else:
                return image

        return aug


@register_augmentation()
@dataclass
class RGBToBGR(DataAugmentation):
    """
    Converts a torch tensor from RGB to BGR
    """

    def __post_init__(self):
        super().__post_init__()

    def _initialize_aug(self):
        def aug(image: torch.Tensor):
            return image.permute(2, 1, 0)

        return aug
