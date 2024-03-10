from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

from xai_torch.core.data.augmentations.base import DataAugmentation
from xai_torch.core.factory.decorators import register_augmentation

if TYPE_CHECKING:
    from numpy.typing import ArrayLike
    from PIL import Image as PILImage


@register_augmentation()
@dataclass
class GaussianBlur(DataAugmentation):
    """
    Applies gaussian blur to a numpy image.
    """

    magnitude: float

    def __post_init__(self):
        super().__post_init__()

    def _initialize_aug(self):
        import scipy.ndimage as ndi

        return lambda image: ndi.gaussian_filter(image, self.magnitude)


@register_augmentation()
@dataclass
class GaussianBlurPIL(DataAugmentation):
    """
    Applies gaussian blur to a PIL image.
    """

    sigma: Tuple[float, float] = (0.1, 2.0)

    def __post_init__(self):
        super().__post_init__()

    def _initialize_aug(self):
        import random

        from PIL import ImageFilter

        def aug(image: PILImage):
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            return image.filter(ImageFilter.GaussianBlur(radius=sigma))

        return aug


@register_augmentation()
@dataclass
class BinaryBlur(DataAugmentation):
    """
    Applies binary blur to a numpy image.
    """

    sigma: float

    def __post_init__(self):
        super().__post_init__()

    def _initialize_aug(self):
        import ocrodeg

        return lambda image: ocrodeg.binary_blur(image, sigma=self.sigma)


@register_augmentation()
@dataclass
class NoisyBinaryBlur(DataAugmentation):
    """
    Applies noisey binary blur to a numpy image.
    """

    sigma: float
    noise: float

    def __post_init__(self):
        super().__post_init__()

    def _initialize_aug(self):
        import ocrodeg

        return lambda image: ocrodeg.binary_blur(image, sigma=self.sigma, noise=self.noise)


@register_augmentation()
@dataclass
class DefocusBlur(DataAugmentation):
    """
    Applies defocus blur to a numpy image.
    """

    radius: float
    alias_blur: float = 0.1

    def __post_init__(self):
        super().__post_init__()

    def _initialize_aug(self):
        import cv2
        import numpy as np

        from xai_torch.data.augmentations.utilities import disk

        def aug(image: ArrayLike):
            kernel = disk(radius=self.radius, alias_blur=self.alias_blur)
            return np.clip(cv2.filter2D(image, -1, kernel), 0, 1)

        return aug


@register_augmentation()
@dataclass
class MotionBlur(DataAugmentation):
    """
    Applies motion blur to a numpy image.
    """

    size: float

    def __post_init__(self):
        super().__post_init__()

    def _initialize_aug(self):
        import cv2
        import numpy as np

        def aug(image: ArrayLike):
            kernel_motion_blur = np.zeros((self.size, self.size))
            kernel_motion_blur[int((self.size - 1) / 2), :] = np.ones(self.size, dtype=np.float32)
            kernel_motion_blur = cv2.warpAffine(
                kernel_motion_blur,
                cv2.getRotationMatrix2D(
                    (self.size / 2 - 0.5, self.size / 2 - 0.5),
                    np.random.uniform(-45, 45),
                    1.0,
                ),
                (self.size, self.size),
            )
            kernel_motion_blur = kernel_motion_blur * (1.0 / np.sum(kernel_motion_blur))
            return cv2.filter2D(image, -1, kernel_motion_blur)

        return aug


@register_augmentation()
@dataclass
class ZoomBlur(DataAugmentation):
    """
    Applies zoom blur to a numpy image.
    """

    zoom_factor_start: float
    zoom_factor_end: float
    zoom_factor_step: float

    def __post_init__(self):
        super().__post_init__()

    def _initialize_aug(self):
        import numpy as np

        from xai_torch.data.augmentations.utilities import clipped_zoom

        def aug(image: ArrayLike):
            out = np.zeros_like(image)
            zoom_factor_range = np.arange(self.zoom_factor_start, self.zoom_factor_end, self.zoom_factor_step)
            for zoom_factor in zoom_factor_range:
                out += clipped_zoom(image, zoom_factor)
            return np.clip((image + out) / (len(zoom_factor_range) + 1), 0, 1)

        return aug
