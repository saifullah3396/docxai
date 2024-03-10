from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from xai_torch.core.data.augmentations.base import DataAugmentation
from xai_torch.core.factory.decorators import register_augmentation

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


@register_augmentation()
@dataclass
class RandomDistortion(DataAugmentation):
    """
    Applies random distortion to a numpy image.
    """

    sigma: float
    maxdelta: float

    def __post_init__(self):
        super().__post_init__()

    def _initialize_aug(self):
        import ocrodeg

        def aug(image: ArrayLike):
            noise = ocrodeg.bounded_gaussian_noise(image.shape, self.sigma, self.maxdelta)
            return ocrodeg.distort_with_noise(image, noise)

        return aug


@register_augmentation()
@dataclass
class RandomBlotches(DataAugmentation):
    """
    Applies random blobs to a numpy image.
    """

    fgblobs: float
    bgblobs = float
    fgscale: float = 10
    bgscale: float = 10

    def __post_init__(self):
        super().__post_init__()

    def _initialize_aug(self):
        import ocrodeg

        def aug(image: ArrayLike):
            return ocrodeg.random_blotches(
                image,
                fgblobs=self.fgblobs,
                bgblobs=self.bgblobs,
                fgscale=self.fgscale,
                bgscale=self.bgscale,
            )

        return aug


@register_augmentation()
@dataclass
class SurfaceDistortion(DataAugmentation):
    """
    Applies surface distortion to a numpy image.
    """

    magnitude: float

    def __post_init__(self):
        super().__post_init__()

    def _initialize_aug(self):
        import ocrodeg

        def aug(image: ArrayLike):
            noise = ocrodeg.noise_distort1d(image.shape, magnitude=self.magnitude)
            return ocrodeg.distort_with_noise(image, noise)

        return aug


@register_augmentation()
@dataclass
class Threshold(DataAugmentation):
    """
    Applies threshold distortion on a numpy image.
    """

    magnitude: float

    def __post_init__(self):
        super().__post_init__()

    def _initialize_aug(self):
        import scipy.ndimage as ndi

        def aug(image: ArrayLike):
            blurred = ndi.gaussian_filter(image, self.magnitude)
            return 1.0 * (blurred > 0.5)

        return aug


@register_augmentation()
@dataclass
class Pixelate(DataAugmentation):
    """
    Applies pixelation to a numpy image.
    """

    magnitude: float

    def __post_init__(self):
        super().__post_init__()

    def _initialize_aug(self):
        import cv2

        def aug(image: ArrayLike):
            h, w = image.shape
            image = cv2.resize(
                image,
                (int(w * self.magnitude), int(h * self.magnitude)),
                interpolation=cv2.INTER_LINEAR,
            )
            return cv2.resize(image, (w, h), interpolation=cv2.INTER_NEAREST)

        return aug


@register_augmentation()
@dataclass
class JPEGCompression(DataAugmentation):
    """
    Applies jpeg compression to a numpy image.
    """

    quality: float

    def __post_init__(self):
        super().__post_init__()

    def _initialize_aug(self):
        import cv2

        def aug(image: ArrayLike):
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
            result, encimg = cv2.imencode(".jpg", image * 255, encode_param)
            decimg = cv2.imdecode(encimg, 0) / 255.0
            return decimg

        return aug


@register_augmentation()
@dataclass
class Solarization(DataAugmentation):
    """
    Applies solarization to a numpy image.
    """

    def __post_init__(self):
        super().__post_init__()

    def _initialize_aug(self):
        from PIL import ImageOps

        def aug(image: ArrayLike):
            return ImageOps.solarize(image)

        return aug
