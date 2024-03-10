from __future__ import annotations

import typing
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple

from xai_torch.core.data.augmentations.base import DataAugmentation
from xai_torch.core.factory.decorators import register_augmentation

if TYPE_CHECKING:
    import torch
    from numpy.typing import ArrayLike


@register_augmentation()
@dataclass
class Translation(DataAugmentation):
    """
    Applies translation to a numpy image based on the magnitude in x-y.
    """

    magnitude: Tuple[float] = (0, 0)

    def __post_init__(self):
        return super().__post_init__()

    def _initialize_aug(self):
        import ocrodeg

        def aug(image: ArrayLike):
            return ocrodeg.transform_image(image, translation=self.magnitude)

        return aug


@register_augmentation()
@dataclass
class Scale(DataAugmentation):
    """
    Changes scale of a numpy image based on the scale in x-y.
    """

    scale: Tuple[float] = (0, 0)
    fill: float = 1.0

    def __post_init__(self):
        return super().__post_init__()

    def _initialize_aug(self):
        import numpy as np
        import torch
        from torchvision.transforms import RandomAffine

        def aug(image: ArrayLike):
            image = torch.tensor(image).unsqueeze(0)
            scale = np.random.choice(self.scale)
            scale = [scale - 0.025, scale + 0.025]
            t = RandomAffine(degrees=0, scale=scale, fill=self.fill)
            image = t(image).squeeze().numpy()
            return image

        return aug


@register_augmentation()
@dataclass
class Rotation(DataAugmentation):
    """
    Applies rotation to a numpy image based on the magnitude in +/-.
    """

    magnitude: float

    def __post_init__(self):
        return super().__post_init__()

    def _initialize_aug(self):
        import scipy.ndimage as ndi

        def aug(image: ArrayLike):
            return ndi.rotate(image, self.magnitude)

        return aug


@register_augmentation()
@dataclass
class RandomChoiceAffine(DataAugmentation):
    """
    Randomly applies affine transformation to a numpy image based on the magnitudes of
    rotation degrees, translation, and shear around the top or the bottom value of
    the input range randomly.
    """

    degrees: Tuple[float, float] = (0, 0)
    translate: Tuple[float, float] = (0, 0)
    shear: Tuple[float, float] = (0, 0)
    fill: float = 1.0

    def __post_init__(self):
        return super().__post_init__()

    def _initialize_aug(self):
        import numpy as np
        import torch
        from torchvision.transforms import RandomAffine

        def aug(image: ArrayLike):
            image = torch.tensor(image).unsqueeze(0)
            translate = np.random.choice(self.translate)
            translate = [translate - 0.01, translate + 0.01]
            degrees = np.random.choice(self.degrees)
            degrees = [degrees - 1, degrees + 1]
            shear = np.random.choice(self.shear)
            shear = [shear - 0.05, shear + 0.05]
            t = RandomAffine(degrees=degrees, translate=translate, shear=shear, fill=self.fill)
            image = t(image).squeeze().numpy()
            return image

        return aug


@register_augmentation()
@dataclass
class Elastic(DataAugmentation):
    """
    Applies elastic transformation to a numpy image.
    """

    alpha: float = 70
    sigma: float = 500
    alpha_affine: float = 10
    random_state: Optional[float] = None

    def __post_init__(self):
        return super().__post_init__()

    def _initialize_aug(self):
        import cv2
        import numpy as np
        from scipy.ndimage.interpolation import map_coordinates
        from skimage.filters import gaussian

        def aug(image: ArrayLike):
            assert len(image.shape) == 2
            shape = image.shape
            shape_size = shape[:2]

            image = np.array(image, dtype=np.float32) / 255.0
            shape = image.shape
            shape_size = shape[:2]

            # random affine
            center_square = np.float32(shape_size) // 2
            square_size = min(shape_size) // 3
            pts1 = np.float32(
                [
                    center_square + square_size,
                    [center_square[0] + square_size, center_square[1] - square_size],
                    center_square - square_size,
                ]
            )
            pts2 = pts1 + np.random.uniform(-self.alpha_affine, self.alpha_affine, size=pts1.shape).astype(np.float32)
            M = cv2.getAffineTransform(pts1, pts2)
            image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

            dx = (
                gaussian(
                    np.random.uniform(-1, 1, size=shape[:2]),
                    self.sigma,
                    mode="reflect",
                    truncate=3,
                )
                * self.alpha
            ).astype(np.float32)
            dy = (
                gaussian(
                    np.random.uniform(-1, 1, size=shape[:2]),
                    self.sigma,
                    mode="reflect",
                    truncate=3,
                )
                * self.alpha
            ).astype(np.float32)

            x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
            indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
            return (
                np.clip(
                    map_coordinates(image, indices, order=1, mode="reflect").reshape(shape),
                    0,
                    1,
                )
                * 255
            )

        return aug


@register_augmentation()
@dataclass
class Rescale(DataAugmentation):
    """
    Rescales a torch tensor image based on the dimensions provided.

    Args:
        size (Tuple[int, int]): Rescale dimensions for the x-y.
        interpolation (str): Interpolation mode.
        max_size (Optional[int]): Max size for any side.
        antialias (Optional[bool]): Whether to use antialiasing while resizing image.
    """

    size: List[int] = field(default_factory=lambda: [224, 224])
    interpolation: str = field(
        default="bilinear",
        metadata={
            "help": "The data augmentation strategy to use.",
        },
    )
    max_size: Optional[int] = None
    antialias: bool = False

    def __post_init__(self):
        return super().__post_init__()

    def _initialize_aug(self):
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.functional import resize

        def aug(image: torch.Tensor):
            return resize(
                image,
                size=self.size,
                max_size=self.max_size,
                antialias=self.antialias,
                interpolation=InterpolationMode(self.interpolation),
            )

        return aug


@register_augmentation()
@dataclass
class RescaleOneDim(DataAugmentation):
    """
    Rescales a torch tensor image based on the dimensions provided.

    Args:
        rescale_dim (int): Rescale dimension.
        rescale_smaller_dim (bool): Whether to rescale smaller dimension or
            the larger dimension.
        max_size (Optional[int]): Max size for any side.
        antialias (Optional[bool]): Whether to use antialiasing while resizing image.
    """

    rescale_dim: int = 224
    rescale_smaller_dim: bool = False
    interpolation: str = (
        field(
            default="bilinear",
            metadata={
                "help": "The data augmentation strategy to use.",
            },
        ),
    )
    antialias: bool = False

    def __post_init__(self):
        return super().__post_init__()

    def _initialize_aug(self):
        from torchvision.transforms.functional import resize

        def aug(image: torch.Tensor):
            # shape (C, H, W)
            image_height = image.shape[1]
            image_width = image.shape[2]

            # get smaller dim
            larger_dim_idx = 0 if image_height > image_width else 1
            smaller_dim_idx = 0 if image_height < image_width else 1

            dim_idx = smaller_dim_idx if self.rescale_smaller_dim else larger_dim_idx
            other_dim_idx = larger_dim_idx if self.rescale_smaller_dim else smaller_dim_idx

            # find the rescale ratio
            rescale_ratio = self.rescale_dim / image.shape[dim_idx]

            # rescale the other dim
            other_dim = rescale_ratio * image.shape[other_dim_idx]

            rescaled_shape = list(image.shape)
            rescaled_shape[dim_idx] = int(self.rescale_dim)
            rescaled_shape[other_dim_idx] = int(other_dim)

            # resize the image according to the output shape
            return resize(
                image,
                size=rescaled_shape[1:],
                interpolation=self.interpolation,
                antialias=self.antialias,
            )

        return aug


@register_augmentation()
@dataclass
class RandomRescale(DataAugmentation):
    """
    Randomly rescales a torch tensor image based on the input list of possible
    dimensions.

    Args:
        rescale_dims (list): List of possible sizes to choose from for
            shorter dimension.
        max_rescale_dim (int): Maximum rescale size for the larger dimension.
        max_iters (int): Maximum number of iterations to do for random sampling.
    """

    rescale_dims: typing.List[int] = field(default_factory=lambda: [320, 416, 512, 608, 704])
    max_larger_dim: int = 512
    max_iters: int = 100

    def __post_init__(self):
        return super().__post_init__()

    def _initialize_aug(self):
        import random

        from torchvision.transforms.functional import resize

        def aug(image: torch.Tensor):
            # randomly rescale the image in the batch as done in ViBertGrid
            # shape (C, H, W)
            image_height = image.shape[1]
            image_width = image.shape[2]

            # get larger dim
            larger_dim_idx = 0 if image_height > image_width else 1
            smaller_dim_idx = 0 if image_height < image_width else 1

            rescale_dims = [i for i in self.rescale_dims]

            # find random rescale dim
            rescaled_shape = None
            for iter in range(self.max_iters):
                if len(rescale_dims) > 0:
                    # get smaller dim out of possible dims
                    idx, smaller_dim = random.choice(list(enumerate(rescale_dims)))

                    # find the rescale ratio
                    rescale_ratio = smaller_dim / image.shape[smaller_dim_idx]

                    # rescale larger dim
                    larger_dim = rescale_ratio * image.shape[larger_dim_idx]

                    # check if larger dim is smaller than max large
                    if larger_dim > self.max_larger_dim:
                        rescale_dims.pop(idx)
                    else:
                        rescaled_shape = list(image.shape)
                        rescaled_shape[larger_dim_idx] = int(larger_dim)
                        rescaled_shape[smaller_dim_idx] = int(smaller_dim)
                        break
                else:
                    # if no smaller dim is possible rescale image according to
                    # larger dim
                    larger_dim = self.max_larger_dim

                    # find the rescale ratio
                    rescale_ratio = larger_dim / image.shape[larger_dim_idx]

                    # rescale smaller dim
                    smaller_dim = rescale_ratio * image.shape[smaller_dim_idx]

                    rescaled_shape = list(image.shape)
                    rescaled_shape[larger_dim_idx] = int(larger_dim)
                    rescaled_shape[smaller_dim_idx] = int(smaller_dim)
                    break

            if rescaled_shape is not None:
                # resize the image according to the output shape
                return resize(image, rescaled_shape[1:])
            else:
                return image

        return aug


@register_augmentation()
@dataclass
class RandomResizedCrop(DataAugmentation):
    """
    Applies random resized cropping on a torch tensor image.
    """

    size: List[int] = field(default_factory=lambda: [224, 224])
    scale: Tuple[int, int] = (0.08, 1)
    ratio: Tuple[float, float] = (3 / 4, 4 / 3)
    interpolation: str = (
        field(
            default="bilinear",
            metadata={
                "help": "The data augmentation strategy to use.",
            },
        ),
    )

    def __post_init__(self):
        return super().__post_init__()

    def _initialize_aug(self):
        from torchvision import transforms

        return transforms.RandomResizedCrop(
            size=self.size, scale=self.scale, ratio=self.ratio, interpolation=self.interpolation
        )


@register_augmentation()
@dataclass
class RandomCrop(DataAugmentation):
    """
    Applies random cropping on a torch tensor image.
    """

    size: List[int] = field(default_factory=lambda: [224, 224])

    def __post_init__(self):
        return super().__post_init__()

    def _initialize_aug(self):
        from torchvision import transforms

        return transforms.RandomCrop(size=self.size)


# class RandomResizedMaskedCrop(object):
#     image_size: int = 224
#     def __init__(self, image_size, scale):
#         self.t = RandomResizedCropCustom((image_size, image_size), scale=scale)

#     def get_black_and_white_regions_mask(self, image_tensor):
#         black_and_white_threshold = 0.5
#         c, h, w = image_tensor.shape
#         ky = 8
#         kx = 8
#         black_and_white_regions_fast = (
#             (
#                 image_tensor[0].unfold(0, ky, kx).unfold(1, ky, kx)
#                 < black_and_white_threshold
#             )
#             .any(dim=2)
#             .any(dim=2)
#         )
#         black_and_white_regions_fast = black_and_white_regions_fast.repeat_interleave(
#             ky, dim=0
#         ).repeat_interleave(kx, dim=1)
#         black_and_white_regions_fast = transforms.functional.resize(
#             black_and_white_regions_fast.unsqueeze(0), [h, w]
#         ).squeeze()
#         return (black_and_white_regions_fast).float()

#     def __call__(self, img):
#         return self.t(img, self.get_black_and_white_regions_mask(img)) / 255.0