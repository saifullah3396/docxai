from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

from xai_torch.core.data.augmentations.base import DataAugmentation
from xai_torch.core.factory.decorators import register_augmentation

if TYPE_CHECKING:
    import torch


class ImageRescaleStrategyEnum(Enum):
    RESCALE = "rescale"
    RESCALE_ONE_DIM = "rescale_one_dim"
    RESCALE_RANDOM = "rescale_random"
    RANDOM_RESIZED_CROP = "random_resized_crop"
    RANDOM_CROP = "random_crop"


@register_augmentation()
@dataclass
class BasicImageAug(DataAugmentation):
    """
    Defines a basic image augmentation for image classification.
    """

    gray_to_rgb: bool = False
    rgb_to_bgr: bool = False
    rescale_strategy: Optional[str] = field(
        default=None,
        metadata={
            "help": "The rescaling strategy to use.",
            "choices": [im.value for im in ImageRescaleStrategyEnum],
        },
    )
    center_crop: Optional[List[int]] = None
    rescale_config: Optional[dict] = None
    normalize: bool = True
    random_hflip: bool = False
    random_vflip: bool = field(default=False, metadata={"help": "Whether to perform random vertical flip."})
    mean: Union[float, List[float]] = (0.485, 0.456, 0.406)
    std: Union[float, List[float]] = (0.229, 0.224, 0.225)

    def __str__(self):
        return str(self._aug)

    def _initialize_aug(self):
        import torch
        from torchvision import transforms

        from xai_torch.core.data.augmentations import GrayScaleToRGB, RGBToBGR
        from xai_torch.core.factory.factory import DataAugmentationFactory

        # create rescaling transform
        self.rescale_transform = None
        if self.rescale_strategy is not None:
            assert self.rescale_config is not None

            self.rescale_strategy = ImageRescaleStrategyEnum(self.rescale_strategy).value
            self.rescale_transform = DataAugmentationFactory.create(
                strategy=self.rescale_strategy, config=self.rescale_config
            )

        # generate transformations list
        aug = []

        # apply gray to rgb if required
        if self.gray_to_rgb:
            aug.append(GrayScaleToRGB())

        # apply rgb to bgr if required
        if self.rgb_to_bgr:
            aug.append(RGBToBGR())

        # apply rescaling if required
        if self.rescale_transform is not None:
            aug.append(self.rescale_transform)

        # apply center crop if required
        if self.center_crop is not None:
            aug.append(transforms.CenterCrop(self.center_crop))

        # apply random horizontal flip if required
        if self.random_hflip:
            aug.append(transforms.RandomHorizontalFlip(0.5))

        # apply random vertical flip if required
        if self.random_vflip:
            aug.append(transforms.RandomVerticalFlip(0.5))

        # change dtype to float
        aug.append(transforms.ConvertImageDtype(torch.float))

        # normalize image if required
        if self.normalize:
            if isinstance(self.mean, float):
                aug.append(transforms.Normalize((self.mean,), (self.std,)))
            else:
                aug.append(transforms.Normalize(self.mean, self.std))

        # generate torch transformation
        return transforms.Compose(aug)

    def __post_init__(self):
        super().__post_init__()


@register_augmentation()
@dataclass
class RandAug(DataAugmentation):
    """
    Applies the ImageNet Random Augmentation to torch tensor or numpy image as
    defined in timm for image classification with little modification.
    """

    input_size: int = 224
    is_training: bool = True
    use_prefetcher: bool = False
    no_aug: bool = False
    scale: Optional[float] = None
    ratio: Optional[float] = None
    hflip: float = 0.5
    vflip: float = 0.0
    color_jitter: Union[float, List[float]] = 0.4
    auto_augment: Optional[str] = "rand-m9-mstd0.5-inc1"
    interpolation: str = "bicubic"
    mean: Union[float, List[float]] = (0.485, 0.456, 0.406)
    std: Union[float, List[float]] = (0.229, 0.224, 0.225)
    re_prob: float = 0.0
    re_mode: str = "const"
    re_count: int = 1
    re_num_splits: int = 0
    crop_pct: Optional[float] = None
    tf_preprocessing: bool = False
    separate: bool = False

    # custom args
    n_augs: int = 1
    fixed_resize: bool = False

    def __str__(self):
        if self.n_augs == 1:
            return str(self._aug)
        else:
            return str([self._aug for _ in self.n_augs])

    def __post_init__(self):
        super().__post_init__()

    def _initialize_aug(self):
        from timm.data import create_transform
        from torchvision import transforms

        from xai_torch.core.data.augmentations.impl.general import GrayScaleToRGB

        torch_str_to_interpolation = {
            "nearest": transforms.InterpolationMode.NEAREST,
            "bilinear": transforms.InterpolationMode.BILINEAR,
            "bicubic": transforms.InterpolationMode.BICUBIC,
            "box": transforms.InterpolationMode.BOX,
            "hamming": transforms.InterpolationMode.HAMMING,
            "lanczos": transforms.InterpolationMode.LANCZOS,
        }

        aug = create_transform(
            input_size=self.input_size,
            is_training=self.is_training,
            use_prefetcher=self.use_prefetcher,
            no_aug=self.no_aug,
            scale=self.scale,
            ratio=self.ratio,
            hflip=self.hflip,
            vflip=self.vflip,
            color_jitter=self.color_jitter,
            auto_augment=self.auto_augment,
            interpolation=self.interpolation,
            mean=self.mean,
            std=self.std,
            re_prob=self.re_prob,
            re_mode=self.re_mode,
            re_count=self.re_count,
            re_num_splits=self.re_num_splits,
            crop_pct=self.crop_pct,
            tf_preprocessing=self.tf_preprocessing,
            separate=self.separate,
        ).transforms

        # replace random resized crop with fixed resizing if required
        if self.fixed_resize:
            aug[0] = transforms.Resize(self.input_size, interpolation=torch_str_to_interpolation[self.interpolation])

            # this makes sure image is always 3-channeled.
            aug.insert(0, GrayScaleToRGB())

            # this makes sure image is always 3-channeled.
            aug.insert(2, transforms.ToPILImage())
        else:
            # this makes sure image is always 3-channeled.
            aug.insert(0, GrayScaleToRGB())

            # this makes sure image is always 3-channeled.
            aug.insert(1, transforms.ToPILImage())

        # generate torch transformation
        return transforms.Compose(aug)

    def __call__(self, image: torch.Tensor):
        if self.n_augs == 1:
            return self._aug(image)
        else:
            augs = []
            for _ in self.n_augs:
                augs.append(self._aug(image))
                augs.append(self._aug(image))
        return augs


@register_augmentation()
@dataclass
class Moco(DataAugmentation):
    """
    Applies the Standard Moco Augmentation to a torch tensor image.
    """

    image_size: Union[int, Tuple[int, int]] = 224
    gray_to_rgb: bool = False
    to_pil: bool = False
    mean: Union[float, List[float]] = (0.485, 0.456, 0.406)
    std: Union[float, List[float]] = (0.229, 0.224, 0.225)

    def __str__(self):
        return str([self.aug, self.aug])

    def __post_init__(self):
        super().__post_init__()

    def _initialize_aug(self):
        from PIL import Image as PILImage
        from torchvision import transforms

        from xai_torch.core.data.augmentations import GaussianBlurPIL, GrayScaleToRGB

        base_aug = []
        if self.gray_to_rgb:
            base_aug.append(GrayScaleToRGB())

        if self.to_pil:
            base_aug.append(transforms.ToPILImage())

        return transforms.Compose(
            base_aug
            + [
                transforms.RandomResizedCrop(self.image_size, scale=(0.2, 1.0), interpolation=PILImage.BICUBIC),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlurPIL(sigma=(0.1, 2.0))], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )

    def __call__(self, image):
        crops = []
        crops.append(self._aug(image))
        crops.append(self._aug(image))
        return crops


@register_augmentation()
@dataclass
class BarlowTwins(DataAugmentation):
    """
    Applies the Standard BarlowTwins Augmentation to a torch tensor image.
    """

    image_size: Union[int, Tuple[int, int]] = 224
    gray_to_rgb: bool = False
    to_pil: bool = False
    mean: Union[float, List[float]] = (0.485, 0.456, 0.406)
    std: Union[float, List[float]] = (0.229, 0.224, 0.225)

    def __str__(self):
        return str([self.aug, self.aug])

    def __post_init__(self):
        super().__post_init__()

    def _initialize_aug(self):
        from PIL import Image as PILImage
        from torchvision import transforms

        from xai_torch.core.data.augmentations.blur import GaussianBlurPIL
        from xai_torch.core.data.augmentations.distortions import Solarization
        from xai_torch.core.data.augmentations.general import GrayScaleToRGB

        base_aug = []
        if self.gray_to_rgb:
            base_aug.append(GrayScaleToRGB())
        if self.to_pil:
            base_aug.append(transforms.ToPILImage())

        aug1 = transforms.Compose(
            base_aug
            + [
                transforms.RandomResizedCrop(self.image_size, interpolation=PILImage.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlurPIL([0.1, 2.0])], p=1.0),
                transforms.RandomApply([Solarization()], p=0.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )
        aug2 = transforms.Compose(
            base_aug
            + [
                transforms.RandomResizedCrop(self.image_size, interpolation=PILImage.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlurPIL([0.1, 2.0])], p=0.1),
                transforms.RandomApply([Solarization()], p=0.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )

        return aug1, aug2

    def __call__(self, image):
        crops = []
        crops.append(self._aug[0](image))
        crops.append(self._aug[1](image))
        return crops


@register_augmentation()
@dataclass
class MultiCrop(DataAugmentation):
    """
    Applies the Standard Multicrop Augmentation to a torch tensor image.
    """

    image_size: Union[int, Tuple[int, int]] = 224
    gray_to_rgb: bool = False
    to_pil: bool = False
    mean: Union[float, List[float]] = (0.485, 0.456, 0.406)
    std: Union[float, List[float]] = (0.229, 0.224, 0.225)
    global_crops_scale: Tuple[float, float] = (0.4, 1.0)
    local_crops_scale: Tuple[float, float] = (0.05, 0.4)
    local_crops_number: int = 12
    local_crop_size: Union[int, Tuple[int, int]] = 96

    def __str__(self):
        return json.dumps(
            {
                "global_1": self.global_transform_1,
                "global_2": self.global_transform_2,
                "local": self.local_transform,
            },
            indent=2,
        )

    def __post_init__(self):
        super().__post_init__()

    def _initialize_aug(self):
        from PIL import Image as PILImage
        from torchvision import transforms

        from xai_torch.core.data.augmentations.blur import GaussianBlurPIL
        from xai_torch.core.data.augmentations.distortions import Solarization
        from xai_torch.core.data.augmentations.general import GrayScaleToRGB

        base_aug = []
        if self.gray_to_rgb:
            base_aug.append(GrayScaleToRGB())
        if self.to_pil:
            base_aug.append(transforms.ToPILImage())

        flip_and_color_jitter = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )
        normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )

        # first global crop
        self.global_transform_1 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    self.image_size,
                    scale=self.global_crops_scale,
                    interpolation=PILImage.BICUBIC,
                ),
                flip_and_color_jitter,
                transforms.RandomApply([GaussianBlurPIL([0.1, 2.0])], p=1.0),
                normalize,
            ]
        )
        # second global crop
        self.global_transform_2 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    self.image_size,
                    scale=self.global_crops_scale,
                    interpolation=PILImage.BICUBIC,
                ),
                flip_and_color_jitter,
                transforms.RandomApply([GaussianBlurPIL([0.1, 2.0])], p=0.1),
                transforms.RandomApply([Solarization()], p=0.2),
                normalize,
            ]
        )
        # transformation for the local small crops
        self.local_crops_number = self.local_crops_number
        self.local_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    self.local_crop_size,
                    scale=self.local_crops_scale,
                    interpolation=PILImage.BICUBIC,
                ),
                flip_and_color_jitter,
                transforms.RandomApply([GaussianBlurPIL([0.1, 2.0])], p=0.5),
                normalize,
            ]
        )

        return None

    def __call__(self, image: torch.Tensor):
        crops = []
        crops.append(self.global_transform_1(image))
        crops.append(self.global_transform_2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transform(image))
        return crops


# @register_augmentation()
# @dataclass
# class TwinDocs(DataAugmentation):
#     """
#     Applies the TwinDocs Augmentation to a torch tensor image
#     (experimental augmentation for document images).
#     """

#     image_size: Union[int, Tuple[int, int]] = 224
#     gray_to_rgb: bool = False
#     to_pil: bool = False
# mean: Union[float, List[float]] = (0.485, 0.456, 0.406)
# std: Union[float, List[float]] = (0.229, 0.224, 0.225)

#     def __post_init__(self):
#         base_aug = []
#         if self.gray_to_rgb:
#             base_aug.append(GrayScaleToRGB())
#         if self.to_pil:
#             base_aug.append(transforms.ToPILImage())
#         self.aug1 = transforms.Compose(
#             [
#                 RandomResizedMaskedCrop(self.image_size, (0.4, 0.8)),
#                 *base_aug,
#                 transforms.RandomApply(
#                     [transforms.RandomAffine((-2, 2), fill=255)], p=0.5
#                 ),
#                 transforms.RandomApply(
#                     [transforms.RandomAffine(0, translate=(0.2, 0.2), fill=255)], p=0.5
#                 ),
#                 transforms.RandomApply(
#                     [transforms.RandomAffine(0, scale=(0.9, 1.0), fill=255)], p=0.5
#                 ),
#                 transforms.RandomApply(
#                     [transforms.RandomAffine(0, shear=(-2, 2), fill=255)], p=0.5
#                 ),
#                 # transforms.RandomHorizontalFlip(p=0.5),
#                 transforms.RandomApply(
#                     [
#                         transforms.ColorJitter(
#                             brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
#                         )
#                     ],
#                     p=0.8,
#                 ),
#                 transforms.RandomGrayscale(p=0.2),
#                 transforms.RandomApply([GaussianBlurPIL([0.1, 0.5])], p=1.0),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=self.mean, std=self.std),
#             ]
#         )
#         self.aug2 = transforms.Compose(
#             [
#                 RandomResizedMaskedCrop(self.image_size, (0.4, 0.8)),
#                 *base_aug,
#                 transforms.RandomApply(
#                     [transforms.RandomAffine((-5, 5), fill=255)], p=0.5
#                 ),
#                 transforms.RandomApply(
#                     [transforms.RandomAffine(0, translate=(0.2, 0.2), fill=255)], p=0.5
#                 ),
#                 transforms.RandomApply(
#                     [transforms.RandomAffine(0, scale=(0.9, 1.0), fill=255)], p=0.5
#                 ),
#                 transforms.RandomApply(
#                     [transforms.RandomAffine(0, shear=(-5, 5), fill=255)], p=0.5
#                 ),
#                 # transforms.RandomHorizontalFlip(p=0.5),
#                 transforms.RandomApply(
#                     [
#                         transforms.ColorJitter(
#                             brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
#                         )
#                     ],
#                     p=0.8,
#                 ),
#                 transforms.RandomGrayscale(p=0.2),
#                 transforms.RandomApply([GaussianBlurPIL([0.1, 0.5])], p=1.0),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=self.mean, std=self.std),
#             ]
#         )

#     def __call__(self, image):
#         crops = []
#         crops.append(self.aug1(image))
#         crops.append(self.aug2(image))
#         return crops
