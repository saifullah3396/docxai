"""
Defines the base class for Image related Datasets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from xai_torch.core.constants import DataKeys
from xai_torch.core.data.datasets.base import DatasetBase

if TYPE_CHECKING:
    pass


class ImageDatasetBase(DatasetBase):
    _labels = []

    def __init__(self, *args, load_images=True, **kwargs):
        super().__init__(*args, **kwargs)

        self._labels = []
        self._load_images = load_images

    @property
    def labels(self):
        if len(self._labels) == 0:
            self._logger.warning(f"You did not define labels for this dataset. Current labels: {self._labels}")
        return self._labels

    def _load_dataset_properties(self):
        self._labels = []

    def read_image_from_sample(self, sample):
        import pandas as pd
        from datadings.reader import MsgpackReader
        import h5py

        if isinstance(self.data, pd.DataFrame):
            import cv2
            import imageio

            if DataKeys.IMAGE in sample:
                return sample[DataKeys.IMAGE]
            elif DataKeys.IMAGE_FILE_PATH in sample:
                image_file_path = sample[DataKeys.IMAGE_FILE_PATH]
                # image = cv2.imread(image_file_path, cv2.IMREAD_ANYCOLOR)
                image = imageio.imread(image_file_path)

                # # if its a 3 channel image convert to RGB
                # if len(image.shape) == 3:
                #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return image
        elif isinstance(self.data, MsgpackReader):
            import cv2
            import numpy as np
            from PIL.Image import Image as PILImage

            if DataKeys.IMAGE in sample:
                if isinstance(sample[DataKeys.IMAGE], np.ndarray) or isinstance(sample[DataKeys.IMAGE], PILImage):
                    return sample[DataKeys.IMAGE]
                else:
                    image = cv2.imdecode(
                        np.fromstring(sample[DataKeys.IMAGE], dtype="uint8"),
                        cv2.IMREAD_ANYCOLOR,
                    )
                    # if its a 3 channel image convert to RGB
                    if len(image.shape) == 3:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    return image
            elif DataKeys.IMAGE_FILE_PATH in sample:
                image_file_path = sample[DataKeys.IMAGE_FILE_PATH]
                image = cv2.imread(image_file_path, cv2.IMREAD_ANYCOLOR)

                # if its a 3 channel image convert to RGB
                if len(image.shape) == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return image
        elif isinstance(self.data, h5py.File):
            import imageio
            import numpy as np
            from PIL.Image import Image as PILImage

            if DataKeys.IMAGE in sample:
                return sample[DataKeys.IMAGE]
            elif DataKeys.IMAGE_FILE_PATH in sample:
                image_file_path = sample[DataKeys.IMAGE_FILE_PATH]
                image = imageio.imread(image_file_path)
                return image

    def get_image(self, sample):
        import torch
        from torchvision.transforms.functional import to_tensor

        if not self._load_images:
            return None

        try:
            # read image according to the available data
            image = self.read_image_from_sample(sample)

            # convert any type to tensor
            image = to_tensor(image)

            # add a channel to image if not present
            if len(image.shape) == 2:
                image = torch.unsqueeze(image, 0)

            # return image
            return image
        except Exception as e:
            self._logger.exception(
                f"Exception raised while getting image for sample {sample}: ",
                e,
            )
            exit(1)

    def get_sample(self, idx):
        import torch

        if isinstance(idx, torch.Tensor):
            idx = idx.item()

        # get the sample
        sample = super().get_sample(idx)

        # read image
        image = self.get_image(sample)

        # set image and return
        if image is not None:
            sample[DataKeys.IMAGE] = image
        else:
            if DataKeys.IMAGE in sample:
                sample.pop(DataKeys.IMAGE)
        return sample
