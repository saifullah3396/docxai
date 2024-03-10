"""
Defines the CIFAR10 dataset.
"""


from torchvision.datasets import CIFAR10
from xai_torch.core.constants import DataKeys
from xai_torch.core.data.datasets.image_dataset_base import ImageDatasetBase


class CIFAR10Dataset(ImageDatasetBase):
    """CIFAR10 dataset."""

    _is_downloadable = True
    _supported_splits = ["train", "test"]

    def _load_dataset_properties(self):
        super()._load_dataset_properties()

        self._labels = [
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]

    def _download_dataset(self):
        CIFAR10(
            self.dataset_dir,
            train=True if self.split in ["train", "val"] else False,
            download=True,
        )

    def _load_dataset(self):
        import pandas as pd

        # load the base dataset
        base_dataset = CIFAR10(
            self.dataset_dir,
            train=True if self.split in ["train", "val"] else False,
            download=False,
        )

        # load the dataset into list of lists
        data = []
        for (img, target) in base_dataset:
            data.append([img, target])

        # create pandas dataframe to hold the dataset
        self.data_columns = [DataKeys.IMAGE, DataKeys.LABEL]
        return pd.DataFrame(data, columns=self.data_columns)
