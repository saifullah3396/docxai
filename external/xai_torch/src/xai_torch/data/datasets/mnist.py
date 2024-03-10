"""
Defines the MNIST dataset.
"""


from torchvision.datasets import MNIST
from xai_torch.core.constants import DataKeys
from xai_torch.core.data.datasets.image_dataset_base import ImageDatasetBase


class MNISTDataset(ImageDatasetBase):
    """MNIST dataset."""

    _is_downloadable = True
    _supported_splits = ["train", "test"]

    def _load_dataset_properties(self):
        super()._load_dataset_properties()

        self._labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    def _download_dataset(self):
        MNIST(
            self.dataset_dir,
            train=True if self.split in ["train", "val"] else False,
            download=True,
        )

    def _load_dataset(self):
        import pandas as pd

        # load the base dataset
        base_dataset = MNIST(
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
