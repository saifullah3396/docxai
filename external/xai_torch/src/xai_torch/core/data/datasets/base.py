"""
Defines the base class for Datasets.
"""
from __future__ import annotations

import logging
import sys
from typing import Callable, Iterable, List, Optional, Union

import pandas as pd
from torch.utils.data import Dataset

from xai_torch.core.constants import DataKeys
from xai_torch.core.data.tokenizers.base import Tokenizer
from xai_torch.utilities.logging_utils import DEFAULT_LOGGER_NAME
import h5py

class DatasetBase(Dataset):
    """
    Defines the base dataset class for custom functionality.

    Args:
        split: Dataset split from ['train', 'test', 'val']
        transforms: Transforms to be applied to the data.
        indices: Custom subset indices to be used instead of the
            original indices.
    """

    _is_downloadable = False
    _supported_splits = []

    def __init__(
        self,
        dataset_name: str,
        dataset_dir: str,
        split: str,
        transforms: Optional[Callable] = None,
        data_cacher_wrapper: Optional[Callable] = None,
        tokenizer: Optional[Tokenizer] = None,
        indices: list = [],
        quiet: bool = False,
        show_transforms: bool = False,
    ):
        from pathlib import Path

        # set arguments
        self._dataset_name = dataset_name
        self._split = split
        self._transforms = transforms
        self._tokenizer = tokenizer
        self._indices = indices
        self._quiet = quiet
        self._show_transforms = show_transforms
        self._logger = logging.getLogger(DEFAULT_LOGGER_NAME)

        # setup dataset_dir, this could be a url
        self._dataset_dir = Path(dataset_dir)

        # initialize the main data holder object
        self._data = None

        # validate split arguments
        self.validate_split()

        # initialize data cacher
        self._data_cacher = data_cacher_wrapper(self)

    @property
    def name(self):
        return self._dataset_name

    @property
    def print_name(self):
        return f"{self._dataset_name}-{self.split}"

    @property
    def data(self):
        return self._data

    @property
    def split(self):
        return self._split

    @property
    def dataset_dir(self):
        return self._dataset_dir

    @property
    def data_cacher(self):
        return self._data_cacher

    @property
    def data_prepared(self):
        return self.data_cacher.validate_cache()

    @property
    def transforms(self):
        return self._transforms

    @transforms.setter
    def transforms(self, transforms: Callable):
        self._transforms = transforms

    @property
    def tokenizer(self):
        return self._tokenizer

    # @property
    # def tokenize_per_sample(self):
    #     return (
    #         self._data_args.data_tokenizer_args is not None
    #         and not self._data_args.data_tokenizer_args.tokenize_per_sample
    #     )

    @property
    def indices(self):
        return self._indices

    @indices.setter
    def indices(self, indices: List[int]):
        self._indices = indices

    @classmethod
    def get_supported_splits(cls):
        return cls._supported_splits

    def validate_split(self):
        """
        Validates the presence of requested split.
        """

        if self.split not in self._supported_splits:
            if self.split == "val" and "train" in self._supported_splits and len(self._indices) == 0:
                raise ValueError(f"Split argument [{self.split}] is not supported.")

    def _download_dataset(self) -> str:
        """
        Performs the dataset downloading routine. It must return the directory where
        the dataset is downloaded to.
        """
        raise NotImplementedError()

    def download_dataset(self) -> str:
        """
        Performs the dataset downloading routine. It must return the directory where
        the dataset is downloaded to.
        """

        if not self._quiet:
            self._logger.info(f"Downloading the dataset [{self._dataset_name}-{self.split}]...")
        self._download_dataset()

    def load_dataset_from_dir(self):
        if not self._quiet:
            # log info
            self._logger.info(
                f"Initializing the dataset [{self._dataset_name}-{self.split}] " f"from directory: {self._dataset_dir}."
            )

        # download the dataset if required
        if self._is_downloadable:
            self.download_dataset()

        # validate directory
        if not self.dataset_dir.exists():
            raise ValueError(f"Could not find the dataset directory: {self.dataset_dir}")

        # load the dataset
        return self._load_dataset()

    # def handle_tokenization(self, data: Union[pd.DataFrame, Iterable]):
    #     if self.tokenize_per_sample:
    #         if not self._quiet:
    #             self._logger.info(f"Tokenizing the dataset...")
    #         return self._tokenize_dataset(data)
    #     return data

    def load(self):
        """
        Loads the dataset, from cache if present, otherwise loads it from directory or
        downloads it. Also tokenizes the dataset if required.
        """
        import ignite.distributed as idist
        import pandas as pd

        try:
            # this gets called everytime so that dataset properties that are not to be cached can be computed directly
            self._load_dataset_properties()

            # try to load data from cache
            self._data = self._load_from_cache()

            # print data transforms
            if self.transforms is not None and self._show_transforms:
                if not self._quiet:
                    self._logger.info(f"[{self._dataset_name}-{self.split}] " f"Defining data transformations:")
                    if idist.get_rank() == 0:
                        from xai_torch.core.data.augmentations.core_transforms import DictTransform

                        if isinstance(self.transforms, DictTransform):
                            for k in self.transforms.keys:
                                print(f"{k}:\n", self.transforms.transform)
                        else:
                            print(self.transforms)

            if not self._quiet and self._data is not None and isinstance(self.data, pd.DataFrame):
                pd.set_option("display.max_columns", 10)
                self._logger.debug(f"Dataset:\n{self.data.head(5)}")
        except Exception as e:
            self._logger.exception(f"Exception raised while loading dataset " f"[{self._dataset_name}]: {e}")
            sys.exit(1)

    def _load_dataset_properties(self):
        """
        This must be defined in the child dataset class to actually load the dataset related parameters. This function
        gets called on every load, so this should be used if you have anything you don't wish to cache such as labels,
        weights, etc.
        """

    def _load_dataset(self):
        """
        This must be defined in the child dataset class to actually load the dataset.
        """
        raise NotImplementedError()

    def __len__(self):
        """
        Returns the total size of the dataset.
        """

        # if data is not defined just return an error
        if self.data is None:
            raise ValueError(f"No data loaded in the dataset: {self._dataset_name}")

        # set _indices if indices are available
        if len(self._indices) > 0:
            return len(self._indices)

        if isinstance(self.data, pd.DataFrame):
            return self.data.shape[0]
        elif isinstance(self.data, h5py.File):
            return len(self.data[DataKeys.INDEX])
        else:
            return len(self.data)

    def get_sample(self, idx):
        import pandas as pd
        from datadings.reader import MsgpackReader

        if isinstance(self.data, pd.DataFrame):
            return self.data.iloc[idx].to_dict()
        elif isinstance(self.data, MsgpackReader):
            import pickle

            return pickle.loads(self.data.get(idx)["data"])
        elif isinstance(self.data, h5py.File):
            sample = {}
            for key in self.data.keys():
                sample[key] = self.data[key][idx]
            return sample

    def __getitem__(self, idx):
        """
        Returns the dataset sample at the given index.

        Args:
            idx: sample index
        """

        # if subset indices are defined then first correct the index
        if len(self._indices) > 0:
            idx = self._indices[idx]

        # get the sample from the cached data if needed
        sample = self.get_sample(idx)

        # tokenize the sample if needed
        # if self.tokenize_per_sample:
        #     sample = self._tokenize_sample(sample)

        # perform transformations on data if required
        if self.transforms is not None:
            sample = self.transforms(sample)

        return {DataKeys.INDEX: idx, **sample}

    def _tokenize_dataset(self, data: Union[pd.DataFrame, Iterable]):
        """
        This must be defined in the child class to tokenize the dataset if required.
        """
        raise NotImplementedError()

    def _tokenize_sample(self, sample: Union[tuple, dict]):
        """
        This must be defined in the child class to tokenize the dataset per sample
        if required.
        """
        raise NotImplementedError()

    def _save_to_cache(self, data):
        """
        Saves the data to cache according to required caching functionality
        """
        self.data_cacher.save_to_cache(data)

    def _load_from_cache(self):
        """
        Loads the data from cache according to required caching functionality
        """
        if not self._quiet:
            self._logger.info(f"Loading dataset [{self.print_name}] from cache.")

        if not self.data_prepared:
            # load the dataset from the dataset dir
            self._data = self.load_dataset_from_dir()

            # tokenizing dataset if required
            # self._data = self.handle_tokenization(self._data)

            # save data to cache
            self._save_to_cache(self._data)
        return self.data_cacher.load_from_cache()
