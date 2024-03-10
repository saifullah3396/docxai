"""
Defines the data cacher base classe.
"""

import logging
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Iterable, Optional, Union

import pandas as pd
from datadings.reader import MsgpackReader

from xai_torch.core.data.datasets.base import DatasetBase
from xai_torch.utilities.abstract_dataclass import AbstractDataclass
from xai_torch.utilities.logging_utils import DEFAULT_LOGGER_NAME


@dataclass
class DataCacherBase(AbstractDataclass):
    dataset: DatasetBase = field(default=None, metadata={"help": "The dataset to cache."})
    dataset_cache_dir: Optional[str] = field(default=None, metadata={"help": "Directory to store dataset caches."})
    cached_data_name: Optional[str] = field(
        default="cached_data",
        metadata={"help": ("The name of the dataset cache file.")},
    )

    def __post_init__(self):
        self._logger = logging.getLogger(DEFAULT_LOGGER_NAME)

    @property
    def cache_file_path(self):
        from pathlib import Path

        cache_file_name = self.cached_data_name
        # if (
        #     self.dataset._data_args.data_tokenizer_args is not None
        #     and not self.dataset.tokenize_per_sample
        #     and self.dataset.tokenizer
        # ):
        #     cache_file_name += f"{self.dataset.tokenizer.name}"

        return Path(self.dataset_cache_dir) / self.dataset.name / self.dataset.split / cache_file_name

    @abstractmethod
    def validate_cache(self):
        pass

    @abstractmethod
    def _save_data(self, data: Union[pd.DataFrame, Iterable]):
        pass

    @abstractmethod
    def _load_data(self) -> Union[pd.DataFrame, MsgpackReader, Iterable]:
        pass

    def save_to_cache(self, data: Union[pd.DataFrame, Iterable]):
        try:
            from xai_torch.utilities.general import make_dir

            # log info
            self._logger.info(f"Saving dataset [{self.dataset.print_name}] to cache.")

            # make target directory if not available
            make_dir(self.cache_file_path.parent)

            # perform saving operation
            self._save_data(data)
        except Exception as e:
            self._logger.exception(
                f"Exception raised while saving dataset [{self.dataset.print_name}] "
                f"to cache file [{self.cache_file_path}]: {e}"
            )
            exit(1)

    def load_from_cache(self) -> Union[pd.DataFrame, MsgpackReader, Iterable]:
        try:
            if not self.validate_cache():
                return None
            return self._load_data()
        except Exception as e:
            self._logger.exception(
                f"Exception raised while loading dataet [{self.dataset.print_name}] "
                f"from cache file [{self.cache_file_path}]: {e}"
            )
            exit(1)
