"""
Defines the datadings based data cacher class.
"""

import logging
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Union

import pandas as pd
import h5py

from xai_torch.core.constants import DataKeys
from xai_torch.core.data.data_cachers.base import DataCacherBase
from xai_torch.core.factory.decorators import register_datacacher
from xai_torch.utilities.logging_utils import DEFAULT_LOGGER_NAME
import numpy as np

def update_dataset_at_indices(
    hf: h5py.File, key: str, indices: np.array, data, maxshape=(None,)
):
    if key not in hf:
        hf.create_dataset(key, data=data, compression=None, maxshape=maxshape)
    else:
        max_len = indices.max() + 1
        if len(hf[key]) < max_len:
            hf[key].resize((indices.max() + 1), axis=0)
        hf[key][indices] = data

@register_datacacher(reg_name="hdf5")
@dataclass
class HDF5Cacher(DataCacherBase):
    """
    Handles the dataset caching functionality based on HDF5 format.
    """

    cache_images: bool = field(
        default=False,
        metadata={"help": "Whether to cache the images at all."},
    )
    cache_resized_images: bool = field(
        default=False,
        metadata={"help": "Whether to resize them before caching."},
    )
    cache_image_size: List[int] = field(
        default_factory=lambda: [224, 224],
        metadata={"help": ("Output image size, if resizing is required during caching.")},
    )
    batch_size: int = field(
        default=1024,
        metadata={"help": "Caching batch size."},
    )
    load_data_to_ram: Optional[bool] = field(
        default=False,
        metadata={"help": ("Whether to load data into RAM or read directly from datadings.")},
    )

    def __post_init__(self):
        self._logger = logging.getLogger(DEFAULT_LOGGER_NAME)

    @property
    def cache_file_path_hdf5(self):
        return self.cache_file_path.with_suffix(".h5")

    @property
    def cache_meta_file_path_with_ext(self):
        return self.cache_file_path.with_suffix(".h5.metadata")

    def save_dataset_meta(self, data: pd.DataFrame):
        import pickle

        sample = data.iloc[0].to_dict()
        dataset_meta = {"size": len(data), "keys": [DataKeys.INDEX, *list(sample.keys())]}
        if self.cache_images and DataKeys.IMAGE_FILE_PATH in data.columns:
            dataset_meta['keys'].append(DataKeys.IMAGE)
        with open(self.cache_meta_file_path_with_ext, "wb") as f:
            pickle.dump(dataset_meta, f)

    def load_dataset_meta(self):
        import pickle

        with open(self.cache_meta_file_path_with_ext, "rb") as f:
            return pickle.load(f)

    def validate_cache(self):
        import h5py

        # make sure cache files exist
        if not self.cache_file_path_hdf5.exists():
            return False

        # make sure the cached data and its metadata align
        dataset_meta = self.load_dataset_meta()
        hf = h5py.File(self.cache_file_path_hdf5, 'r')

        # make sure both have same keys
        if len(set(hf.keys()).intersection(set(dataset_meta['keys']))) != len(dataset_meta['keys']):
            return False
        if len(hf[DataKeys.INDEX]) != dataset_meta["size"]:
            return False
        return True

    def setup_image(self, row):
        import imageio
        import cv2

        image = imageio.imread(row[DataKeys.IMAGE_FILE_PATH])
        if self.cache_resized_images:
            image = cv2.resize(image, self.cache_image_size)
        return image

    def _save_data(self, data: pd.DataFrame):
        import tqdm
        try:
            # save dataset meta info
            self.save_dataset_meta(data)

            hf = h5py.File(self.cache_file_path_hdf5, 'a')
            if DataKeys.INDEX in hf:
                start_idx = len(hf[DataKeys.INDEX])
            else:
                start_idx = 0

            self._logger.info(
                f"Writing  dataset [{self.dataset.print_name}] to a datadings " "file. This might take a while..."
            )

            for g, df in tqdm.tqdm(data.groupby(np.arange(len(data[start_idx:])) // self.batch_size)):
                if self.cache_images and DataKeys.IMAGE_FILE_PATH in df.columns:
                    df[DataKeys.IMAGE] = df.apply(self.setup_image, axis=1)
                samples = df.to_dict(orient='list')
                update_dataset_at_indices(hf, DataKeys.INDEX, np.array(df.index), df.index, maxshape=(None,))
                for key, value in samples.items():
                    if key == DataKeys.IMAGE:
                        value = np.array(value)
                    if isinstance(value, np.ndarray):
                        maxshape = (None, *value.shape[1:])
                    elif isinstance(value, list):
                        maxshape = (None,)
                    update_dataset_at_indices(hf, key, np.array(df.index), value, maxshape=maxshape)
            hf.close()
        except KeyboardInterrupt as exc:
            self._logger.error(f"Data caching interrupted. Exiting...")
            exit(1)

    def _load_data(self):
        hf = h5py.File(self.cache_file_path_hdf5, 'r')
        return hf
