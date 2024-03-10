"""
Defines the datadings based data cacher class.
"""

import logging
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Union

import pandas as pd
from datadings.reader import MsgpackReader

from xai_torch.core.constants import DataKeys
from xai_torch.core.data.data_cachers.base import DataCacherBase
from xai_torch.core.factory.decorators import register_datacacher
from xai_torch.utilities.logging_utils import DEFAULT_LOGGER_NAME


@register_datacacher(reg_name="datadings")
@dataclass
class DatadingsCacher(DataCacherBase):
    """
    Handles the dataset caching functionality based on datadings msgpack writer/reader.
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
    cache_encoded_images: bool = field(
        default=True,
        metadata={"help": "Whether to encode images before caching."},
    )
    workers: int = field(default=1, metadata={"help": "Number of workers for parallel caching."})
    load_data_to_ram: Optional[bool] = field(
        default=False,
        metadata={"help": ("Whether to load data into RAM or read directly from datadings.")},
    )

    def __post_init__(self):
        self._logger = logging.getLogger(DEFAULT_LOGGER_NAME)

    @property
    def cache_file_path_msgpack(self):
        return self.cache_file_path.with_suffix(".msgpack")

    @property
    def cache_meta_file_path_with_ext(self):
        return self.cache_file_path.with_suffix(".msgpack.metadata")

    def save_dataset_meta(self, data: pd.DataFrame):
        import pickle

        sample = data.iloc[0].to_dict()
        dataset_meta = {"size": len(data), "keys": [DataKeys.INDEX, *list(sample.keys())]}
        with open(self.cache_meta_file_path_with_ext, "wb") as f:
            pickle.dump(dataset_meta, f)

    def load_dataset_meta(self):
        import pickle

        with open(self.cache_meta_file_path_with_ext, "rb") as f:
            return pickle.load(f)

    def validate_cache(self):
        from xai_torch.core.data.data_cachers.impl.datadings_reader import CustomMsgpackReader

        # make sure cache files exist
        if not (self.cache_meta_file_path_with_ext.exists() and self.cache_file_path_msgpack.exists()):
            return False

        # make sure the cached data and its metadata align
        data_reader = CustomMsgpackReader(self.cache_file_path_msgpack)
        dataset_meta = self.load_dataset_meta()
        if len(data_reader) != dataset_meta["size"]:
            return False
        return True

    def setup_image(self, image_file_path: str):
        import cv2

        if self.cache_resized_images:
            image = cv2.imread(image_file_path, cv2.IMREAD_ANYCOLOR)
            image = cv2.resize(image, self.cache_image_size)
            if self.cache_encoded_images:
                image = cv2.imencode(".png", image)[1].tobytes()
            return image
        else:
            try:
                with open(image_file_path, "rb") as f:
                    return f.read()
            except Exception as e:
                self._logger.exception(f"Exception raised while loading image [{image_file_path}]: {e}")
                exit(1)

    def sample_generator(self, data: Union[pd.DataFrame, Iterable], ignore_samples_to_idx: int):
        import pandas as pd

        if isinstance(data, pd.DataFrame):
            for idx, _ in data.iterrows():
                if idx < ignore_samples_to_idx:
                    yield idx, None
                else:
                    yield idx, data.iloc[idx].to_dict()
        else:
            for idx, _ in data:
                if idx < ignore_samples_to_idx:
                    yield idx, None
                else:
                    yield idx, data[idx]

    def sample_preprocessor(self, sample: dict):
        import pickle

        idx, sample = sample
        if sample is None:
            return None

        # see if image file path is in the sample
        if self.cache_images and DataKeys.IMAGE_FILE_PATH in sample:
            sample[DataKeys.IMAGE] = self.setup_image(sample[DataKeys.IMAGE_FILE_PATH])

        return {"key": str(idx), "data": pickle.dumps(sample)}

    def _save_data(self, data: pd.DataFrame):
        from multiprocessing.pool import ThreadPool

        import tqdm

        from xai_torch.core.data.data_cachers.impl.datadings_reader import CustomMsgpackReader
        from xai_torch.core.data.data_cachers.impl.datadings_writer import DatadingsFileWriter

        try:
            # save dataset meta info
            self.save_dataset_meta(data)

            # open existing data file if available
            reader = None
            if self.cache_file_path_msgpack.exists():
                reader = CustomMsgpackReader(self.cache_file_path_msgpack)

            # find the size of already saved data to continue from it
            cached_data_size = len(reader) if reader is not None else 0

            # make a samples generator function
            sample_generator = self.sample_generator(data, cached_data_size)

            # create writier instnace
            writer = DatadingsFileWriter(self.cache_file_path_msgpack, overwrite=False)

            if 'test' in str(self.cache_file_path_msgpack):
                with writer:
                    self._logger.info(
                        f"Writing  dataset [{self.dataset.print_name}] to a datadings " "file. This might take a while..."
                    )
                    for sample in tqdm.tqdm(sample_generator):
                        sample = self.sample_preprocessor(sample)
                        if sample:
                            writer.write({**sample})
            else:
                # create a thread pool for parallel writing
                pool = ThreadPool(self.workers)
                with writer:
                    self._logger.info(
                        f"Writing  dataset [{self.dataset.print_name}] to a datadings " "file. This might take a while..."
                    )
                    progress_bar = tqdm.tqdm(
                        pool.imap_unordered(self.sample_preprocessor, sample_generator),
                        total=len(data),
                    )
                    for sample in progress_bar:
                        if sample:
                            writer.write({**sample})

                            # note: progress bar might be slightly off with
                            # multiple processes
                            progress_bar.update(1)
        except KeyboardInterrupt as exc:
            self._logger.error(f"Data caching interrupted. Exiting...")
            exit(1)

    def read_all_data(self, data_reader: MsgpackReader):
        import pandas as pd
        from tqdm import tqdm

        self._logger.info("Loading all data into RAM. This might take a while...")
        data = []
        for idx in tqdm(range(len(data_reader))):
            sample = data_reader.get(idx)
            data.append(sample)
        return pd.DataFrame(data)

    def _load_data(self):
        from xai_torch.core.data.data_cachers.impl.datadings_reader import CustomMsgpackReader

        reader = CustomMsgpackReader(self.cache_file_path_msgpack)
        if self.load_data_to_ram:
            return self.read_all_data(reader)
        return reader
