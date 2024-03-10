"""
Defines the default data cacher class.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Union

from xai_torch.core.data.data_cachers.base import DataCacherBase
from xai_torch.core.factory.decorators import register_datacacher

if TYPE_CHECKING:
    import pandas as pd
    from datadings.reader import MsgpackReader


@register_datacacher(reg_name="default")
@dataclass
class DefaultDataCacher(DataCacherBase):
    """
    Handles the default pickle based dataset caching functionality.
    """

    @property
    def cache_file_path_df(self):
        return self.cache_file_path.with_suffix(".df")

    @property
    def cache_file_path_pickle(self):
        return self.cache_file_path.with_suffix(".df")

    def validate_cache(self):
        return self.cache_file_path_df.exists() or self.cache_file_path_pickle.exists()

    def _save_data(self, data: Union[pd.DataFrame, Iterable]):
        import pickle

        import pandas as pd

        # save the data
        if isinstance(data, pd.DataFrame):
            data.to_pickle(self.cache_file_path_df)
        else:
            with open(self.cache_file_path_pickle, "wb") as f:
                pickle.dump(data, f)

    def _load_data(self) -> Union[pd.DataFrame, MsgpackReader, Iterable]:
        import pickle

        import pandas as pd

        # load the data
        if self.cache_file_path_df.exists():
            return pd.read_pickle(self.cache_file_path_df)
        elif self.cache_file_path_pickle.exists():
            with open(self.cache_file_path_pickle, "rb") as f:
                return pickle.load(f)
