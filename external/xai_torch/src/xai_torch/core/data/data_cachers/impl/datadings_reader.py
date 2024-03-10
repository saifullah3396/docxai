from __future__ import annotations

from typing import TYPE_CHECKING

from datadings.reader import MsgpackReader

# if TYPE_CHECKING:
#     from pathlib import Path

# from typing import Union

# from datadings.tools.msgpack import unpackb


class CustomMsgpackReader(MsgpackReader):
    pass
    # def __init__(self, path: Union[str, Path], buffering=0):
    #     super().__init__(path, buffering)
    #     self.__infile = None
    #     self._worker_in_files = {}

    # def _close(self):
    #     super()._close()

    #     for f in self._worker_in_files.values():
    #         f.close()

    # def get(self, index, yield_key=False, raw=False, copy=True):
    #     import torch

    #     # open the file if not already open. this allows not have to worry about pickling the file io object while also
    #     # keeping it threadsafe among workers > 0
    #     worker_info = torch.utils.data.get_worker_info()
    #     if worker_info is not None:
    #         if worker_info.id not in self._worker_in_files:
    #             self._worker_in_files[worker_info.id] = open(self._path, "rb", self._buffering)
    #     else:
    #         if self.__infile is None:
    #             self.__infile = open(self._path, "rb", self._buffering)

    #     f = self._infile
    #     pos = self._offsets
    #     offset = pos[index]
    #     n = pos[index + 1] - offset
    #     f.seek(offset, 0)
    #     data = f.read(n)
    #     if not raw:
    #         data = unpackb(data)
    #     if yield_key:
    #         return self._keys[index], data
    #     else:
    #         return data

    # @property
    # def _infile(self):
    #     import torch

    #     worker_info = torch.utils.data.get_worker_info()
    #     if worker_info is not None:
    #         if worker_info.id in self._worker_in_files:
    #             return self._worker_in_files[worker_info.id]
    #     return self.__infile
