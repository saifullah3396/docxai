from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

if TYPE_CHECKING:
    import torch

from fsspec.core import url_to_fs
from fsspec.implementations.local import AbstractFileSystem


def get_filesystem(path: Path, **kwargs: Any) -> AbstractFileSystem:
    fs, _ = url_to_fs(str(path), **kwargs)
    return fs


def load(
    path_or_url: Union[str, Path],
    map_location: Optional[
        Union[str, Callable, torch.device, Dict[Union[str, torch.device], Union[str, torch.device]]]
    ] = None,
) -> Any:
    """Loads a checkpoint.

    Args:
        path_or_url: Path or URL of the checkpoint.
        map_location: a function, ``torch.device``, string or a dict specifying how to remap storage locations.
    """
    import torch

    if not isinstance(path_or_url, (str)):
        # any sort of BytesIO or similar
        return torch.load(path_or_url, map_location=map_location)
    if str(path_or_url).startswith("http"):
        return torch.hub.load_state_dict_from_url(str(path_or_url), map_location=map_location)
    fs = get_filesystem(path_or_url)
    with fs.open(path_or_url, "rb") as f:
        return torch.load(f, map_location=map_location)


def filter_keys(checkpoint, state_dict_key: str, keys: List[str]):
    checkpoint_filtered = {state_dict_key: {}}
    for state in checkpoint[state_dict_key]:
        updated_state = state
        for key in keys:
            if key in updated_state:
                updated_state = updated_state.replace(key, "")
        checkpoint_filtered[state_dict_key][updated_state] = checkpoint[state_dict_key][state]
    return checkpoint_filtered


def prepend_keys(checkpoint, state_dict_key: str, keys: List[str]):
    checkpoint_prepended = {state_dict_key: {}}
    for state in checkpoint[state_dict_key]:
        updated_state = state
        for key in keys:
            if key not in updated_state:
                updated_state = key + updated_state

        checkpoint_prepended[state_dict_key][updated_state] = checkpoint[state_dict_key][state]
    return checkpoint_prepended
