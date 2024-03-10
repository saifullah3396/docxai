"""
Defines the dataclass for holding data cacher arguments.
"""
from dataclasses import dataclass, field
from typing import Optional

from xai_torch.core.args_base import ArgumentsBase


@dataclass
class DataCacherArguments(ArgumentsBase):
    """
    Dataclass that holds the train/validation sampling arguments.
    """

    # Caching Strategy
    strategy: str = "default"

    # Strategy config
    config: Optional[dict] = None
