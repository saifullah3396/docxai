"""
Defines the dataclass for holding data augmentation / transformation arguments.
"""

from dataclasses import dataclass, field
from typing import List, Union

from xai_torch.core.args_base import ArgumentsBase


@dataclass
class DataAugmentationArguments(ArgumentsBase):
    """
    Dataclass that holds the train/validation sampling arguments.
    """

    # Data augmentation strategy to use
    strategy: str = ""

    # Strategy kwargs
    config: Union[dict, List[dict]] = field(default_factory=lambda: {})

    # The data keys on which to apply the augmentations.
    keys: List[str] = field(default_factory=lambda: [])
