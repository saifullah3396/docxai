"""
Defines the random split sampling strategy.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from torch.utils.data.dataloader import DataLoader

from xai_torch.core.data.train_val_samplers.base import TrainValSampler
from xai_torch.core.factory.decorators import register_train_val_sampler


@register_train_val_sampler(reg_name="random_split")
@dataclass
class RandomSplitSampler(TrainValSampler):
    """
    Random split sampling strategy.
    """

    # The default seed used for random sampling
    seed: int = 42

    # The train/validation dataset split ratio
    random_split_ratio: float = 0.8

    def __call__(self, train_dataset: Dataset) -> typing.Tuple[DataLoader, DataLoader]:
        """
        Takes the training dataset as input and returns split train / validation
        sets based on the split ratio.
        """
        import torch
        from torch.utils.data.dataset import random_split

        train_dataset_size = len(train_dataset)
        val_dataset_size = int(train_dataset_size * round(1.0 - self.random_split_ratio, 2))
        train_set, val_set = random_split(
            train_dataset,
            [train_dataset_size - val_dataset_size, val_dataset_size],
            generator=torch.Generator().manual_seed(self.seed),
        )
        yield train_set, val_set
