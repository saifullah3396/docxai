"""
Defines the base DataModule for text datasets.
"""

from abc import ABC
from functools import cached_property

from xai_torch.core.data.data_modules.base import BaseDataModule


class TextDataModule(BaseDataModule, ABC):
    @cached_property
    def gt_metadata(self):
        """Returns the data labels."""
        return dict(class_labels=self.dataset_class.LABELS)

    @cached_property
    def class_labels(self):
        """Returns the data labels."""
        return self.gt_metadata["class_labels"]

    @cached_property
    def num_labels(self) -> int:
        """Returns the number of data labels."""
        return len(self.class_labels)
