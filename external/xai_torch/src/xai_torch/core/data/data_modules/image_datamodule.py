"""
Defines the base DataModule for image datasets.
"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from xai_torch.core.constants import DataKeys
from xai_torch.core.data.data_modules.base import BaseDataModule
from xai_torch.core.training.constants import TrainingStage


class ImageDataModule(BaseDataModule, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # make sure image data modules only work with childs of ImageDatasetBase
        from xai_torch.core.data.datasets.image_dataset_base import ImageDatasetBase

        assert issubclass(self.dataset_class, ImageDatasetBase)

    @property
    def gt_metadata(self):
        """Returns the data labels."""

        from torch.utils.data import Subset

        if hasattr(self, "train_dataset"):
            if isinstance(self.train_dataset, Subset):
                return dict(class_labels=self.train_dataset.dataset.labels)
            else:
                return dict(class_labels=self.train_dataset.labels)
        elif hasattr(self, "test_dataset"):
            if isinstance(self.test_dataset, Subset):
                return dict(class_labels=self.test_dataset.dataset.labels)
            else:
                return dict(class_labels=self.test_dataset.labels)
        else:
            raise RuntimeError(
                "You accessed labels property before calling datamodule.prepare_data() "
                "and datamodule.setup() on either train or test."
            )

    @property
    def class_labels(self) -> int:
        """Returns the number of data labels."""
        return self.gt_metadata["class_labels"]

    @property
    def num_labels(self) -> int:
        """Returns the number of data labels."""
        return len(self.class_labels)

    def show_images(self, batch, nmax=16, show=True):
        from matplotlib import pyplot as plt
        from torchvision.utils import make_grid

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks([])
        ax.set_yticks([])
        image_grid = make_grid((batch[DataKeys.IMAGE].detach()[:nmax]), nrow=4)
        if show:
            ax.imshow(image_grid.permute(1, 2, 0))
            plt.show()
        return image_grid

    def show_batch(self, stage: TrainingStage, nmax: int = 16, show=True):
        dataloader = self.get_dataloader(stage)
        for batch in dataloader:
            return self.show_images(batch, nmax=nmax, show=show)
