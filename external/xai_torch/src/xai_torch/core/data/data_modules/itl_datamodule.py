"""
Defines the base DataModule for image-text-layout multimodal datasets.
"""

from __future__ import annotations

import logging
from abc import ABC
from functools import cached_property
from typing import TYPE_CHECKING, Optional

from xai_torch.core.constants import DataKeys
from xai_torch.core.data.data_modules.base import BaseDataModule
from xai_torch.core.data.utilities.typing import CollateFnDict, TransformsDict
from xai_torch.core.training.constants import TrainingStage
from xai_torch.utilities.logging_utils import DEFAULT_LOGGER_NAME

if TYPE_CHECKING:
    from xai_torch.core.args import Arguments


class ImageTextLayoutDataModule(BaseDataModule, ABC):
    def __init__(
        self,
        args: Arguments,
        collate_fns: CollateFnDict = CollateFnDict(),
        transforms: TransformsDict = TransformsDict(),
        n_partitions: Optional[int] = None,
        partition_id: Optional[int] = None,
    ):
        super().__init__(args, collate_fns, transforms, n_partitions, partition_id)

    @cached_property
    def gt_metadata(self):
        return dict(class_labels=self.dataset_class.labels)

    @cached_property
    def class_labels(self):
        return self.gt_metadata["class_labels"]

    @cached_property
    def class_num_labels(self) -> int:
        return len(self.class_labels)

    def show_images(self, batch, nmax=16, show=True):
        from matplotlib import pyplot as plt
        from torchvision.utils import make_grid

        image_grid = make_grid((batch[:nmax]), nrow=4)
        if show:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(image_grid.permute(1, 2, 0))
            plt.show()
        return image_grid

    def show_batch(self, pl_loggers, stage: TrainingStage, nmax: int = 16, show=True):
        import cv2
        import numpy as np
        import torch

        # get logger
        logger = logging.getLogger(DEFAULT_LOGGER_NAME)

        # first we save data without collation by taking n samples
        dataloader = self.get_dataloader(stage)

        # create a batch
        batch = []
        for idx in range(nmax):
            sample = dataloader.dataset[idx]
            batch.append(sample)

        # give some info about the batch
        logger.info(f"Keys present in the un-collated dataset batch {batch[0].keys()}")

        # get collated batch
        collate_batch = next(iter(dataloader))
        logger.info(f"Keys present in the collated dataset batch {collate_batch.keys()}")

        draw_batch = []
        for sample in batch:
            image = sample[DataKeys.IMAGE].permute(1, 2, 0).cpu().numpy()
            image = np.ascontiguousarray(image)
            h, w, c = image.shape

            for word, box in zip(
                sample[DataKeys.WORDS], sample[DataKeys.WORD_BBOXES]
            ):  # each box is [x1,y1,x2,y2] normalized
                p1 = (int(box[0] * w), int(box[1] * h))
                p2 = (int(box[2] * w), int(box[3] * h))
                cv2.rectangle(image, p1, p2, (255, 0, 0), 1)
                cv2.putText(
                    image,
                    text=word,
                    org=p1,
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=1.5,
                    color=(0, 0, 255),
                    thickness=1,
                )
            draw_batch.append(torch.from_numpy(image).permute(2, 0, 1))

        # draw images
        self.show_images(draw_batch, nmax=nmax, show=True)
