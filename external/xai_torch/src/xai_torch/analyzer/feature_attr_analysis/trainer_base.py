import copy
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import torch
from torchvision.transforms import Resize
from xai_torch.core.constants import DataKeys
from xai_torch.core.models.xai_model import XAIModel
from xai_torch.core.training.trainer import TrainerBase


class FeatureAttrTrainerBase(TrainerBase):
    @classmethod
    def initialize_prediction_engine(
        cls,
        model: XAIModel,
        transforms: List[Callable],
        collate_fn: Callable,
        device: Optional[Union[str, torch.device]] = torch.device("cpu"),
        orig_resize: Tuple[int, int] = (1024, 1024),
    ) -> Callable:
        from ignite.engine import Engine

        def step(
            engine: Engine, batch: Sequence[torch.Tensor]
        ) -> Union[Any, Tuple[torch.Tensor]]:
            """
            Define the computation step
            """

            from ignite.utils import convert_tensor

            # ready model for evaluation
            model.torch_model.eval()

            # here we first store the original images that are not transformed
            resize = Resize(orig_resize)
            original_image = copy.copy(batch[DataKeys.IMAGE])
            original_image = torch.stack([resize(x) for x in original_image])

            # here we first store the image file paths
            image_file_paths = None
            if DataKeys.IMAGE_FILE_PATH in batch:
                image_file_paths = batch[DataKeys.IMAGE_FILE_PATH]

            # now transform the images in the batch and apply collate fns to create tensors
            samples_list = [dict(zip(batch, t)) for t in zip(*batch.values())]
            for sample in samples_list:
                for transform in transforms:
                    sample = transform(sample)
            batch = collate_fn(samples_list)

            # put batch to device
            batch = convert_tensor(batch, device=device)

            # debugging
            # for key, val in batch.items():
            #     import numpy as np
            #     if isinstance(val, torch.Tensor):
            #         print(key, val.shape)
            #         if len(val.shape) == 4:
            #             import matplotlib.pyplot as plt
            #             plt.imshow(val[0].permute(1,2,0).cpu().numpy())
            #             plt.show()
            #         if len(val.shape) == 5:
            #             import matplotlib.pyplot as plt
            #             plt.imshow(val[0][0].permute(1,2,0).cpu().numpy())
            #             plt.show()

            # forward pass
            output = {
                **batch,
                DataKeys.PRED: model.torch_model.predict_step(
                    {
                        DataKeys.IMAGE: batch[DataKeys.IMAGE],
                        DataKeys.LABEL: batch[DataKeys.LABEL],
                    }
                ),
                DataKeys.ORIG_IMAGE: original_image,
            }
            if image_file_paths is not None:
                output = {**output, DataKeys.IMAGE_FILE_PATH: image_file_paths}

            return output

        return Engine(step)
