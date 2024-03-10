"""
Defines the factory for TrainValSampler class and its children.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Type

from xai_torch.core.models.xai_model import XAIModel
from xai_torch.utilities.logging_utils import DEFAULT_LOGGER_NAME

if TYPE_CHECKING:
    from xai_torch.core.args import Arguments
    from xai_torch.core.models.args import ModelArguments
    from xai_torch.core.models.base import BaseModule
    from xai_torch.core.models.xai_model import XAIModel


class ModelFactory:
    """
    The model factory that initializes the model based on its name, subtype, and
    training task.
    """

    @staticmethod
    def get_model_class(model_args: ModelArguments) -> Type[BaseModule]:
        """
        Find the model given the task and its name
        """
        from xai_torch.core.models.constants import MODELS_REGISTRY

        models_in_task = MODELS_REGISTRY.get(model_args.model_task, None)
        if models_in_task is None:
            raise ValueError(f"Task [{model_args.model_task}] is not supported.")
        model_class = models_in_task.get(model_args.name, None)
        if model_class is None:
            raise ValueError(f"Model [{model_args.model_task}/{model_args.name}] " "is not supported.")
        return model_class

    @staticmethod
    def create(
        args: Arguments,
        checkpoint: Optional[str] = None,
        strict: bool = False,
        wrapper_class=XAIModel,
        **model_kwargs,
    ) -> XAIModel:
        """
        Initialize the model
        """

        from pathlib import Path

        # get logger
        logger = logging.getLogger(DEFAULT_LOGGER_NAME)

        model_args = args.model_args
        model_class = ModelFactory.get_model_class(model_args)
        if checkpoint is None and model_args.pretrained:
            checkpoint = model_args.pretrained_checkpoint
        if checkpoint is not None and not str(checkpoint).startswith("http"):
            checkpoint = Path(checkpoint)
            if not checkpoint.exists():
                logger.warning(f"Checkpoint not found, cannot load weights from {checkpoint}.")
                checkpoint = None
        model = wrapper_class(args=args, model_class=model_class, **model_kwargs)

        if checkpoint is not None and checkpoint != "":
            logger.info(f"Loading model from checkpoint file [{checkpoint}] with strict [{strict}]")
            model.load_from_checkpoint(model, checkpoint_path=checkpoint, strict=strict)

        return model
