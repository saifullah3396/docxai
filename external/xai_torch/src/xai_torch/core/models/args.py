"""
Defines the dataclass for holding model arguments.
"""

import os
from dataclasses import dataclass, field
from functools import cached_property
from typing import Optional

from xai_torch.core.args_base import ArgumentsBase
from xai_torch.core.models.constants import MODELS_REGISTRY


@dataclass
class ModelArguments(ArgumentsBase):
    """
    Dataclass that holds the model arguments.
    """

    name: str = field(
        default="",
        metadata={
            "help": "The name of the model to use.",
            "choices": [e for e in MODELS_REGISTRY.keys()],
        },
    )
    model_task: str = field(
        default="image_classification",
        metadata={"help": "Training task for which the model is loaded."},
    )
    cache_dir: str = field(
        default=os.environ.get("XAI_TORCH_OUTPUT_DIR", "./") + "/pretrained/",
        metadata={"help": "The location to store pretrained or cached models."},
    )
    pretrained_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "Checkpoint file name to load the model weights from."},
    )
    pretrained: bool = field(
        default=True,
        metadata={"help": ("Whether to load the model weights if available.")},
    )
    checkpoint_state_dict_key: str = field(
        default="state_dict",
        metadata={"help": "The state dict key for checkpoint"},
    )
    config: dict = field(
        default_factory=lambda: {},
        metadata={"help": "The model configuration."},
    )
    convert_bn_to_gn: bool = field(
        default=False,
        metadata={"help": "If true, converts all batch norm layers to group norm."},
    )

    @cached_property
    def full_name(self):
        return f"{self.name}_{self.config.model_type}" if self.config.model_type != "" else self.name

    def __post_init__(self):
        from xai_torch.core.models.base_config import ModelConfig
        from xai_torch.core.models.factory import ModelFactory
        from xai_torch.utilities.dacite_wrapper import from_dict

        # update config
        model_class = ModelFactory.get_model_class(self)
        config_class = ModelConfig
        if hasattr(model_class, "Config"):
            config_class = model_class.Config
            if not issubclass(model_class.Config, ModelConfig):
                raise ValueError(
                    f"Model configuration [{model_class.Config}] must be a " f"child of the [{ModelConfig}] class."
                )
        if self.config is None:
            self.config = config_class()
        elif isinstance(self.config, dict):
            self.config = from_dict(
                data_class=config_class,
                data=self.config,
            )
