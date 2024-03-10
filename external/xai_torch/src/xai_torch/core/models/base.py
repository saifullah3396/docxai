""" PyTorch module that defines the base model for training/testing etc. """

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Optional

from torch import nn

from xai_torch.core.data.utilities.typing import CollateFnDict
from xai_torch.core.training.constants import TrainingStage
from xai_torch.utilities.logging_utils import DEFAULT_LOGGER_NAME

if TYPE_CHECKING:
    from ignite.engine import Engine

    from xai_torch.core.args import Arguments


class BaseModule(nn.Module):
    def __init__(
        self,
        args: Arguments,
        **kwargs,
    ):
        super().__init__()

        # initialize arguments
        self._args = args
        self._logger = logging.getLogger(DEFAULT_LOGGER_NAME)

    @property
    def args(self):
        return self._args

    @property
    def model_args(self):
        return self._args.model_args

    @property
    def training_args(self):
        return self._args.training_args

    @property
    def config(self):
        return self._args.model_args.config

    @property
    def metrics(self):
        return self._metrics

    @metrics.setter
    def metrics(self, metrics):
        self._metrics = metrics

    @property
    def model_name(self):
        return self.model_args.full_name

    def init_metrics(self):
        self._metrics = self._init_metrics()

    def _init_metrics(self):
        return None

    def build_model(self):
        self._build_model()

    @abstractmethod
    def _build_model(self):
        pass

    def add_training_hooks(self, training_engine: Engine):
        self._training_engine = training_engine

    def get_param_groups(self):
        return {
            "default": list(self.parameters()),
        }

    def training_step(self, batch) -> None:
        return self(**batch, stage=TrainingStage.train)

    def evaluation_step(self, batch, stage: TrainingStage = TrainingStage.test) -> None:
        assert stage in [TrainingStage.train, TrainingStage.val, TrainingStage.test]
        return self(**batch, stage=stage)

    def predict_step(self, batch) -> None:
        return self(**batch, stage=TrainingStage.predict)

    def get_checkpoint_map(self, checkpoint):
        pass

    def on_load_checkpoint(self, checkpoint, strict: bool = True):
        state_dict_key = "state_dict"
        current_state_dict = self.state_dict()
        new_state_dict = {}
        unmatched_keys = []
        for state in checkpoint[state_dict_key]:
            current_state_name = state
            if current_state_name in current_state_dict:
                if current_state_dict[current_state_name].size() == checkpoint[state_dict_key][state].size():
                    new_state_dict[current_state_name] = checkpoint[state_dict_key][state]
                else:
                    unmatched_keys.append(state)
        if len(unmatched_keys) > 0:
            if strict:
                raise RuntimeError(
                    f"Found keys that are in the model state dict but their sizes don't match: {unmatched_keys}"
                )
            else:
                self._logger.warning(
                    f"Found keys that are in the model state dict but their sizes don't match: {unmatched_keys}"
                )
        return self.load_state_dict(new_state_dict, strict=False)

    @classmethod
    def get_data_collators(
        cls, args: Optional[Arguments] = None, data_key_type_map: Optional[dict] = None
    ) -> CollateFnDict:
        return CollateFnDict()
