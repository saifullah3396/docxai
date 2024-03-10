from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from xai_torch.core.models.base import BaseModule


def register_model(task: str, reg_name: str = ""):
    def f(cls: BaseModule):

        from xai_torch.core.models.base import BaseModule
        from xai_torch.core.models.constants import MODELS_REGISTRY
        from xai_torch.utilities.general import str_to_underscored_lower

        try:
            if not issubclass(cls, BaseModule):
                raise ValueError("Only a sub-class of [{base_class_type}] can be registered as a [{base_class_type}].")

            if task not in MODELS_REGISTRY:
                MODELS_REGISTRY[task] = {}

            if reg_name != "":
                MODELS_REGISTRY[task][reg_name] = cls
            else:
                MODELS_REGISTRY[task][str_to_underscored_lower(cls.__name__)] = cls
            return cls
        except Exception as e:
            logging.exception(f"Exception raised while registering class [{cls}] as [{BaseModule}].")
            exit(1)

    return f
