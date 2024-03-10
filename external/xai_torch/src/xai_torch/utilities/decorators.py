from __future__ import annotations

import logging
from typing import Any, Type


def register_as_child(base_class_type: Type[Any], registry: dict, reg_name: str = ""):
    def f(cls: base_class_type):
        from xai_torch.utilities.general import str_to_underscored_lower

        try:
            if not issubclass(cls, base_class_type):
                raise ValueError("Only a sub-class of [{base_class_type}] can be registered as a [{base_class_type}].")

            if reg_name != "":
                registry[reg_name] = cls
            else:
                registry[str_to_underscored_lower(cls.__name__)] = cls
            return cls
        except Exception as e:
            logging.exception(f"Exception raised while registering class [{cls}] as [{base_class_type}].")
            exit(1)

    return f


def import_modules(names):
    def f(cls: Any):
        import importlib

        for module_name in names:
            try:
                importlib.import_module(module_name)
            except Exception as e:
                logging.exception(
                    f"Exception raised while importing module [{module_name}] " f"for class [{cls.__name__}]: {e}"
                )
        return cls

    return f
