from __future__ import annotations

import copy
from typing import Optional, Type, TypeVar, get_type_hints

from dacite.config import Config
from dacite.core import T, _build_value
from dacite.dataclasses import DefaultValueNotFoundError, get_default_value_for_field, get_fields
from dacite.exceptions import ForwardReferenceError, MissingValueError, WrongTypeError
from dacite.types import is_instance
from xai_torch.utilities.general import string_to_bool

T = TypeVar("T")


def update_dataclass(dataclass: Type[T], data: dict, config: Optional[Config] = None):
    """
    Updates the given dataclass from the arguments.
    """
    config = config or Config()
    fields = get_fields(dataclass)
    try:
        dataclass_hints = get_type_hints(dataclass)
    except NameError as error:
        raise ForwardReferenceError(str(error))

    for field in fields:
        if field.name not in data:
            continue

        field = copy.copy(field)
        field_data = data[field.name]
        if field.name not in dataclass_hints:
            field.type = type(field_data)
        else:
            field.type = dataclass_hints[field.name]
        try:
            if field.type == bool:
                field_data = string_to_bool(field_data)

            value = _build_value(type_=field.type, data=field_data, config=config)
            if config.check_types and not is_instance(value, field.type):
                raise WrongTypeError(field_path=field.name, field_type=field.type, value=value)
            setattr(dataclass, field.name, value)
        except KeyError:
            try:
                value = get_default_value_for_field(field)
            except DefaultValueNotFoundError:
                if not field.init:
                    continue
                raise MissingValueError(field.name)

        dataclass.__post_init__()
