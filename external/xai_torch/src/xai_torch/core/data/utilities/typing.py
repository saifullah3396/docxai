import typing
from dataclasses import dataclass


@dataclass
class CollateFnDict:
    train: typing.Optional[typing.Callable] = None
    val: typing.Optional[typing.Callable] = None
    test: typing.Optional[typing.Callable] = None


@dataclass
class TransformsDict:
    train: typing.Optional[typing.Callable] = None
    val: typing.Optional[typing.Callable] = None
    test: typing.Optional[typing.Callable] = None
