"""Defines generic utility functions related to file read/write operations.
"""
from __future__ import annotations

import os
import re
from typing import Any

import yaml

_var_matcher = re.compile(r"\${([^}^{]+)}")
_tag_matcher = re.compile(r"[^$]*\${([^}^{]+)}.*")


class PrettySafeLoader(yaml.SafeLoader):
    def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))


def _path_constructor(_loader: Any, node: Any):
    def replace_fn(match):
        envparts = f"{match.group(1)}:".split(":")
        return os.environ.get(envparts[0], envparts[1])

    return _var_matcher.sub(replace_fn, node.value)


def read_yaml_file(filename: str) -> dict:
    from xai_torch.utilities.optuna_yaml_loader import OptunaTag
    from yaml.parser import ParserError

    yaml.add_implicit_resolver("!envvar", _tag_matcher, None, yaml.SafeLoader)
    yaml.add_constructor("!envvar", _path_constructor, yaml.SafeLoader)

    PrettySafeLoader.add_constructor("tag:yaml.org,2002:python/tuple", PrettySafeLoader.construct_python_tuple)
    PrettySafeLoader.add_constructor("!optuna", OptunaTag.from_yaml)
    yaml.SafeDumper.add_multi_representer(OptunaTag, OptunaTag.to_yaml)
    try:
        with open(filename, "r") as f:
            return yaml.load(f.read(), Loader=PrettySafeLoader)
    except (FileNotFoundError, PermissionError, ParserError):
        return dict()
