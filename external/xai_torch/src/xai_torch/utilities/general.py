"""
Defines general-purpose utility functions/classes.
"""
from __future__ import annotations

import re
import textwrap
from argparse import ArgumentTypeError
from pathlib import Path
from typing import Generator, Union


def empty_cuda_cache(_) -> None:
    import torch

    torch.cuda.empty_cache()
    import gc

    gc.collect()


def reset_random_seeds(seed):
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_dist_avail_and_initialized():
    import torch.distributed as dist

    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


# From https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def string_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError(
            f"Truthy value expected: got {v} but expected one of yes/no, true/false, t/f, y/n, 1/0 (case insensitive)."
        )


def indent_string(s: str, ind):
    return textwrap.indent(s, ind)


def str_to_underscored_lower(s: str):
    return "_".join(l.lower() for l in re.findall("[A-Z][^A-Z]*", s))


def check_max_len(x: list, max_len: int = 2):
    if len(x) != 2:
        raise ValueError("List should of size [{max_len}]")


def make_dir(path: Union[str, Path]):
    if not path.exists():
        path.mkdir(parents=True)


def concatenate_list_dict_to_dict(list_dict):
    import torch

    output = {}
    for d in list_dict:
        for k, v in d.items():
            if k not in output:
                output[k] = []
            output[k].append(v)
    output = {k: torch.cat(v) if len(v[0].shape) > 0 else torch.tensor(v) for k, v in output.items()}
    return output


def generate_default_config(output: str):
    import dataclasses

    import yaml

    from xai_torch.core.args import Arguments

    args = Arguments()
    d = dataclasses.asdict(args)
    with open(output, "w") as f:
        yaml.dump(d, f)


def get_matplotlib_grid(rows, cols, figsize=16):
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    # plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.serif": ["Times"], "font.size": 16})
    ratio = cols / rows
    figsize = (figsize * ratio, figsize)
    # fig, axs = plt.subplots(len(images), len(attribution_maps.keys()) + 1, figsize=figsize)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(rows, cols)
    gs.update(
        wspace=0.0,
        hspace=0.0,
        top=1.0 - 0.5 / (rows + 1),
        bottom=0.5 / (rows + 1),
        left=0.5 / (cols + 1),
        right=1 - 0.5 / (cols + 1),
    )
    return fig, gs


def drange(
    min_val: Union[int, float], max_val: Union[int, float], step_val: Union[int, float]
) -> Generator[Union[int, float], None, None]:
    curr = min_val
    while curr < max_val:
        yield curr
        curr += step_val
