"""
The main script that serves as the entry-point for evaluating the models.
"""


from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import hydra

if TYPE_CHECKING:
    from omegaconf import DictConfig


def run(cfg: DictConfig):
    from omegaconf import OmegaConf

    from xai_torch.core.args import Arguments
    from xai_torch.utilities.dacite_wrapper import from_dict

    # initialize general configuration for script
    cfg = OmegaConf.to_object(cfg)
    args = from_dict(data_class=Arguments, data=cfg["args"])
    if args.general_args.do_test:
        try:
            from xai_torch.core.training.trainer import Trainer

            Trainer(args).test(0)
        except Exception as e:
            logging.exception(e)


@hydra.main(version_base=None, config_name="hydra")
def app(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    app()
