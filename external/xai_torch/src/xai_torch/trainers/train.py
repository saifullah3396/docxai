"""
The main script that serves as the entry-point for training the models.
"""


from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import hydra
from xai_torch.core.training.trainer import Trainer

if TYPE_CHECKING:
    from omegaconf import DictConfig


@hydra.main(version_base=None, config_name="hydra")
def app(cfg: DictConfig) -> None:
    Trainer.run(cfg)


if __name__ == "__main__":
    app()
