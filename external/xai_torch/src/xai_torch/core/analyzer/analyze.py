"""
The main script that serves as the entry-point for training the models.
"""


from __future__ import annotations

from typing import TYPE_CHECKING

import hydra

from xai_torch.core.analyzer.analyzer import Analyzer

if TYPE_CHECKING:
    from omegaconf import DictConfig


@hydra.main(version_base=None, config_name="hydra")
def app(cfg: DictConfig) -> None:
    Analyzer.run(cfg)


if __name__ == "__main__":
    app()
