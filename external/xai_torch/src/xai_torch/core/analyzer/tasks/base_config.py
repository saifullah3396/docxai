"""
Defines the base TrainValSampler class for defining training/validation split samplers.
"""

from dataclasses import dataclass

from xai_torch.utilities.abstract_dataclass import AbstractDataclass


@dataclass
class AnalyzerTaskConfig(AbstractDataclass):
    """
    Base task configuration.
    """
