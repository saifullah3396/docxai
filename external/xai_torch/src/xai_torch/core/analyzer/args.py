"""
Defines the dataclass for holding analyzer arguments.
"""

import os
from dataclasses import dataclass, field
from typing import List, Mapping, Union

from xai_torch.core.args_base import ArgumentsBase


@dataclass
class AnalyzerArguments(ArgumentsBase):
    """
    Dataclass that holds the analyzer related arguments.
    """

    analyzer_output_dir: str = (f"{os.environ.get('XAI_TORCH_OUTPUT_DIR', '.')}/analyzer",)
    model_checkpoints: Union[List[List[str]], List[str]] = ""
    tasks: Union[Mapping[str, dict], List[Mapping[str, dict]]] = field(
        default=None,
    )

    def __post_init__(self):
        from xai_torch.core.analyzer.tasks.constants import ANALYZER_TASKS_REGISTRY

        if self.tasks is not None:
            for task_name, config in self.tasks.items():
                if "task_type" not in config.keys():
                    raise ValueError(
                        f"Analyzer config task item [{task_name}] must have 'task_type' argument "
                        f"from the following choices: {[e for e in ANALYZER_TASKS_REGISTRY.keys()]}"
                    )
                if "task_config" not in config.keys():
                    raise ValueError(
                        f"Analyzer config task item [{task_name}] must have 'task_config' argument with task config dict."
                    )
