"""
Defines the factory for DataAugmentation class and its children.
"""


from __future__ import annotations

from typing import TYPE_CHECKING, Type

from xai_torch.core.analyzer.tasks.base import AnalyzerTask
from xai_torch.core.args import Arguments

if TYPE_CHECKING:
    from xai_torch.core.analyzer.args import AnalyzerArguments


class AnalyzerTaskFactory:
    @staticmethod
    def get_class(analyzer_args: AnalyzerArguments) -> Type[AnalyzerTask]:
        """
        Find the model given the task and its name
        """
        from xai_torch.core.analyzer.tasks.constants import ANALYZER_TASKS_REGISTRY

        if isinstance(analyzer_args.config, list):
            tasks = []
            for task, config in zip(analyzer_args.task, analyzer_args.config):
                cls = ANALYZER_TASKS_REGISTRY.get(task, None)
                if cls is None:
                    raise ValueError(f"Analyzer task [{task}] is not supported.")
                task.append(cls)
            return tasks
        else:
            cls = ANALYZER_TASKS_REGISTRY.get(analyzer_args.task, None)
            if cls is None:
                raise ValueError(f"Analyzer task [{analyzer_args.task}] is not supported.")
            return cls

    @staticmethod
    def create(
        args: Arguments,
    ):
        from xai_torch.core.analyzer.tasks.constants import ANALYZER_TASKS_REGISTRY

        tasks = dict()
        for task_name, config in args.analyzer_args.tasks.items():
            task_type = config["task_type"]
            task_config = config["task_config"]

            cls = ANALYZER_TASKS_REGISTRY.get(task_type, None)
            print(ANALYZER_TASKS_REGISTRY)
            if cls is None:
                raise ValueError(f"Analyzer task [{task_name}:{task_type}] is not supported.")
            tasks[task_name] = cls(args, task_config)
        return tasks
