from xai_torch.core.analyzer.tasks.base import AnalyzerTask
from xai_torch.core.analyzer.tasks.constants import ANALYZER_TASKS_REGISTRY
from xai_torch.utilities.decorators import register_as_child


def register_analyzer_task(reg_name: str = ""):
    return register_as_child(
        base_class_type=AnalyzerTask,
        registry=ANALYZER_TASKS_REGISTRY,
        reg_name=reg_name,
    )
