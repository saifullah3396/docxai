"""
Defines the dataclass for holding data sampling arguments.
"""

from dataclasses import dataclass, field


@dataclass
class BatchSamplerArguments:
    # Batch sampling strategy to use
    strategy: str = ""

    # Strategy kwargs
    config: dict = field(
        default_factory=lambda: {},
    )
