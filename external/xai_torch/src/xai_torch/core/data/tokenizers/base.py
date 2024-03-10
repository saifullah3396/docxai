"""
Defines the base Tokenizer class for defining custom tokenizers.
"""
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

from xai_torch.utilities.abstract_dataclass import AbstractDataclass


@dataclass
class Tokenizer(AbstractDataclass):
    name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The tokenizer name.",
        },
    )
    max_seqs_per_sample: int = field(default=5, metadata={"help": "Max number of seqeunces per sample."})

    @abstractmethod
    def __call__(self, *args, **kwargs) -> List:
        pass
