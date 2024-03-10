"""
Defines the dataclass for holding data loader arguments.
"""

from dataclasses import dataclass, field
from typing import Optional

from xai_torch.core.data.data_samplers.args import BatchSamplerArguments


@dataclass
class DataLoaderArguments:
    # Training batch size
    per_device_train_batch_size: int = 64

    # Evaluation batch size
    per_device_eval_batch_size: int = 64

    # Whether to drop last batch in data
    dataloader_drop_last: bool = False

    # Whether to shuffle the data
    shuffle_data: bool = True

    # Whether to pin memory for data loading
    pin_memory: bool = True

    # Dataloader number of workers
    dataloader_num_workers: int = 4

    # Maximum training samples to use
    max_train_samples: Optional[int] = None

    # Maximum val samples to use
    max_val_samples: Optional[int] = None

    # Maximum test samples to use
    max_test_samples: Optional[int] = None

    # The batch sampler arguments for using custom batch samplers for training
    train_batch_sampler_args: BatchSamplerArguments = field(default_factory=lambda: BatchSamplerArguments())

    # The batch sampler arguments for using custom batch samplers for evaluation
    eval_batch_sampler_args: BatchSamplerArguments = field(default_factory=lambda: BatchSamplerArguments())

    # Whether to replace test set for validation set
    use_test_set_for_val: bool = False
