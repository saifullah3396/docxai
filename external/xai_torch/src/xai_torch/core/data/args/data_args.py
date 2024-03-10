"""
Defines the dataclass for holding data related arguments.
"""

from dataclasses import dataclass, field
from typing import Optional

from xai_torch.core.data.args.data_loader_args import DataLoaderArguments
from xai_torch.core.data.augmentations.args import DataAugmentationArguments
from xai_torch.core.data.data_cachers.args import DataCacherArguments
from xai_torch.core.data.tokenizers.args import TokenizersArguments
from xai_torch.core.data.train_val_samplers.args import TrainValSamplerArguments


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training
    and eval.
    """

    # Name of the dataset
    dataset_name: str = ""

    # Dataset directory path
    dataset_dir: str = "."

    # Dataset cacher arguments
    data_cacher_args: DataCacherArguments = field(default_factory=lambda: DataCacherArguments())

    # Train validation sampling arguments
    train_val_sampling_args: TrainValSamplerArguments = field(default_factory=lambda: TrainValSamplerArguments())

    # Arguments related to defining default data augmentations for training.
    train_aug_args: DataAugmentationArguments = field(default_factory=lambda: DataAugmentationArguments())

    # Arguments related to defining default data augmentations for evaluation.
    eval_aug_args: DataAugmentationArguments = field(default_factory=lambda: DataAugmentationArguments())

    # Arguments related to data loading or specifically torch dataloaders.
    data_loader_args: DataLoaderArguments = field(
        default_factory=lambda: DataLoaderArguments(),
    )

    # Data tokenization related arguments
    data_tokenizer_args: Optional[TokenizersArguments] = field(
        default=None,
    )

    # whether to show transformations
    show_transforms: bool = field(
        default=True,
    )

    # Whether to compute normalization params from the dataset
    use_dataset_normalization_params: bool = False

    # Any additional argument required specifically for the dataset.
    dataset_kwargs: dict = field(
        default_factory=lambda: {},
        metadata={"help": ("Any additional argument required specifically for the dataset.")},
    )
