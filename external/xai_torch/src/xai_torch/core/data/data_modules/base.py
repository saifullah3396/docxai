"""
Defines the abstract base class for handling the functionality for DataModules.
"""

from __future__ import annotations

import logging
import sys
from abc import ABC, abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING, Callable, Optional, Tuple, Type

from xai_torch.core.data.utilities.typing import CollateFnDict, TransformsDict
from xai_torch.core.factory.constants import DATAMODULES_REGISTRY
from xai_torch.core.training.constants import TrainingStage
from xai_torch.utilities.logging_utils import DEFAULT_LOGGER_NAME
from torch.utils.data import BatchSampler, DataLoader, Dataset, Subset

if TYPE_CHECKING:
    from torch.utils.data import BatchSampler, DataLoader, Dataset, Subset

    from xai_torch.core.data.tokenizers.base import Tokenizer
    from xai_torch.core.data.train_val_samplers.base import TrainValSampler


class BaseDataModule(ABC):
    """
    The base data module class for loading and transforming data.

    Args:
        collate_fns: Collate functions to use for train/val/test.
        transforms: Data transforms to use for train/val/test. This is overrided by
            aug.data_aug.train_augs/aug.data_aug.eval_augs if they are set.
        n_partitions: If required, the number of data partitions to create
        partition_id: The id of current partition if partitions are generated.
    """

    def __init__(
        self,
        dataset_dir: str,
        collate_fns: CollateFnDict = CollateFnDict(),
        transforms: TransformsDict = TransformsDict(),
        data_cacher_wrapper: Optional[Callable] = None,
        tokenizer: Optional[Tokenizer] = None,
        train_val_sampler: Optional[TrainValSampler] = None,
        show_transforms: bool = False,
        n_partitions: Optional[int] = None,
        partition_id: Optional[int] = None,
    ):
        # initialize base classes
        super().__init__()

        # initialize the arguments
        self._dataset_dir = dataset_dir
        self._collate_fns = collate_fns
        self._transforms = transforms
        self._data_cacher_wrapper = data_cacher_wrapper
        self._tokenizer = tokenizer
        self._train_val_sampler = train_val_sampler
        self._show_transforms = show_transforms

        # setup partition ids if required
        self._n_partitions = n_partitions
        self._partition_id = partition_id

        # setup logger
        self._logger = logging.getLogger(DEFAULT_LOGGER_NAME)

    @cached_property
    @abstractmethod
    def dataset_class(self) -> Type[Dataset]:
        """Returns the underlying dataset class. Must be implemented in child."""

    @cached_property
    def dataset_name(self) -> Type[Dataset]:
        """Returns the dataset name."""
        return next(key for key, value in DATAMODULES_REGISTRY.items() if isinstance(self, value))

    def show_batch(self, stage: TrainingStage, nmax: int = 16):
        pass

    @property
    def gt_metadata(self):
        """Returns the metadata related to ground truth such as class labels."""
        return {}

    @property
    def val_dataloader_available(self) -> bool:
        """Returns whether val_dataloader is available."""
        return self._train_val_sampler is not None or "val" in self.dataset_class.get_supported_splits()

    @property
    def test_dataloader_available(self) -> bool:
        """Returns whether test_dataloader is available."""
        return "test" in self.dataset_class.get_supported_splits()

    @property
    def is_data_sampler_custom(self) -> bool:
        # we always keep it custom since DPP often messes up in lightning
        return True

    def _load_dataset(
        self,
        split: str = "train",
        transforms=None,
        indices: list = [],
        quiet=False,
        **dataset_kwargs,
    ) -> Dataset:
        """
        Loads the dataset based on its name and performs validation of some arguments.

        Args:
            split: Train, val or test split to load
            data_transforms: Data transformations to be used on the dataset
            indices: Indices to use for training/validation
                subsets if required.
        """
        try:
            # check if dataset_class property is correctly defined
            if self.dataset_class is None:
                raise ValueError("Dataset class [{self.dataset_class}] not found.")

            # initialize the underlying dataset class
            dataset = self.dataset_class(
                self.dataset_name,
                self._dataset_dir,
                split=split,
                transforms=transforms,
                data_cacher_wrapper=self._data_cacher_wrapper,
                tokenizer=self._tokenizer,
                indices=indices,
                quiet=quiet,
                show_transforms=self._show_transforms,
                **dataset_kwargs,
            )

            # load the dataset
            dataset.load()

            return dataset
        except Exception as exc:
            self._logger.exception(f"Exception raised while loading the dataset " f"[{self.dataset_name}]: {exc}")
            sys.exit(1)

    def _load_train_val_datasets(self, quiet=False) -> Optional[Tuple[Dataset, Dataset]]:
        """
        Loads the train and validation datasets.
        """

        # here we load the dataset itself.
        train_dataset = self._load_dataset(
            split="train",
            transforms=self._transforms.train,
            quiet=quiet,
        )

        if "val" in self.dataset_class.get_supported_splits():
            val_dataset = self._load_dataset(
                split="val",
                transforms=self._transforms.val,
                quiet=quiet,
            )
            return train_dataset, val_dataset
        elif self._train_val_sampler is not None:
            import copy

            train_subset, val_subset = next(self._train_val_sampler(train_dataset))

            # generate new train dataset based on indices
            train_dataset.indices = train_subset.indices

            # generate val dataset from train dataset
            val_dataset = copy.deepcopy(train_dataset)
            val_dataset.indices = val_subset.indices
            val_dataset.transforms = self._transforms.val
            return train_dataset, val_dataset
        else:
            return train_dataset, None

    def _load_test_dataset(self, quiet=False) -> Optional[Dataset]:
        """
        Loads the test dataset.
        """

        return self._load_dataset(
            split="test",
            transforms=self._transforms.test,
            quiet=quiet,
        )

    def setup(
        self,
        quiet=False,
        stage: TrainingStage = TrainingStage.train,
        do_train: bool = True,
        max_train_samples: Optional[int] = None,
        max_val_samples: Optional[int] = None,
        max_test_samples: Optional[int] = None,
        use_test_set_for_val: bool = False,
    ) -> None:
        """
        Loads the data on on every device for the given stage.

        Args:
            stage: Training stage for which to load the data.
        """
        from torch.utils.data import Subset

        if not quiet:
            if stage is not None:
                self._logger.info(f"Loading data for stage == {stage}")
            else:
                self._logger.info(f"Loading data for stage == train|test|val")

        # Assign train/val datasets for use in dataloaders using the train/val sampler
        # lightning calls training stage 'fit'
        if do_train and (stage == TrainingStage.train or stage is None):
            self.train_dataset, self.val_dataset = self._load_train_val_datasets(quiet=quiet)

            # if max_train_samples is set get the given number of examples
            # from the dataset
            if max_train_samples is not None:
                self.train_dataset = Subset(
                    self.train_dataset,
                    range(0, max_train_samples),
                )

            # if max_val_samples is set get the given number of examples
            # from the dataset
            if max_val_samples is not None:
                self.val_dataset = Subset(
                    self.val_dataset,
                    range(0, max_val_samples),
                )

            # create partitions over original dataset if required
            if self._n_partitions is not None and self._partition_id is not None:
                assert self._partition_id < self._n_partitions
                self.train_dataset = self._create_dataset_partitions(self.train_dataset)
                if self.val_dataset is not None:
                    self.val_dataset = self._create_dataset_partitions(self.val_dataset)

            if use_test_set_for_val:
                import logging

                logger = logging.getLogger(DEFAULT_LOGGER_NAME)
                logger.warning(
                    "Using test set as validation set."
                    " If this behavior is not required set, use_test_set_for_val=False in config."
                )

                self.val_dataset = self._load_test_dataset(quiet=quiet)

                if self.val_dataset is not None:
                    if max_val_samples is not None:
                        self.val_dataset = Subset(
                            self.val_dataset,
                            range(0, max_val_samples),
                        )

                    # create partitions over original dataset if required
                    if self._n_partitions is not None and self._partition_id is not None:
                        assert self._partition_id < self._n_partitions
                        self.val_dataset = self._create_dataset_partitions(self.val_dataset)

                    if not quiet:
                        if self.val_dataset is not None:
                            self._logger.info(f"Test set size = {len(self.val_dataset)}")
            if not quiet:
                self._logger.info(f"Training set size = {len(self.train_dataset)}")
                if self.val_dataset is not None:
                    self._logger.info(f"Validation set size = {len(self.val_dataset)}")

        # Assign test dataset for use in dataloader(s)
        if stage == TrainingStage.test or stage is None:
            self.test_dataset = self._load_test_dataset(quiet=quiet)

            if self.test_dataset is not None:
                # if max_test_samples is set get the given number of examples
                # from the dataset
                if max_test_samples is not None:
                    self.test_dataset = Subset(
                        self.test_dataset,
                        range(0, max_test_samples),
                    )

                # create partitions over original dataset if required
                if self._n_partitions is not None and self._partition_id is not None:
                    import os

                    assert self._partition_id < self._n_partitions
                    self.test_dataset = self._create_dataset_partitions(
                        self.test_dataset, seed=int(os.environ.get("DEFAULT_SEED"))
                    )

                if not quiet:
                    if self.test_dataset is not None:
                        self._logger.info(f"Test set size = {len(self.test_dataset)}")

    def _create_dataset_partitions(self, dataset: Dataset, seed: int) -> Subset:
        """
        Creates partitioned subset for the given dataset.
        """

        from xai_torch.core.data.data_modules.utilities import create_dataset_partitions

        return create_dataset_partitions(
            dataset,
            self._n_partitions,
            self._partition_id,
            seed,
        )

    def setup_train_dataloader(
        self,
        dataset: Dataset,
        per_device_train_batch_size: int,
        dataloader_num_workers: int = 4,
        pin_memory: bool = True,
        shuffle_data: bool = True,
        dataloader_drop_last: bool = True,
        batch_sampler_wrapper: Optional[BatchSampler] = None,
    ):
        """
        Defines the torch dataloader for train dataset.
        """

        import ignite.distributed as idist
        from torch.utils.data import RandomSampler, SequentialSampler

        # setup sampler
        if shuffle_data:
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)

        # setup custom batch sampler
        batch_sampler = batch_sampler_wrapper(sampler) if batch_sampler_wrapper is not None else None
        if batch_sampler is None:
            return idist.auto_dataloader(
                dataset,
                sampler=sampler,
                batch_size=per_device_train_batch_size * idist.get_world_size(),
                collate_fn=self._collate_fns.train,
                num_workers=dataloader_num_workers,
                pin_memory=pin_memory,
                drop_last=True if idist.get_world_size() > 1 else dataloader_drop_last,
            )
        else:

            return idist.auto_dataloader(
                dataset,
                batch_sampler=batch_sampler,
                collate_fn=self._collate_fns.train,
                num_workers=dataloader_num_workers,
                pin_memory=pin_memory,
                drop_last=dataloader_drop_last,
            )

    def train_dataloader(
        self,
        per_device_train_batch_size: int,
        dataloader_num_workers: int = 4,
        pin_memory: bool = True,
        shuffle_data: bool = True,
        dataloader_drop_last: bool = True,
        batch_sampler_wrapper: Optional[BatchSampler] = None,
    ) -> DataLoader:
        return self.setup_train_dataloader(
            self.train_dataset,
            per_device_train_batch_size=per_device_train_batch_size,
            dataloader_num_workers=dataloader_num_workers,
            pin_memory=pin_memory,
            shuffle_data=shuffle_data,
            dataloader_drop_last=dataloader_drop_last,
            batch_sampler_wrapper=batch_sampler_wrapper,
        )

    def val_dataloader(
        self,
        per_device_eval_batch_size: int,
        dataloader_num_workers: int = 4,
        pin_memory: bool = True,
    ) -> DataLoader:
        """
        Defines the torch dataloader for validation dataset.
        """

        import ignite.distributed as idist
        from torch.utils.data import SequentialSampler

        if idist.get_world_size() > 1:
            if len(self.val_dataset) % idist.get_world_size() != 0:
                self._logger.warning(
                    "Enabling distributed evaluation with an eval dataset not divisible by process number. "
                    "This will slightly alter validation results as extra duplicate entries are added to achieve "
                    "equal num of samples per-process."
                )

            # let ignite handle distributed sampler
            sampler = None
        else:
            sampler = SequentialSampler(self.val_dataset)

        return idist.auto_dataloader(
            self.val_dataset,
            sampler=sampler,
            batch_size=per_device_eval_batch_size * idist.get_world_size(),
            collate_fn=self._collate_fns.val,
            num_workers=dataloader_num_workers,
            pin_memory=pin_memory,
            drop_last=False,  # drop last is always false for validation
        )

    def setup_test_dataloader(
        self,
        dataset,
        per_device_eval_batch_size: int,
        dataloader_num_workers: int = 4,
        pin_memory: bool = True,
    ) -> DataLoader:
        """
        Defines the torch dataloader for test dataset.
        """
        from torch.utils.data import DataLoader, SequentialSampler

        sampler = SequentialSampler(dataset)
        return DataLoader(
            dataset,
            sampler=sampler,
            batch_size=per_device_eval_batch_size,
            collate_fn=self._collate_fns.test,
            num_workers=dataloader_num_workers,
            pin_memory=pin_memory,
            shuffle=False,  # data is always sequential for test
            drop_last=False,  # drop last is always false for test
        )

    def test_dataloader(
        self,
        per_device_eval_batch_size: int,
        dataloader_num_workers: int = 4,
        pin_memory: bool = True,
    ) -> DataLoader:
        return self.setup_test_dataloader(
            self.test_dataset,
            per_device_eval_batch_size=per_device_eval_batch_size,
            dataloader_num_workers=dataloader_num_workers,
            pin_memory=pin_memory,
        )

    def test_dataloader_indices(
        self,
        start_idx: int,
        end_idx: int,
        per_device_eval_batch_size: int,
        dataloader_num_workers: int = 4,
        pin_memory: bool = True,
    ) -> DataLoader:
        import os
        import numpy as np

        # only works with envs
        subset = Subset(
            self.test_dataset,
            range(start_idx,end_idx),
        )
        print('subset' , len(subset), range(start_idx,end_idx))
        return self.setup_test_dataloader(
            subset,
            per_device_eval_batch_size=per_device_eval_batch_size,
            dataloader_num_workers=dataloader_num_workers,
            pin_memory=pin_memory,
        )

    def get_dataloader(self, stage: TrainingStage):
        if stage == TrainingStage.train:
            return self.train_dataloader()
        elif stage == TrainingStage.test:
            return self.test_dataloader()
        elif stage == TrainingStage.val:
            return self.val_dataloader()
