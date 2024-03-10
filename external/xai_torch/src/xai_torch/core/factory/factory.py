from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Union

from xai_torch.utilities.logging_utils import DEFAULT_LOGGER_NAME

if TYPE_CHECKING:
    from xai_torch.core.data.data_modules.base import BaseDataModule


class DataAugmentationFactory:
    @staticmethod
    def create(
        strategy: str,
        config: Union[List, List[dict]],
        keys: List[str] = None,
    ):
        from xai_torch.core.data.augmentations.core_transforms import DictTransform
        from xai_torch.core.factory.constants import DATA_AUGMENTATIONS_REGISTRY
        from xai_torch.utilities.dacite_wrapper import from_dict

        aug_class = DATA_AUGMENTATIONS_REGISTRY.get(strategy, None)
        if aug_class is None:
            raise ValueError(f"DataAugmentation [{strategy}] is not supported.")
        if keys is not None:
            if isinstance(config, dict):
                from xai_torch.utilities.dacite_wrapper import from_dict

                return DictTransform(
                    keys=keys,
                    transform=from_dict(
                        data_class=aug_class,
                        data=config,
                    ),
                )
            elif isinstance(config, list):
                print('config', keys)
                return [
                    DictTransform(
                        keys=keys[idx],
                        transform=from_dict(
                            data_class=aug_class,
                            data=c,
                        ),
                    )
                    for idx, c in enumerate(config)
                ]
        else:
            if isinstance(config, dict):
                return from_dict(
                    data_class=aug_class,
                    data=config,
                )
            elif isinstance(config, list):
                return [
                    from_dict(
                        data_class=aug_class,
                        data=c,
                    )
                    for c in config
                ]


def data_cacher_wrapper(data_cacher_class, **kwargs):

    from xai_torch.utilities.dacite_wrapper import from_dict

    def wrap(dataset):
        if dataset is not None:
            kwargs["dataset"] = dataset
        return from_dict(
            data_class=data_cacher_class,
            data=kwargs,
        )
        # import inspect

        # logger = logging.getLogger(DEFAULT_LOGGER_NAME)

        # try:
        #     return data_cacher_class(dataset, **kwargs)
        # except TypeError as exc:
        #     # get arguments
        #     args = inspect.signature(data_cacher_class.__init__).parameters  # remove self argument

        #     msg = f"Exception raised while initializing data_cacher_class [{data_cacher_class}]: {exc}. "
        #     msg += f"Valid arguments are:\n"
        #     for k, value in args.items():
        #         if k == "self":
        #             continue
        #         msg += f"{value}\n"
        #     logger.exception(msg)
        #     exit()

    return wrap


class DataCacherFactory:
    @staticmethod
    def create(strategy: str, **kwargs):
        from xai_torch.core.factory.constants import DATA_CACHERS_REGISTRY

        data_cacher_class = DATA_CACHERS_REGISTRY.get(strategy, None)
        if data_cacher_class is None:
            raise ValueError(f"DataCacher [{strategy}] is not supported.")
        return data_cacher_wrapper(data_cacher_class, **kwargs)


class DataModuleFactory:
    @staticmethod
    def create(
        dataset_name: str,
        dataset_dir: str,
        **kwargs,
    ) -> BaseDataModule:
        from xai_torch.core.factory.constants import DATAMODULES_REGISTRY

        datamodule_class = DATAMODULES_REGISTRY.get(dataset_name, None)
        if datamodule_class is None:
            raise ValueError(
                f"Datamodule [{dataset_name}] is not supported. " f"Possible choices are:\n{DATAMODULES_REGISTRY}"
            )
        return datamodule_class(dataset_dir, **kwargs)

    @staticmethod
    def create_partitioned_modules(
        dataset_name: str,
        n_partitions: int,
        initialize: bool = True,
        **kwargs,
    ) -> BaseDataModule:

        from xai_torch.core.factory.constants import DATAMODULES_REGISTRY

        logger = logging.getLogger(DEFAULT_LOGGER_NAME)
        datamodule_class = DATAMODULES_REGISTRY.get(dataset_name, None)
        if datamodule_class is None:
            raise ValueError(
                f"Datamodule [{dataset_name}] is not supported. " f"Possible choices are:\n{DATAMODULES_REGISTRY}"
            )

        datamodules = []
        for idx in range(n_partitions):
            logger.info(f"Creating datamodule partition [{idx}].")
            datamodules.append(
                datamodule_class(
                    n_partitions=n_partitions,
                    partition_id=idx,
                    **kwargs,
                )
            )
            if initialize:
                datamodules[-1].setup()
        return datamodules


def batch_sampler_wrapper(batch_sampler_class, **kwargs):
    def wrap(sampler):
        return batch_sampler_class(sampler=sampler, **kwargs)

    return wrap


class BatchSamplerFactory:
    @staticmethod
    def create(strategy: str, **kwargs):
        from xai_torch.core.factory.constants import BATCH_SAMPLERS_REGISTRY

        if strategy == "":
            return

        batch_sampler_class = BATCH_SAMPLERS_REGISTRY.get(strategy, None)
        if batch_sampler_class is None:
            raise ValueError(f"BatchSampler [{strategy}] is not supported.")
        return batch_sampler_wrapper(batch_sampler_class=batch_sampler_class, **kwargs)


class TokenizerFactory:
    initialized = {}

    @staticmethod
    def create(strategy: str, **kwargs):
        from xai_torch.core.factory.constants import TOKENIZERS_REGISTRY
        from xai_torch.utilities.dacite_wrapper import from_dict

        tokenizer_class = TOKENIZERS_REGISTRY.get(strategy, None)

        # return the tokenizer if it is already initialized before
        if strategy in TokenizerFactory.initialized:
            return TokenizerFactory.initialized[strategy]

        # otherwise, find and initialize a new one
        if tokenizer_class is None:
            raise ValueError(f"Tokenizer [{strategy}] is not supported.")
        tokenizer = from_dict(
            data_class=tokenizer_class,
            data=kwargs,
        )

        # tokenizer has to be initialized only once so that it can be reused across
        # the code, so we save it here
        TokenizerFactory.initialized[strategy] = tokenizer

        # return the tokenizer
        return tokenizer


class TrainValSamplerFactory:
    @staticmethod
    def create(strategy: str, **kwargs):
        from xai_torch.core.factory.constants import TRAIN_VAL_SAMPLERS_REGISTRY
        from xai_torch.utilities.dacite_wrapper import from_dict

        if strategy == "":
            return

        sampler_class = TRAIN_VAL_SAMPLERS_REGISTRY.get(strategy, None)
        if sampler_class is None:
            raise ValueError(f"TrainValSampler [{strategy}] is not supported.")
        return from_dict(
            data_class=sampler_class,
            data=kwargs,
        )
