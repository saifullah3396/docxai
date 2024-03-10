def register_augmentation(reg_name: str = ""):
    from xai_torch.core.data.augmentations.base import DataAugmentation
    from xai_torch.core.factory.constants import DATA_AUGMENTATIONS_REGISTRY
    from xai_torch.utilities.decorators import register_as_child

    return register_as_child(
        base_class_type=DataAugmentation,
        registry=DATA_AUGMENTATIONS_REGISTRY,
        reg_name=reg_name,
    )


def register_datacacher(reg_name: str = ""):
    from xai_torch.core.data.data_cachers.base import DataCacherBase
    from xai_torch.core.factory.constants import DATA_CACHERS_REGISTRY
    from xai_torch.utilities.decorators import register_as_child

    return register_as_child(
        base_class_type=DataCacherBase,
        registry=DATA_CACHERS_REGISTRY,
        reg_name=reg_name,
    )


def register_datamodule(reg_name: str = ""):
    from xai_torch.core.data.data_modules.base import BaseDataModule
    from xai_torch.core.factory.constants import DATAMODULES_REGISTRY
    from xai_torch.utilities.decorators import register_as_child

    return register_as_child(base_class_type=BaseDataModule, registry=DATAMODULES_REGISTRY, reg_name=reg_name)


def register_tokenizer(reg_name: str = ""):
    from xai_torch.core.data.tokenizers.base import Tokenizer
    from xai_torch.core.factory.constants import TOKENIZERS_REGISTRY
    from xai_torch.utilities.decorators import register_as_child

    return register_as_child(base_class_type=Tokenizer, registry=TOKENIZERS_REGISTRY, reg_name=reg_name)


def register_hf_tokenizer(reg_name: str = ""):
    from transformers import PreTrainedTokenizerBase

    from xai_torch.core.factory.constants import HF_TOKENIZERS_REGISTRY
    from xai_torch.utilities.decorators import register_as_child

    return register_as_child(
        base_class_type=PreTrainedTokenizerBase,
        registry=HF_TOKENIZERS_REGISTRY,
        reg_name=reg_name,
    )


def register_batch_sampler(reg_name: str = ""):
    from torch.utils.data.sampler import BatchSampler

    from xai_torch.core.factory.constants import BATCH_SAMPLERS_REGISTRY
    from xai_torch.utilities.decorators import register_as_child

    return register_as_child(
        base_class_type=BatchSampler,
        registry=BATCH_SAMPLERS_REGISTRY,
        reg_name=reg_name,
    )


def register_train_val_sampler(reg_name: str = ""):
    from xai_torch.core.data.train_val_samplers.base import TrainValSampler
    from xai_torch.core.factory.constants import TRAIN_VAL_SAMPLERS_REGISTRY
    from xai_torch.utilities.decorators import register_as_child

    return register_as_child(
        base_class_type=TrainValSampler,
        registry=TRAIN_VAL_SAMPLERS_REGISTRY,
        reg_name=reg_name,
    )
