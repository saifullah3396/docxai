import logging

import h5py
import numpy as np
import torch
from xai_torch.core.constants import DataKeys
from xai_torch.core.data.args.data_args import DataArguments
from xai_torch.core.data.data_modules.base import BaseDataModule
from xai_torch.core.factory.factory import BatchSamplerFactory
from xai_torch.core.training.constants import TrainingStage
from xai_torch.utilities.logging_utils import DEFAULT_LOGGER_NAME

def generate_image_baselines(
    datamodule: BaseDataModule, data_args: DataArguments, target_baselines_per_label: int = 10
):
    baselines_per_label = {}
    num_labels = datamodule.num_labels
    for label in range(num_labels):
        baselines_per_label[label] = []

    # setup batch sampler if needed
    batch_sampler_wrapper = BatchSamplerFactory.create(
        data_args.data_loader_args.train_batch_sampler_args.strategy,
        **data_args.data_loader_args.train_batch_sampler_args.config,
    )

    # setup dataloaders
    train_dataloader = datamodule.train_dataloader(
        data_args.data_loader_args.per_device_train_batch_size,
        dataloader_num_workers=data_args.data_loader_args.dataloader_num_workers,
        pin_memory=data_args.data_loader_args.pin_memory,
        shuffle_data=data_args.data_loader_args.shuffle_data,
        dataloader_drop_last=data_args.data_loader_args.dataloader_drop_last,
        batch_sampler_wrapper=batch_sampler_wrapper,
    )

    for batch in train_dataloader:
        for image, label in zip(batch[DataKeys.IMAGE], batch[DataKeys.LABEL]):
            label = label.item()
            if len(baselines_per_label[label]) < target_baselines_per_label:
                baselines_per_label[label].append(image)
        finish = np.array(
            [len(baselines) == target_baselines_per_label for baselines in baselines_per_label.values()]
        ).all()
        if finish:
            break

    baselines = list(set().union(*baselines_per_label.values()))
    baselines = torch.stack(baselines)
    return baselines


def fix_model_inplace(model):
    # remove relu inplace operations as this causes problems in some XAI methods
    for module in model.torch_model.modules():
        if isinstance(module, torch.nn.ReLU):
            module.inplace = False


def wrap_model_output(torch_model_unwrapped: torch.nn.Module):
    class Wrapped(torch.nn.Module):
        def __init__(self, model) -> None:
            super().__init__()
            self._model = model

        def predict_step(self, batch) -> None:
            return self(**batch, stage=TrainingStage.predict).argmax(-1)

        def forward(self, *args, **kwargs):
            return self._model(*args, **kwargs)[DataKeys.LOGITS]

        def __getattr__(self, name):
            """Forward missing attributes to twice-wrapped module."""
            try:
                # defer to nn.Module's logic
                return super().__getattr__(name)
            except AttributeError:
                # forward to the twice-wrapped module
                return getattr(self._model, name)

    return Wrapped(torch_model_unwrapped)


def setup_msgpack_output_file(output_file_path):
    from datadings.reader import MsgpackReader

    reader = None
    if output_file_path.exists():
        reader = MsgpackReader(output_file_path)
        if output_file_path.with_suffix(".msgpack.new").exists():
            output_file_path.with_suffix(".msgpack.new").unlink()

        output_file_path = output_file_path.with_suffix(".msgpack.new")

    return output_file_path, reader


def load_key_from_file(reader, key, indices):
    if indices is None:
        return None
    results = [reader[idx][key] if key in reader[idx].keys() else None for idx in indices]
    if any(x is None for x in results):
        return None
    else:
        return results


def update_dataset_at_indices(
    hf: h5py.File, key: str, indices: np.array, data, maxshape=(None,), overwrite: bool = False
):
    if key not in hf:
        hf.create_dataset(key, data=data, compression="gzip", chunks=True, maxshape=maxshape)
    else:
        if maxshape[1:] != hf[key].shape[1:]:
            logger = logging.getLogger(DEFAULT_LOGGER_NAME)
            if overwrite:
                logger.info(f"Reinitializing data due to shape mismatch for key={key} since overwrite is set to True.")
                del hf[key]
                hf.create_dataset(key, data=data, compression="gzip", chunks=True, maxshape=maxshape)
            else:
                logger.error(f"Data overwrite is set to False but there is mismatch between data shapes for key = {key}")
                exit()

        max_len = indices.max() + 1
        if len(hf[key]) < max_len:
            hf[key].resize((indices.max() + 1), axis=0)
            hf[key][indices] = data
        elif overwrite:
            hf[key][indices] = data
