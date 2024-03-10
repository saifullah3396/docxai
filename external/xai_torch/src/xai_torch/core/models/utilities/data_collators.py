from dataclasses import dataclass, field

import torch

from xai_torch.core.constants import DataKeys
from xai_torch.core.data.args.data_args import DataArguments
from xai_torch.core.factory.factory import TokenizerFactory
from xai_torch.core.models.utilities.general import pad_sequences
from xai_torch.core.training.args.training import TrainingArguments


@dataclass
class PassThroughCollator:
    """
    Data collator for converting data in the batch to a dictionary of pytorch tensors.
    """

    def __call__(self, features):
        batch = {}
        for k in features[0].keys():
            batch[k] = [sample[k] for sample in features]
        return batch


@dataclass
class BatchToTensorDataCollator:
    """
    Data collator for converting data in the batch to a dictionary of pytorch tensors.
    """

    data_key_type_map: dict = field(default_factory=lambda: {})

    def __call__(self, features):
        batch = {}
        for k, dtype in self.data_key_type_map.items():
            if isinstance(features[0][k], torch.Tensor):
                batch[k] = torch.stack([sample[k] for sample in features]).type(dtype)
            elif isinstance(features[0][k], list):
                batch[k] = torch.tensor([sample[k] for sample in features], dtype=dtype)
            elif isinstance(features[0][k], str):
                batch[k] = [sample[k] for sample in features]
            else:
                batch[k] = torch.tensor([sample[k] for sample in features], dtype=dtype)
        return batch


@dataclass
class SequenceDataCollator:
    data_args: DataArguments
    training_args: TrainingArguments
    data_key_type_map: dict = field(default_factory=lambda: {})
    data_padding_dict: dict = field(default_factory=lambda: {})
    expand_batch: bool = True

    def __post_init__(self) -> None:
        # get the tokenizer
        self.tokenizer = TokenizerFactory.create(self.data_args.data_tokenizer_args)

        # get token padding configuration
        self.padding = "max_length" if self.data_args.data_tokenization_args.pad_to_max_length else False

        # initialize padding skips
        if self.training_args is not None:
            self.pad_to_multiple_of = 8 if self.training_args.with_amp == 16 else None
        else:
            self.pad_to_multiple_of = None

        # sequence keys dict
        if DataKeys.TOKEN_IDS not in self.data_padding_dict:
            self.data_padding_dict[DataKeys.TOKEN_IDS] = self.tokenizer.pad_token_id
        if DataKeys.TOKEN_TYPE_IDS not in self.data_padding_dict:
            self.data_padding_dict[DataKeys.TOKEN_TYPE_IDS] = self.tokenizer.pad_token_type_id
        if DataKeys.ATTENTION_MASKS not in self.data_padding_dict:
            self.data_padding_dict[DataKeys.ATTENTION_MASKS] = 0
        if DataKeys.TOKEN_BBOXES not in self.data_padding_dict:
            self.data_padding_dict[DataKeys.TOKEN_BBOXES] = [0, 0, 0, 0]
        if DataKeys.TOKEN_ANGLES not in self.data_padding_dict:
            self.data_padding_dict[DataKeys.TOKEN_ANGLES] = 0

    def __call__(self, features):
        batch = {}
        for k in features[0].keys():
            if k not in self.data_key_type_map.keys():
                continue
            if k not in batch:
                batch[k] = []
            for f in features:
                batch[k].append(f[k])

        # pad sequences
        for k, padding_elem in self.data_padding_dict.items():
            if k in batch:
                batch[k] = pad_sequences(
                    batch[k],
                    self.tokenizer.padding_side,
                    self.data_args.data_tokenization_args.seq_max_length,
                    padding_elem,
                )

        # convert all objects in batch to torch tensors
        for (k, v) in batch.items():
            if isinstance(v, list):
                if isinstance(v[0], torch.Tensor):
                    batch[k] = torch.stack(v).type(self.data_key_type_map[k])
                elif isinstance(v[0], list):
                    batch[k] = [torch.tensor(vv, dtype=self.data_key_type_map[k]) for vv in v]
                else:
                    batch[k] = torch.tensor(v, dtype=self.data_key_type_map[k])

        if self.data_args.data_tokenizer_args:
            # generate overflow sample ids
            batch[DataKeys.OVERFLOW_MAPPING] = []
            for idx, token_ids in enumerate(batch[DataKeys.TOKEN_IDS]):
                for _ in range(len(token_ids)):
                    batch[DataKeys.OVERFLOW_MAPPING].append(idx)
            batch[DataKeys.OVERFLOW_MAPPING] = torch.tensor(batch[DataKeys.OVERFLOW_MAPPING])

            # generate overflow token mapping
            overflow_to_sample_matrix = torch.zeros(
                len(batch["overflow_to_sample_mapping"]),
                batch["overflow_to_sample_mapping"].max() + 1,
            ).scatter_(1, batch["overflow_to_sample_mapping"].unsqueeze(1), 1.0)
            overflow_to_sample_matrix = torch.nn.functional.normalize(overflow_to_sample_matrix.T, p=1, dim=1)
            batch["overflow_to_sample_matrix"] = overflow_to_sample_matrix

        return batch
