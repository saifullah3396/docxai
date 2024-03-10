from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Union

from xai_torch.core.constants import DataKeys
from xai_torch.core.data.tokenizers.base import Tokenizer
from xai_torch.core.factory.constants import HF_TOKENIZERS_REGISTRY
from xai_torch.core.factory.decorators import register_tokenizer

if TYPE_CHECKING:
    from transformers.file_utils import PaddingStrategy
    from transformers.tokenization_utils_base import TruncationStrategy


@dataclass
class HuggingfaceTokenizerInitKwargs:
    cache_dir: str = f"{os.environ['XAI_TORCH_OUTPUT_DIR']}/.huggingface/"
    local_files_only: bool = True
    add_prefix_space: bool = True
    do_lower_case: bool = True


@dataclass
class HuggingfaceTokenizerCallKwargs:
    add_special_tokens: bool = True
    padding: Union[bool, str, PaddingStrategy] = "max_length"
    truncation: Union[bool, str, TruncationStrategy] = True
    max_length: Optional[int] = 512
    stride: int = 0
    pad_to_multiple_of: Optional[int] = 8


@register_tokenizer("hugging_face")
@dataclass
class HuggingfaceTokenizer(Tokenizer):
    tokenizer_class: Optional[str] = None
    init_kwargs: HuggingfaceTokenizerInitKwargs = HuggingfaceTokenizerInitKwargs()
    call_kwargs: HuggingfaceTokenizerCallKwargs = HuggingfaceTokenizerCallKwargs()
    overflow_samples_combined: bool = field(
        default=False,
        metadata={
            "help": "Whether to combine overflowing tokens into one sample or to make "
            "multiple separate samples for those."
        },
    )
    compute_word_to_toke_maps: bool = field(
        default=False, metadata={"help": "Whether to compute word to token mapping."}
    )

    def __post_init__(self):
        if self.tokenizer_class is not None:
            if self.tokenizer_class in HF_TOKENIZERS_REGISTRY:
                self.tokenizer = HF_TOKENIZERS_REGISTRY[self.tokenizer_class].from_pretrained(
                    self.name, **dataclass.asdict(self.init_kwargs)
                )
            else:
                raise ValueError(f"Input tokenizer class [{self.tokenizer_class}] is not supported.")
        else:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.name, **dataclasses.asdict(self.init_kwargs))

    def __call__(
        self,
        data: Union[Iterable, List, Dict],
    ) -> List:
        """
        Tokenizes the data using the associated torchtext tokenizer.

        Args:
            data: List of words.
            data_args: The dataset arguments.
        """

        # tokenize the words
        if isinstance(data, dict):
            tokenized_words = self.tokenizer(
                **data,
                **dataclasses.asdict(
                    self.call_kwargs,
                ),
                is_split_into_words=True,
                return_overflowing_tokens=True,  # set some arguments that we need to stay fixed for our case
                return_token_type_ids=None,
                return_attention_mask=None,
                return_special_tokens_mask=False,
                return_offsets_mapping=False,
                return_length=False,
                return_tensors=None,
                verbose=True,
            )
        else:
            tokenized_words = self.tokenizer(
                data,
                **dataclasses.asdict(
                    self.call_kwargs,
                ),
                is_split_into_words=True,
                return_overflowing_tokens=True,  # set some arguments that we need to stay fixed for our case
                return_token_type_ids=None,
                return_attention_mask=None,
                return_special_tokens_mask=False,
                return_offsets_mapping=False,
                return_length=False,
                return_tensors=None,
                verbose=True,
            )

        return tokenized_words

    def tokenize_sample(self, sample: dict):
        tokenized_data = self(sample[DataKeys.WORDS])

        if self.overflow_samples_combined:
            tokenized_size = len(tokenized_data["input_ids"])  # token_ids -> input_ids in huggingface
            token_bboxes_list = []
            token_angles_list = []
            for batch_index in range(tokenized_size):
                word_ids = tokenized_data.word_ids(batch_index=batch_index)
                token_bboxes = []
                token_angles = []

                for (idx, word_idx) in enumerate(word_ids):
                    # Special tokens have a word id that is None. We set the labels only
                    # for our known words since our classifier is word basd
                    # We set the label and bounding box for the first token of each word.
                    if word_idx is not None:
                        token_bboxes.append(sample[DataKeys.WORD_BBOXES][word_idx])
                        token_angles.append(sample[DataKeys.WORD_ANGLES][word_idx])
                    else:
                        token_bboxes.append([0, 0, 0, 0])
                        token_angles.append(0)

                token_bboxes_list.append(token_bboxes)
                token_angles_list.append(token_angles)

            sample[DataKeys.TOKEN_IDS] = tokenized_data["input_ids"]  # token_ids -> input_ids in huggingface
            sample[DataKeys.TOKEN_BBOXES] = token_bboxes_list
            sample[DataKeys.TOKEN_ANGLES] = token_angles_list
            for k in [
                DataKeys.ATTENTION_MASKS,
                DataKeys.TOKEN_TYPE_IDS,
            ]:
                if k in tokenized_data:
                    sample[k] = tokenized_data[k]

            if len(sample[DataKeys.TOKEN_IDS]) > self.max_seqs_per_sample:
                indices = list(range(len(sample[DataKeys.TOKEN_IDS])))
                # random.shuffle(indices)
                indices = indices[: self.max_seqs_per_sample]
                for k in [
                    DataKeys.TOKEN_IDS,
                    DataKeys.ATTENTION_MASKS,
                    DataKeys.TOKEN_TYPE_IDS,
                    DataKeys.TOKEN_BBOXES,
                    DataKeys.TOKEN_ANGLES,
                    DataKeys.WORD_TO_TOKEN_MAPS,
                ]:
                    if k in sample:
                        sample[k] = [sample[k][i] for i in indices]
            return sample
        else:
            raise ValueError("overflow_samples_combined cannot be set to False for per sample " "tokenization.")
