from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Union

from xai_torch.core.data.tokenizers.base import Tokenizer
from xai_torch.core.factory.decorators import register_tokenizer


@register_tokenizer("torch_text")
@dataclass
class TorchTextTokenizer(Tokenizer):
    vocab_file: Optional[str] = None

    def __post_init__(self):
        from torchtext.data.utils import get_tokenizer

        self.tokenizer = get_tokenizer(self.tokenizer_name)

        # initialize vocabulary
        self._vocab = None
        if self.vocab_file is not None:
            self.vocab_file = Path(self.vocab_file)
            if not self.vocab_file.exists():
                return ValueError(f"Vocabulary file [{self.vocab_file}] does not exist.")

            if self.vocab_file.suffix == ".pth":
                import torch

                self.vocab = torch.load(self.vocab_file)
            else:
                return ValueError(f"Vocabulary file of type [{self.vocab_file.suffix}] is not supported.")

    def yield_tokens(self, data_iter: Iterable):
        for text in data_iter:
            yield self.tokenizer(text)

    def generate_vocab_from_data(self, data: Union[Iterable, List]):
        from torchtext.vocab import build_vocab_from_iterator

        vocab = build_vocab_from_iterator(self.yield_tokens(data), specials=["<unk>"])
        vocab.set_default_index(self._vocab["<unk>"])
        return vocab

    def __call__(
        self,
        data: Union[Iterable, List],
    ) -> List:
        """
        Tokenizes the data using the associated torchtext tokenizer.

        Args:
            data: List of words.
            data_args: The dataset arguments.
        """

        if self._vocab is None:
            self._vocab = self.generate_vocab_from_data(data)

        return [self._vocab(self.tokenizer(sample)) for sample in data]
