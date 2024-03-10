"""
Defines the dataclass for holding tokenizer arguments.
"""

from dataclasses import dataclass



@dataclass
class TokenizersArguments:
    """
    Dataclass that holds the tokenizer related arguments.
    """

    # Data tokenizer lirary to use
    strategy: str = ""

    # # Tokenizer configuration
    # config: Optional[dict] = None

    # # Whether to tokenize the sample on the go while loading the samples
    # tokenize_per_sample: bool = False
