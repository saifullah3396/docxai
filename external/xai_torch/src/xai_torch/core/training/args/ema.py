from dataclasses import dataclass


@dataclass
class ModelEmaArguments:
    enabled: bool = False

    # EMA decay
    momentum: float = 0.002

    # EMA warmup
    momentum_warmup: float = 0.2

    # warmup iteartions
    warmup_iters = 100

    # update every n epochs
    update_every: int = 5
