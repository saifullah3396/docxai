import itertools
import math
import numbers
import tempfile
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from copy import copy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Type, Union, cast

import torch
from ignite.engine import Engine
from ignite.handlers import ConcatScheduler, LRScheduler, ParamGroupScheduler, ParamScheduler, PiecewiseLinear
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
from torch.optim.optimizer import Optimizer


class PolynomialLRDecay(_LRScheduler):
    """Polynomial learning rate decay until step reach to max_decay_step

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_decay_steps: after this step, we stop decreasing learning rate
        end_learning_rate: scheduler stoping learning rate decay, value of learning rate must be this value
        power: The power of the polynomial.
    """

    def __init__(self, optimizer, max_decay_steps, end_learning_rate=0.0001, power=1.0):
        if max_decay_steps <= 1.0:
            raise ValueError("max_decay_steps should be greater than 1.")
        self.max_decay_steps = max_decay_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.max_decay_steps:
            return [self.end_learning_rate for _ in self.base_lrs]

        return [
            (base_lr - self.end_learning_rate) * ((1 - self.last_epoch / self.max_decay_steps) ** (self.power))
            + self.end_learning_rate
            for base_lr in self.base_lrs
        ]

    def _get_closed_form_lr(self):
        return [
            (base_lr - self.end_learning_rate) * ((1 - self.last_epoch / self.max_decay_steps) ** (self.power))
            + self.end_learning_rate
            for base_lr in self.base_lrs
        ]

    # def step(self, step=None):
    #     if step is None:
    #         step = self.last_step + 1
    #     self.last_step = step if step != 0 else 1
    #     if self.last_step <= self.max_decay_steps:
    #         decay_lrs = [
    #             (base_lr - self.end_learning_rate)
    #             * ((1 - self.last_step / self.max_decay_steps) ** (self.power))
    #             + self.end_learning_rate
    #             for base_lr in self.base_lrs
    #         ]
    #         for param_group, lr in zip(self.optimizer.param_groups, decay_lrs):
    #             param_group["lr"] = lr


class CosineScheduler(object):
    def __init__(
        self,
        base_value,
        final_value,
        epochs,
        niter_per_ep,
        warmup_epochs=0,
        start_warmup_value=0,
    ):
        import numpy as np

        warmup_schedule = np.array([])
        warmup_iters = warmup_epochs * niter_per_ep
        if warmup_epochs > 0:
            warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = np.arange(epochs * niter_per_ep - warmup_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

        self.schedule = np.concatenate((warmup_schedule, schedule))
        assert len(self.schedule) == epochs * niter_per_ep
        self.last_step = 0

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items()}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def step(self):
        result = self.schedule[self.last_step]
        self.last_step += 1
        return result


def create_lr_scheduler_with_warmup(
    lr_scheduler: Union[ParamScheduler, _LRScheduler],
    warmup_start_value: float,
    warmup_duration: int,
    warmup_end_value: Optional[float] = None,
    save_history: bool = False,
    output_simulated_values: Optional[List] = None,
) -> "ConcatScheduler":
    """
    Helper method to create a learning rate scheduler with a linear warm-up.

    Args:
        lr_scheduler: learning rate scheduler after the warm-up.
        warmup_start_value: learning rate start value of the warm-up phase.
        warmup_duration: warm-up phase duration, number of events.
        warmup_end_value: learning rate end value of the warm-up phase, (default=None). If None,
             warmup_end_value is set to optimizer initial lr.
        save_history: whether to log the parameter values to
            `engine.state.param_history`, (default=False).
        output_simulated_values: optional output of simulated learning rate values.
            If output_simulated_values is a list of None, e.g. `[None] * 100`, after the execution it will be filled
            by 100 simulated learning rate values.

    Returns:
        ConcatScheduler

    Note:
        If the first learning rate value provided by `lr_scheduler` is different from `warmup_end_value`, an additional
        event is added after the warm-up phase such that the warm-up ends with `warmup_end_value` value and then
        `lr_scheduler` provides its learning rate values as normally.

    Examples:

        .. include:: defaults.rst
            :start-after: :orphan:

        .. testcode::

            from torch.optim.lr_scheduler import ExponentialLR

            torch_lr_scheduler = ExponentialLR(optimizer=default_optimizer, gamma=0.98)

            default_trainer = get_default_trainer()

            scheduler = create_lr_scheduler_with_warmup(torch_lr_scheduler,
                                                        warmup_start_value=0.0,
                                                        warmup_end_value=0.1,
                                                        warmup_duration=3)

            default_trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

            @default_trainer.on(Events.ITERATION_COMPLETED)
            def print_lr():
                print(default_optimizer.param_groups[0]["lr"])

            default_trainer.run([0] * 8, max_epochs=1)

        .. testoutput::

            0.0
            0.05
            0.1
            0.098
            0.09604
            0.09411...
            0.09223...
            0.09039...

    .. versionadded:: 0.4.5
    """
    if not isinstance(lr_scheduler, (ParamScheduler, _LRScheduler)):
        raise TypeError(
            "Argument lr_scheduler should be a subclass of torch.optim.lr_scheduler._LRScheduler or "
            f"ParamScheduler, but given {type(lr_scheduler)}"
        )

    if not isinstance(warmup_duration, numbers.Integral):
        raise TypeError(f"Argument warmup_duration should be integer, but given {warmup_duration}")

    if not (warmup_duration > 1):
        raise ValueError(f"Argument warmup_duration should be at least 2 events, but given {warmup_duration}")

    warmup_schedulers = []  # type: List[ParamScheduler]

    for param_group_index, param_group in enumerate(lr_scheduler.optimizer.param_groups):

        if warmup_end_value is None:
            param_group_warmup_end_value = param_group["lr"]
        else:
            param_group_warmup_end_value = warmup_end_value

        milestones_values = [(0, warmup_start_value), (warmup_duration - 1, param_group_warmup_end_value)]

        if isinstance(lr_scheduler, _LRScheduler):
            init_lr = param_group["lr"]
            if init_lr != param_group_warmup_end_value:
                milestones_values.append((warmup_duration, init_lr))

            # We need to advance torch lr_scheduler to avoid duplicated lr value
            # given by PiecewiseLinear and LRScheduler.
            # We suggest to attach output scheduler on ITERATION_STARTED but
            # torch lr_scheduler works with ITERATION_COMPLETED
            # See also https://github.com/pytorch/ignite/pull/2496#issuecomment-1065984440
            lr_scheduler.last_epoch += 1
            lr_scheduler = LRScheduler(lr_scheduler, save_history=save_history)
        # else:
        #     init_lr = lr_scheduler.get_param()
        #     if init_lr == param_group_warmup_end_value:
        #         if warmup_duration > 2:
        #             d = (param_group_warmup_end_value - warmup_start_value) / (warmup_duration - 1)
        #             milestones_values[-1] = (warmup_duration - 2, param_group_warmup_end_value - d)
        #         else:
        #             milestones_values.pop(-1)

        warmup_schedulers.append(
            PiecewiseLinear(
                lr_scheduler.optimizer,
                param_name="lr",
                milestones_values=milestones_values,
                param_group_index=param_group_index,
                save_history=save_history,
            )
        )

    warmup_scheduler = ParamGroupScheduler(warmup_schedulers, save_history=save_history)

    schedulers = [
        warmup_scheduler,
        lr_scheduler,
    ]  # type: List[Union[ParamScheduler, ParamGroupScheduler, _LRScheduler]]
    durations = [milestones_values[-1][0] + 1]
    combined_scheduler = ConcatScheduler(schedulers, durations=durations, save_history=save_history)

    if output_simulated_values is not None:
        if not isinstance(output_simulated_values, list):
            raise TypeError(
                "Argument output_simulated_values should be a list of None, e.g. `[None] * 100`, "
                f"but given {type(output_simulated_values)}."
            )
        num_events = len(output_simulated_values)
        result = ConcatScheduler.simulate_values(num_events=num_events, schedulers=schedulers, durations=durations)
        for i in range(num_events):
            output_simulated_values[i] = result[i]
    return combined_scheduler
