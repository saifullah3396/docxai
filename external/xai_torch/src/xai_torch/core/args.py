"""Defines the Arguments class that holds all the possible arguments."""
# from __future__ import annotations

import logging

# if TYPE_CHECKING:
from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Type, Union

import optuna

from xai_torch.core.analyzer.args import AnalyzerArguments  # noqa
from xai_torch.core.args_base import ArgumentsBase
from xai_torch.core.data.args.data_args import DataArguments
from xai_torch.core.general_args import GeneralArguments
from xai_torch.core.models.args import ModelArguments
from xai_torch.core.training.args.training import TrainingArguments

# from xai_torch.differential_privacy.args import PrivacyArguments # noqa
# from xai_torch.federated_learning.args import FederatedArguments # noqa


@dataclass
class Arguments(ArgumentsBase):
    """
    A aggregated container for all the arguments required for model configuration,
    data loading, and training.
    """

    # General arguments
    general_args: GeneralArguments = field(default=GeneralArguments())

    # Data related arguments
    data_args: DataArguments = field(default=DataArguments())

    # Training related arguments
    training_args: TrainingArguments = field(default=TrainingArguments())

    # Model related arguments
    model_args: Optional[ModelArguments] = field(default=None)

    # Add analyzer arguments
    analyzer_args: Optional[AnalyzerArguments] = field(default=None)

    # privacy_args: PrivacyArguments = field(default=None)
    # federated_args: FederatedArguments = field(default=None)

    def __repr__(self) -> str:
        return super().__repr__()


@dataclass
class ArgumentsLoader:
    arguments_class = Arguments

    def __init__(self, cfg: Union[Path, str], extra_args: Optional[Namespace] = None) -> None:
        """
        Initializes the arguments from a yaml configuration file along with extra args
        passed through command line.

        Args:
            cfg: Path to the yaml configuration file from which to load the arguments.
            extra_args: Extra command line arguments that override the arguments in cfg.
        """

        # assign parameters
        self.cfg = cfg

        self.extra_args = extra_args

    def __post_init__(self):
        if self.general.do_train or self.general.do_eval or self.general.do_test:
            if self.model_args is None:
                raise ValueError("Model arguments must be provided in the configuration file for training.")

    def _load_child_arg_from_dict(self, arg_type: Type[ArgumentsBase], data: dict):
        """Reads a child argument from dict data."""
        from xai_torch.utilities.dacite_wrapper import Config, from_dict

        return from_dict(data_class=arg_type, data=data, config=Config(strict=True))

    def _load_child_args_from_dict(self, args: Arguments, data: dict, field_name: Optional[str] = None):
        """Reads all the child arguments."""
        from xai_torch.utilities.dacite_wrapper import get_fields

        # assign child argument types
        child_arg_fields = get_fields(args)

        # field names
        # field_names = [field_name]
        # for k in data.keys():
        #     if k not in field_names:
        #         logging.warning(f"{k} arguments provided when they are not present in Arguments class.")

        # load each child argument from the file
        for field in child_arg_fields:
            if field.name not in data:
                logging.warning(f"{field.name} arguments not found in the yaml config {self.cfg}.")
                continue

            # set child arg to self
            setattr(
                args,
                field.name,
                self._load_child_arg_from_dict(field.type, data[field.name]),
            )

    # def _load_extra_args(self, args: Arguments):
    #     """Substitutes the extra args into the arguments of the class."""
    #     extra_args_list = list(self.extra_args)

    #     # convert args into key value pairs
    #     keys = extra_args_list[::2]
    #     values = extra_args_list[1::2]
    #     extra_args = dict(zip(keys, values))

    #     args_dict = dataclasses.asdict(args)

    #     def get_field_names(field_names, input):
    #         fields = []
    #         if isinstance(input, dict):
    #             fields = list(input.keys())
    #         print(fields)
    #         for field in fields:
    #             field_names.append(field.name)
    #             get_field_names(field_names, field.value)
    #         return field_names

    #     print(get_field_names([], args_dict))

    def setup_optuna_vars(self, data: dict, trial: optuna.trial.Trial = None):
        from xai_torch.utilities.optuna_yaml_loader import OptunaTag

        for key, value in data.items():
            if isinstance(value, OptunaTag):
                data[key] = value.setup_suggestion(key, trial)
            elif isinstance(value, dict):
                self.setup_optuna_vars(value, trial=trial)
            elif isinstance(value, list):
                for v in value:
                    if isinstance(v, dict):
                        self.setup_optuna_vars(v, trial=trial)

    def load_from_yaml_file(
        self, arguments_class=Arguments, field_name: Optional[str] = None, trial: optuna.trial.Trial = None
    ):
        """Loads the arguments from a yaml file."""
        from xai_torch.utilities.file_io import read_yaml_file

        # initialize arguments object
        args = arguments_class()

        # load the arguments from the yaml configuration file
        data = read_yaml_file(self.cfg)

        # setup optuna variables from data
        self.setup_optuna_vars(data, trial=trial)

        if field_name is not None:
            if field_name not in data:
                raise ValueError(f"{field_name} arguments not found in the yaml config {self.cfg}.")

            from xai_torch.utilities.dacite_wrapper import Config, from_dict

            return from_dict(data_class=args.__class__, data=data[field_name], config=Config(strict=True))
        else:
            # load child arguments into class variables
            self._load_child_args_from_dict(args, data, field_name=field_name)

        # replace arguments with extra args passed to cli
        # self._load_extra_args(args)

        return args

    @staticmethod
    def parse(
        arguments_class=Arguments, field_name: Optional[str] = None, trial: optuna.trial.Trial = None
    ) -> Tuple[Arguments, str]:
        """
        Parses the arguments from command line and initializes the class.
        """
        try:
            import argparse
            import importlib
            from pathlib import Path

            # create a generic arguments parser and look for cfg argument.
            arg_parser_main = argparse.ArgumentParser()
            arg_parser_main.add_argument("--cfg", required=True)
            arg_parser_main.add_argument("--task", default=0)

            # parse args
            cli_args, unknown = arg_parser_main.parse_known_args()

            if not Path(cli_args.cfg).exists():
                raise ValueError(f"Input cfg file [{cli_args.cfg}] does not exist.")

            # initialize the arguments. No longer use yaml files for loading arguments
            # args = ArgumentsLoader(cli_args.cfg, unknown).load_from_yaml_file(
            #     arguments_class=arguments_class, trial=trial, field_name=field_name
            # )

            import importlib.util
            import sys

            spec = importlib.util.spec_from_file_location("task_cfg", cli_args.cfg)
            cfg_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cfg_module)

            return list(cfg_module.TASKS.values())[int(cli_args.task)], cli_args.cfg
        except Exception as e:
            logging.exception(f"Exception raised while initializing arguments from config file: {e}")
            sys.exit(1)
