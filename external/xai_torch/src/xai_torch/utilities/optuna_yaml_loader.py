"""Defines the yaml reader for optuna tags."""


import yaml

OPTUNA_FN_DICT = {
    "suggest_categorical": lambda trial, key, values: trial.suggest_categorical(key, values),
    "suggest_discrete_uniform": lambda trial, key, values: trial.suggest_discrete_uniform(key, *values),
    "suggest_float": lambda trial, key, values: trial.suggest_float(key, *values),
    "suggest_int": lambda trial, key, values: trial.suggest_int(key, *values),
    "suggest_loguniform": lambda trial, key, values: trial.suggest_loguniform(key, *values),
    "suggest_uniform": lambda trial, key, values: trial.suggest_uniform(key, *values),
}


class OptunaTag(yaml.YAMLObject):
    yaml_tag = "!optuna"

    def __init__(self, values, type):
        self.values = values
        self.type = type

    def setup_suggestion(self, key, trial):
        fn = OPTUNA_FN_DICT.get(self.type, None)
        if fn is None:
            raise ValueError(f"Optuna trial suggestion of type [{self.type}] does not exist.")
        if trial is not None:
            return fn(trial, key, self.values)
        else:
            return self.values[0]

    def __repr__(self):
        return "Optuna(values={}, type={})".format(self.values, self.type)

    @classmethod
    def from_yaml(cls, loader, node):
        return OptunaTag(*eval(node.value))

    @classmethod
    def to_yaml(cls, dumper, data):
        return dumper.represent_scalar(cls.yaml_tag, (data.values, data.type))
