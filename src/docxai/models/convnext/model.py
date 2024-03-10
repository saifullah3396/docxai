""" PyTorch lightning module for the visual backbone of the AlexNetv2 model. """


from dataclasses import dataclass
from typing import Any, Dict

from torch import nn
from xai_torch.core.models.base_config import ModelConfig
from xai_torch.core.models.decorators import register_model
from xai_torch.core.models.image_base import BaseModuleForImageClassification

from docxai.models.convnext.definition import convnext_base, convnext_large, convnext_xlarge


def get_num_layer_for_convnext(var_name):
    """
    Divide [3, 3, 27, 3] layers into 12 groups; each group is three
    consecutive blocks, including possible neighboring downsample layers;
    adapted from https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py
    """
    num_max_layer = 12
    if var_name.startswith("downsample_layers"):
        stage_id = int(var_name.split(".")[1])
        if stage_id == 0:
            layer_id = 0
        elif stage_id == 1 or stage_id == 2:
            layer_id = stage_id + 1
        elif stage_id == 3:
            layer_id = 12
        return layer_id

    elif var_name.startswith("stages"):
        stage_id = int(var_name.split(".")[1])
        block_id = int(var_name.split(".")[2])
        if stage_id == 0 or stage_id == 1:
            layer_id = stage_id + 1
        elif stage_id == 2:
            layer_id = 3 + block_id // 3
        elif stage_id == 3:
            layer_id = 12
        return layer_id
    else:
        return num_max_layer + 1


class LayerDecayValueAssigner(object):
    def __init__(self, values):
        self.values = values

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        return get_num_layer_for_convnext(var_name)


def get_parameter_groups(model, weight_decay=1e-5, skip_list=(), get_num_layer=None, get_layer_scale=None):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            group_name = "no_decay"
            this_weight_decay = 0.0
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.0

            parameter_group_names[group_name] = {
                "name": group_name,
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale,
            }
            parameter_group_vars[group_name] = {
                "name": group_name,
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale,
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    # print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


@register_model(reg_name="convnext", task="image_classification")
class ConvNextForImageClassification(BaseModuleForImageClassification):
    @dataclass
    class Config(ModelConfig):
        use_layer_decay: bool = True
        layer_decay: float = 0.8
        head_init_scale: float = 0.001
        drop_path_rate: float = 0.4

    @classmethod
    def load_model(cls, model_name, num_labels, use_timm=False, pretrained=True, config=None, cache_dir=''):
        if model_name == "convnext_base":
            model = convnext_base(
                pretrained=pretrained,
                in_22k=True,
                num_classes=num_labels,
                drop_path_rate=config.drop_path_rate,
                head_init_scale=config.head_init_scale,
            )
        elif model_name == "convnext_large":
            model = convnext_large(
                pretrained=pretrained,
                in_22k=True,
                num_classes=num_labels,
                drop_path_rate=config.drop_path_rate,
                head_init_scale=config.head_init_scale,
            )
        elif model_name == "convnext_xlarge":
            model = convnext_xlarge(
                pretrained=pretrained,
                in_22k=True,
                num_classes=num_labels,
                drop_path_rate=config.drop_path_rate,
                head_init_scale=config.head_init_scale,
            )
        return cls.update_classifier_for_labels(model, num_labels=num_labels)

    @classmethod
    def update_classifier_for_labels(cls, model, num_labels):
        num_ftrs = model.head.in_features
        model.head = nn.Linear(num_ftrs, num_labels)
        return model

    def get_param_groups(self):
        if self.config.use_layer_decay:
            self.config.bypass_params_creation = True
            if self.config.layer_decay < 1.0 or self.config.layer_decay > 1.0:
                # convnext layers divided into 12 parts, each with a different
                # decayed lr value.
                num_layers = 12
                assigner = LayerDecayValueAssigner(
                    list(self.config.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2))
                )
            else:
                assigner = None

            # if assigner is not None:
            #     print("Assigned values = %s" % str(assigner.values))

            skip = {}
            if hasattr(self.model, "no_weight_decay"):
                skip = self.model.no_weight_decay()
            wd = self.training_args.optimizers["default"].group_params[0].kwargs["weight_decay"]
            self.trainable_parameters = get_parameter_groups(
                self.model, wd, skip, assigner.get_layer_id, assigner.get_scale
            )
            return {
                "default": self.trainable_parameters,
            }
        else:
            self.config.bypass_params_creation = False

            return super().get_param_groups()

    def on_load_checkpoint(self, checkpoint: Dict[str, Any], strict: bool = True):
        state_dict_key = "state_dict"
        checkpoint_fixed = {state_dict_key: {}}
        for state in checkpoint[state_dict_key]:
            updated_state = state
            if "model." not in updated_state:
                updated_state = "model." + updated_state
            if "model.model." in updated_state:
                updated_state = updated_state.replace("model.model.", "model.")
            if "layer_scale.gamma" not in updated_state and "gamma" in updated_state:
                updated_state = updated_state.replace("gamma", "layer_scale.gamma")

            checkpoint_fixed[state_dict_key][updated_state] = checkpoint[state_dict_key][state]
        return super().on_load_checkpoint(checkpoint_fixed, strict)


SUPPORTED_TASKS = {
    "image_classification": ConvNextForImageClassification,
}
