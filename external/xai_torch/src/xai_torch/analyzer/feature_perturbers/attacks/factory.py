from xai_torch.analyzer.feature_perturbers.attacks.base import AttacksBase
from xai_torch.analyzer.feature_perturbers.attacks.constants import ATTACKS_REGISTRY
from xai_torch.utilities.decorators import register_as_child


def register_attack(reg_name: str = ""):
    return register_as_child(
        base_class_type=AttacksBase,
        registry=ATTACKS_REGISTRY,
        reg_name=reg_name,
    )


class AttacksFactory:
    @staticmethod
    def create(name: str, **kwargs):
        attack_class = ATTACKS_REGISTRY.get(name, None)
        if attack_class is None:
            raise ValueError(f"Perturbation attack [{name}] is not supported.")
        return attack_class(
            **kwargs,
        )
