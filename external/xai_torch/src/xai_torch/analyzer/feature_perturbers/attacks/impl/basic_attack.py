from dataclasses import dataclass

from xai_torch.analyzer.feature_perturbers.attacks.base import AttacksBase
from xai_torch.analyzer.feature_perturbers.attacks.factory import register_attack


@register_attack("basic_attack")
@dataclass
class BasicAttack(AttacksBase):
    pass
