from dataclasses import dataclass

from xai_torch.analyzer.feature_perturbers.attacks.base import AttacksBase
from xai_torch.analyzer.feature_perturbers.attacks.factory import register_attack


@register_attack("black_white_dropout")
@dataclass
class BlackWhiteDropout(AttacksBase):
    def apply_attack(self, images, attack_masks):
        images = images.permute(0, 2, 3, 1)
        images[attack_masks] = 0.5  # just setting indices to black
        return images.permute(0, 3, 1, 2)
