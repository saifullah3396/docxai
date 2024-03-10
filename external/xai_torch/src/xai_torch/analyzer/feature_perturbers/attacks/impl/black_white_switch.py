from dataclasses import dataclass

import numpy as np
import torch
from xai_torch.analyzer.feature_perturbers.attacks.base import AttacksBase
from xai_torch.analyzer.feature_perturbers.attacks.factory import register_attack


@register_attack("black_white_switch")
@dataclass
class BlackWhiteSwitch(AttacksBase):
    black_white_threshold: float = 0.5

    def __post_init__(self):
        super().__post_init__()
        self._black_white_regions = None

    def get_rand_indices(self, attr_map_grid, positive_attr_mask):
        if self._rand_indices is None:
            white_feature_mask = torch.logical_and(
                positive_attr_mask, self._black_white_regions[:: self.grid_cell_size, :: self.grid_cell_size] == True
            )
            black_feature_mask = torch.logical_and(
                positive_attr_mask, self._black_white_regions[:: self.grid_cell_size, :: self.grid_cell_size] == False
            )
            rand_indices_white = white_feature_mask.nonzero()
            rand_indices_black = black_feature_mask.nonzero()
            black_prob_total = 0.5
            white_prob_total = 1 - 0.5
            black_prob = black_prob_total / rand_indices_black.shape[0]
            white_prob = white_prob_total / rand_indices_white.shape[0]
            probs = torch.zeros_like(attr_map_grid)
            probs[black_feature_mask] = black_prob
            probs[white_feature_mask] = white_prob
            nelements = attr_map_grid[positive_attr_mask].nelement()
            p = probs[positive_attr_mask].cpu().numpy()
            p /= p.sum()
            rand_indices = np.random.choice(
                nelements,
                size=nelements,
                replace=False,
                p=p,
            ).flatten()
            self._rand_indices = torch.from_numpy(rand_indices)
        return self._rand_indices

    def setup(self, attr_map_grid, image):
        super().setup(attr_map_grid, image)
        kx = self.grid_cell_size
        ky = self.grid_cell_size
        black_white_regions = (
            (image[0].unfold(0, ky, kx).unfold(1, ky, kx) < self.black_white_threshold).any(dim=2).any(dim=2)
        )
        black_white_regions = black_white_regions.repeat_interleave(ky, dim=0).repeat_interleave(kx, dim=1)
        self._black_white_regions = ~black_white_regions

    def apply_attack(self, images, attack_masks):
        attack_masks_white = []
        attack_masks_black = []
        for attack_mask in attack_masks:
            attack_masks_white.append(torch.logical_and(self._black_white_regions == True, attack_mask))
            attack_masks_black.append(torch.logical_and(self._black_white_regions == False, attack_mask))
        attack_masks_white = torch.stack(attack_masks_white)
        attack_masks_black = torch.stack(attack_masks_black)
        # import matplotlib.pyplot as plt

        # idx = 25
        # f, (ax0, ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 6)

        images = images.permute(0, 2, 3, 1)
        # ax5.imshow(images[idx])

        images[attack_masks_white] = 0.0  # just setting indices to black
        images[attack_masks_black] = 1.0  # just setting indices to black

        # print(
        #     images.shape,
        #     attack_masks.shape,
        # )
        # ax0.imshow(images[idx])
        # ax1.imshow(attack_masks[idx])
        # ax2.imshow(self._black_white_regions)
        # ax3.imshow(attack_masks_white[idx])
        # ax4.imshow(attack_masks_black[idx])
        # plt.show()
        return {
            "image": images.permute(0, 3, 1, 2),
            "attack_masks_black": attack_masks_black,
            "attack_masks_white": attack_masks_white,
        }
