from dataclasses import dataclass

import torch
from matplotlib import pyplot as plt

from xai_torch.analyzer.feature_perturbers.attacks.base import AttacksBase
from xai_torch.analyzer.feature_perturbers.attacks.factory import register_attack


@register_attack("black_dropout")
@dataclass
class BlackDropout(AttacksBase):
    black_white_threshold: int = 0.75

    def __post_init__(self):
        super().__post_init__()
        self._black_white_regions = None

    def get_desc_indices(self, attr_map_grid, positive_attr_mask):
        if self._desc_indices is None:
            black_positive_feature_mask = torch.logical_and(
                positive_attr_mask, self._black_white_regions[:: self.grid_cell_size, :: self.grid_cell_size] == False
            )

            # get only positive values from attr grid
            masked_attr_map_grid = attr_map_grid[black_positive_feature_mask]

            # get top k values from the grid
            _, self._desc_indices = torch.sort(masked_attr_map_grid, descending=True)
        return self._desc_indices

    def get_asc_indices(self, attr_map_grid, positive_attr_mask):
        if self._asc_indices is None:
            black_positive_feature_mask = torch.logical_and(
                positive_attr_mask, self._black_white_regions[:: self.grid_cell_size, :: self.grid_cell_size] == False
            )

            # get only positive values from attr grid
            masked_attr_map_grid = attr_map_grid[black_positive_feature_mask]

            # get top k values from the grid
            _, self._asc_indices = torch.sort(masked_attr_map_grid, descending=False)
        return self._asc_indices

    def get_rand_indices(self, attr_map_grid, positive_attr_mask):
        if self._rand_indices is None:
            black_positive_feature_mask = torch.logical_and(
                positive_attr_mask, self._black_white_regions[:: self.grid_cell_size, :: self.grid_cell_size] == False
            )

            self._rand_indices = torch.randperm(attr_map_grid[black_positive_feature_mask].nelement())
        return self._rand_indices

    def get_attack_indices(self, attr_map_grid, positive_attr_mask, n_drops):
        top_values = torch.zeros_like(attr_map_grid).bool()

        if self.importance_order == "descending":
            # get indices for grid sorted in descending order
            indices = self.get_desc_indices(attr_map_grid, positive_attr_mask)
        elif self.importance_order == "ascending":
            # get indices for grid sorted in ascending order
            indices = self.get_asc_indices(attr_map_grid, positive_attr_mask)
        elif self.importance_order == "random":
            indices = self.get_rand_indices(attr_map_grid, positive_attr_mask)
        else:
            logger.error(f"importance_order [{self.importance_order}] is not supported.")
            exit(1)

        black_positive_feature_mask = torch.logical_and(
            positive_attr_mask, self._black_white_regions[:: self.grid_cell_size, :: self.grid_cell_size] == False
        )
        # positive_attr_mask.nonzero() is the list of pixel indices [[0, 0], [1, 1]]
        # indices is the the list of indices if flattened [0, 1, 2]
        patch_xy_indices = black_positive_feature_mask.nonzero()[indices[:n_drops]]

        # all the coordinates that are set to True will be replaced by the attack
        top_values[patch_xy_indices[:, 0], patch_xy_indices[:, 1]] = True
        return top_values

    def create_black_white_regions_mask(self, image):
        kx = self.grid_cell_size
        ky = self.grid_cell_size
        black_white_regions = (
            (image[0].unfold(0, ky, kx).unfold(1, ky, kx) < self.black_white_threshold).any(dim=2).any(dim=2)
        )
        black_white_regions = black_white_regions.repeat_interleave(ky, dim=0).repeat_interleave(kx, dim=1)
        self._black_white_regions = ~black_white_regions

    def apply_attack(self, image, attack_mask):
        assert image.shape[0] == 1
        image[0][attack_mask] = 1.0  # just setting black areas to white
        return image

    def __call__(
        self,
        image,
        n_drops,
        attr_map_grid=None,
        show=False,
    ):
        if n_drops == 0:
            self.create_black_white_regions_mask(image)
            return image

        # clone the image here
        image = image.clone()

        # get positive values over grid
        positive_attr_mask = attr_map_grid > 0

        # make a container for indices to attack
        attack_mask = self.get_attack_indices(attr_map_grid, positive_attr_mask, n_drops)

        # convert grid top_values to image top_values
        # basically we roll out each grid patch into all of its underlying pixel values
        step_y = self.grid_cell_size
        step_x = self.grid_cell_size
        attack_mask = attack_mask.repeat_interleave(step_y, dim=0).repeat_interleave(step_x, dim=1)

        # apply the attack on given pixels
        image = self.apply_attack(image, attack_mask)

        # if show image
        if show:
            plt.imshow(image.cpu().permute(1, 2, 0).numpy(), cmap="gray")
            plt.show()

        return image
