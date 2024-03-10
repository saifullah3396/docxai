import logging
from abc import ABC
from dataclasses import dataclass

import torch
from xai_torch.utilities.logging_utils import DEFAULT_LOGGER_NAME


@dataclass
class AttacksBase(ABC):
    importance_order: int = "descending"
    grid_cell_size: int = 4
    return_perturbation_output: bool = False
    target_value: float = 0.0

    def __post_init__(self):
        self._patch_xy_indices = None
        self._attack_masks = None
        self._attack_indices = None

    def get_desc_indices(self, attr_map_grid):
        masked_attr_map_grid = attr_map_grid.flatten()
        _, desc_indices = torch.sort(masked_attr_map_grid, descending=True)
        return desc_indices

    def get_asc_indices(self, attr_map_grid):
        masked_attr_map_grid = attr_map_grid.flatten()
        _, asc_indices = torch.sort(masked_attr_map_grid, descending=False)
        return asc_indices

    def get_rand_indices(self, attr_map_grid):
        masked_attr_map_grid = attr_map_grid.flatten()
        rand_indices = torch.randperm(masked_attr_map_grid.nelement())
        return rand_indices

    def get_attack_indices(self, attr_map_grid):
        if self.importance_order == "descending":
            # get indices for grid sorted in descending order
            indices = self.get_desc_indices(attr_map_grid)
        elif self.importance_order == "ascending":
            # get indices for grid sorted in ascending order
            indices = self.get_asc_indices(attr_map_grid)
        elif self.importance_order == "random":
            indices = self.get_rand_indices(attr_map_grid)
        else:
            logger = logging.getLogger(DEFAULT_LOGGER_NAME)
            logger.error(f"importance_order [{self.importance_order}] is not supported.")
            exit(1)
        return indices

    def apply_attack(self, images, attack_masks):
        images = images.permute(0, 2, 3, 1)
        images[attack_masks] = self.target_value  # just setting indices to black
        return images.permute(0, 3, 1, 2)

    def setup(self, attr_map_grid, image):
        # get positive values over grid
        self._attack_indices = self.get_attack_indices(attr_map_grid)
        self._patch_xy_indices = (attr_map_grid != torch.nan).nonzero()
        self._attr_map_grid = attr_map_grid

    def __call__(
        self,
        images,
        perturbation_params,
        show=False,
    ):
        step_y = self.grid_cell_size
        step_x = self.grid_cell_size
        attack_masks = torch.zeros((images.shape[0], *self._attr_map_grid.shape)).bool()
        for idx, drop in enumerate(perturbation_params):
            # attr_map_grid.nonzero() is the list of pixel indices [[0, 0], [1, 1]]
            # indices is the the list of indices if flattened [0, 1, 2]
            coordinates = self._patch_xy_indices[self._attack_indices[:drop]]
            attack_masks[idx][coordinates[:, 0], coordinates[:, 1]] = True

        # convert grid top_values to image top_values
        # basically we roll out each grid patch into all of its underlying pixel values
        attack_masks = attack_masks.repeat_interleave(step_y, dim=1).repeat_interleave(step_x, dim=2)

        # apply the attack on given pixels
        images = self.apply_attack(images, attack_masks)
        return images
