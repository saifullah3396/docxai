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

    def __post_init__(self):
        self._patch_xy_indices = None
        self._attack_masks = None
        self._attack_indices = None

    def get_desc_indices(self, attr_map_grid):
        _, desc_indices = torch.sort(attr_map_grid.flatten(), descending=True)
        return desc_indices

    def get_asc_indices(self, attr_map_grid):
        _, asc_indices = torch.sort(attr_map_grid.flatten(), descending=False)
        return asc_indices

    def get_rand_indices(self, attr_map_grid):
        rand_indices = torch.randperm(attr_map_grid.flatten().nelement())
        return rand_indices

    def get_attack_indices(self, attr_map_grid):
        logger = logging.getLogger(DEFAULT_LOGGER_NAME)

        indices_batch = []
        for grid in attr_map_grid:
            if self.importance_order == "descending":
                # get indices for grid sorted in descending order
                indices = self.get_desc_indices(grid)
            elif self.importance_order == "ascending":
                # get indices for grid sorted in ascending order
                indices = self.get_asc_indices(grid)
            elif self.importance_order == "random":
                indices = self.get_rand_indices(grid)
            else:
                logger.error(f"importance_order [{self.importance_order}] is not supported.")
                exit(1)
            indices_batch.append(indices)
        return torch.stack(indices_batch)

    def apply_attack(self, images, attack_masks):
        images = images.permute(0, 2, 3, 1)
        images[attack_masks] = 0.0  # just setting indices to black
        return images.permute(0, 3, 1, 2)

    def setup(self, attr_map_grid):
        # get positive values over grid
        self._attack_indices = self.get_attack_indices(attr_map_grid)  # .reshape(-1, 1)
        self._patch_xy_indices = torch.stack([grid.nonzero() for grid in attr_map_grid])  # .reshape(-1, 2)
        self._target_shape = attr_map_grid[0].shape

    def __call__(
        self,
        inputs,
        perturbation_params,
        perturbation_len,
    ):
        step_y = self.grid_cell_size
        step_x = self.grid_cell_size
        attack_masks = torch.zeros((inputs.shape[0], *self._target_shape)).bool()

        self._patch_xy_indices = torch.repeat_interleave(self._patch_xy_indices, perturbation_len, dim=0)
        self._attack_indices = torch.repeat_interleave(self._attack_indices, perturbation_len, dim=0)

        for idx, n_perturbs in enumerate(perturbation_params):
            # attr_map_grid.nonzero() is the list of pixel indices [[0, 0], [1, 1]]
            # indices is the the list of indices if flattened [0, 1, 2]
            coordinates = self._patch_xy_indices[idx][self._attack_indices[idx][:n_perturbs]]
            attack_masks[idx][coordinates[:, 0], coordinates[:, 1]] = True

            # import matplotlib.pyplot as plt

            # print(attack_masks[idx].shape)
            # plt.imshow(attack_masks[idx])
            # plt.show()

        # convert grid top_values to image top_values
        # basically we roll out each grid patch into all of its underlying pixel values
        attack_masks = attack_masks.repeat_interleave(step_y, dim=1).repeat_interleave(step_x, dim=2)

        # apply the attack on given pixels
        return self.apply_attack(inputs, attack_masks)
