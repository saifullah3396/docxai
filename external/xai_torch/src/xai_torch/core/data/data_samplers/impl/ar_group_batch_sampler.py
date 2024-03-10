"""
Defines the GroupBatchSampler batch sampling strategy.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from xai_torch.core.data.data_samplers.impl.group_batch_sampler import GroupBatchSampler
from xai_torch.core.factory.decorators import register_batch_sampler

if TYPE_CHECKING:
    from torch.utils.data.sampler import Sampler


@register_batch_sampler(reg_name="aspect_ratio")
class AspectRatioGroupBatchSampler(GroupBatchSampler):
    """
    Groups the input sample images based on their aspect ratio.
    """

    def __init__(self, sampler: Sampler, group_factor: int, batch_size: int):
        from xai_torch.core.data.data_samplers.utilities import create_aspect_ratio_groups

        group_ids = create_aspect_ratio_groups(
            sampler.data_source,
            k=group_factor,
        )

        super().__init__(sampler=sampler, group_ids=group_ids, batch_size=batch_size)
