import torch

from ..builder import BBOX_SAMPLERS
from .base_sampler import BaseSampler
from .sampling_result import SamplingResult
from .point_sampling_result import PointSamplingResult


@BBOX_SAMPLERS.register_module()
class PointPseudoSampler(BaseSampler):
    """A pseudo sampler that does not do sampling actually."""

    def __init__(self, **kwargs):
        pass

    def _sample_pos(self, **kwargs):
        """Sample positive samples."""
        raise NotImplementedError

    def _sample_neg(self, **kwargs):
        """Sample negative samples."""
        raise NotImplementedError

    def sample(self, assign_result, points, gt_points, **kwargs):
        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        gt_flags = points.new_zeros(points.shape[0], dtype=torch.uint8)
        sampling_result = PointSamplingResult(pos_inds, neg_inds, points, gt_points,
                                         assign_result, gt_flags)
        return sampling_result
