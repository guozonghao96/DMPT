import torch

from mmdet.utils import util_mixins


class PointSamplingResult(util_mixins.NiceRepr):

    def __init__(self, pos_inds, neg_inds, points, gt_points, assign_result,
                 gt_flags):
        self.pos_inds = pos_inds
        self.neg_inds = neg_inds
        self.pos_points = points[pos_inds]
        self.neg_points = points[neg_inds]
        self.pos_is_gt = gt_flags[pos_inds]

        self.num_gts = gt_points.shape[0]
        self.pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1

        if gt_points.numel() == 0:
            # hack for index error case
            assert self.pos_assigned_gt_inds.numel() == 0
            self.pos_gt_points = torch.empty_like(gt_points).view(-1, 2)
        else:
            if len(gt_points.shape) < 2:
                gt_points = gt_points.view(-1, 2)

            self.pos_gt_points = gt_points[self.pos_assigned_gt_inds, :]

        if assign_result.labels is not None:
            self.pos_gt_labels = assign_result.labels[pos_inds]
        else:
            self.pos_gt_labels = None
            
        self.inner_matches = assign_result.inner_matches

    @property
    def points(self):
        return torch.cat([self.pos_points, self.neg_points])