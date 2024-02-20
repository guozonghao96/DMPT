import torch

from ..builder import BBOX_ASSIGNERS
from ..match_costs import build_match_cost
from ..transforms import bbox_cxcywh_to_xyxy
from .assign_result import AssignResult
from .base_assigner import BaseAssigner

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


@BBOX_ASSIGNERS.register_module()
class HungarianPointAssigner(BaseAssigner):

    def __init__(self,
                 cls_cost=dict(type='ClassificationCost', weight=1.),
                 reg_cost=dict(type='PointL1Cost', weight=1.0),
                ):
        self.cls_cost = build_match_cost(cls_cost)
        self.reg_cost = build_match_cost(reg_cost)
        
    def assign(self,
               point_pred,
               cls_pred,
               gt_points,
               gt_labels,
               img_meta,
               gt_bboxes_ignore=None,
               eps=1e-7):
        
        num_gts, num_points = gt_points.size(0), point_pred.size(0)

        # 1. assign -1 by default
        assigned_gt_inds = point_pred.new_full((num_points, ),
                                              -1,
                                              dtype=torch.long)
        assigned_labels = point_pred.new_full((num_points, ),
                                             -1,
                                             dtype=torch.long)
        if num_gts == 0 or num_points == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)
        img_h, img_w, _ = img_meta['img_shape']
        factor = gt_points.new_tensor([img_w, img_h]).unsqueeze(0)
        # 2. compute the weighted costs
        # classification and bboxcost.
        cls_cost = self.cls_cost(cls_pred, gt_labels)
        # regression L1 cost
        normalize_gt_points = gt_points / factor
        reg_cost = self.reg_cost(point_pred, normalize_gt_points)
        # regression iou cost, defaultly giou is used in official DETR.
        # weighted sum of above three costs
        cost = cls_cost + reg_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                                'to install scipy first.')
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(
            point_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(
            point_pred.device)
        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        return AssignResult(
            num_gts, assigned_gt_inds, None, labels=assigned_labels)
        
