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
class HungarianKeyPointAssigner(BaseAssigner):

    def __init__(self,
                 cls_cost=dict(type='ClassificationCost', weight=1.),
                 reg_cost=dict(type='KeyPointL1Cost', weight=1.0),
                 inner_match=True,
                 bbox_reg_cost=dict(type='BBoxL1Cost', weight=1.0),
                 bbox_iou_cost=dict(type='IoUCost', weight=1.0),
                ):
        self.cls_cost = build_match_cost(cls_cost)
        self.reg_cost = build_match_cost(reg_cost)
        self.inner_match = inner_match
        self.bbox_reg_cost = build_match_cost(bbox_reg_cost)
        self.bbox_iou_cost = build_match_cost(bbox_iou_cost)
        
    def assign(self,
               bbox_pred,
               cls_pred,
               point_pred,
               gt_bboxes,
               gt_labels,
               gt_points,
               gt_visible_weights,
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
        # 2. compute the weighted costs
        # classification and bboxcost.
        cls_cost = self.cls_cost(cls_pred, gt_labels)
        # bbox的reg
#         factor = gt_bboxes.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)
#         normalize_gt_bboxes = gt_bboxes / factor
#         bbox_reg_cost = self.bbox_reg_cost(bbox_pred, normalize_gt_bboxes)
        
#         bboxes = bbox_cxcywh_to_xyxy(bbox_pred) * factor
#         bbox_iou_cost = self.bbox_iou_cost(bboxes, gt_bboxes)
        
        # regression L1 cost
        factor = gt_points.new_tensor([img_w, img_h]).unsqueeze(0)
        normalize_gt_points = gt_points.reshape(num_gts, -1, 2) / factor.unsqueeze(0)
        normalize_gt_points = normalize_gt_points.reshape(num_gts, -1)
        
        if self.inner_match:
            reg_cost, inner_matched_inds = self.reg_cost(point_pred, 
                                                         normalize_gt_points,
                                                         gt_visible_weights.reshape(num_gts, -1))
            # cost = cls_cost.detach().cpu() + bbox_reg_cost.detach().cpu() + \
            #        bbox_iou_cost.detach().cpu() + reg_cost
            cost = cls_cost.detach().cpu() + reg_cost
        else:
            reg_cost = self.reg_cost(point_pred, normalize_gt_points)
            # regression iou cost, defaultly giou is used in official DETR.
            # weighted sum of above three costs
            # cost = cls_cost + bbox_reg_cost + bbox_iou_cost + reg_cost
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
        
#         inner_matches = tuple(inner_matched_inds[0][matched_row_inds], 
#                               inner_matched_inds[1][matched_row_inds])
        inner_matches_row_inds = inner_matched_inds[0][matched_row_inds, matched_col_inds]
        inner_matches_col_inds = inner_matched_inds[1][matched_row_inds, matched_col_inds]
        
        return AssignResult(
            num_gts, assigned_gt_inds, None, labels=assigned_labels, 
            inner_matches=(inner_matches_row_inds, inner_matches_col_inds))
        
