import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16, force_fp32
from torch.nn.modules.utils import _pair

from mmdet.models.builder import HEADS, build_loss
from mmdet.models.losses import accuracy
from mmdet.core import build_bbox_coder, multi_apply, multiclass_nms
from mmdet.models.dense_heads.atss_head import reduce_mean
from models.utils import trunc_normal_

from einops import rearrange

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

from ...losses import weight_reduce_loss


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    
@HEADS.register_module()
class MLPHead(nn.Module):

    def __init__(self,
                 in_channels=256,
                 num_classes=20,
                 cls_mlp_depth=3,
                 reg_mlp_depth=3,
                 loss_cls=dict(type='FocalLoss', use_sigmoid=True, 
                          gamma=2.0, alpha=0.25, loss_weight=1.0),
                 loss_point=dict(type='L1Loss', loss_weight=10.0)
                ):
        super(MLPHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.loss_cls = build_loss(loss_cls)
        self.loss_point = build_loss(loss_point)
        self.cls_embed = MLP(in_channels, in_channels, num_classes, cls_mlp_depth)
        self.point_embed = MLP(in_channels, in_channels, 2, reg_mlp_depth)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def init_weights(self):
        self.apply(self._init_weights)
    
    def forward(self, x):
        cls_score = self.cls_embed(x)
        point_pred = self.point_embed(x).sigmoid()
        return cls_score, point_pred
    
    def _get_target_single(self, pos_inds, neg_inds, pos_points, neg_points,
                           pos_gt_points, pos_gt_labels, cfg):
        num_pos = pos_points.size(0)
        num_neg = neg_points.size(0)
        num_samples = num_pos + num_neg
        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_points.new_full((num_samples, ),
                                     self.num_classes,
                                     dtype=torch.long)
        label_weights = pos_points.new_zeros(num_samples).type_as(pos_gt_labels)
        point_targets = pos_points.new_zeros(num_samples, 2).type_as(pos_gt_points)
        point_weights = pos_points.new_zeros(num_samples, 2).type_as(pos_gt_points)
        if num_pos > 0:
            labels[pos_inds] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[pos_inds] = pos_weight
            pos_point_targets = pos_gt_points
            point_targets[pos_inds, :] = pos_point_targets
            point_weights[pos_inds, :] = 1
        if num_neg > 0:
            label_weights[neg_inds] = 1.0

        return labels, label_weights, point_targets, point_weights

    def get_targets(self,
                    sampling_results,
                    gt_points,
                    gt_labels,
                    rcnn_train_cfg,
                    concat=True):
        pos_inds_list = [res.pos_inds for res in sampling_results]
        neg_inds_list = [res.neg_inds for res in sampling_results]
        pos_points_list = [res.pos_points for res in sampling_results]
        neg_points_list = [res.neg_points for res in sampling_results]
        pos_gt_points_list = [res.pos_gt_points for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        labels, label_weights, point_targets, point_weights = multi_apply(
            self._get_target_single,
            pos_inds_list,
            neg_inds_list,
            pos_points_list,
            neg_points_list,
            pos_gt_points_list,
            pos_gt_labels_list,
            cfg=rcnn_train_cfg)
        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            point_targets = torch.cat(point_targets, 0)
            point_weights = torch.cat(point_weights, 0)
        return labels, label_weights, point_targets, point_weights

    @force_fp32(apply_to=('cls_score', 'point_pred'))
    def loss(self,
             cls_score,
             point_pred,
             labels,
             label_weights,
             point_targets,
             point_weights,
             imgs_whwh=None,
             reduction_override=None,
             **kwargs):
        losses = dict()
        bg_class_ind = self.num_classes 
        # note in spare rcnn num_gt == num_pos
        pos_inds = (labels >= 0) & (labels < bg_class_ind)
        num_pos = pos_inds.sum().float()
        avg_factor = reduce_mean(num_pos)
        cls_score = cls_score.reshape(-1, self.num_classes)
        if cls_score is not None:
            if cls_score.numel() > 0:
                losses['loss_point_cls'] = self.loss_cls(
                    cls_score.float(),
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                losses['pos_acc'] = accuracy(cls_score[pos_inds],
                                             labels[pos_inds])
        if point_pred is not None:
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                pos_point_pred = point_pred.reshape(-1, 2)[pos_inds.type(torch.bool)]
                imgs_whwh = imgs_whwh.reshape(-1, 2)[pos_inds.type(torch.bool)]
                losses['loss_point'] = self.loss_point(
                    pos_point_pred,
                    point_targets[pos_inds.type(torch.bool)] / imgs_whwh,
                    point_weights[pos_inds.type(torch.bool)],
                    avg_factor=avg_factor)
            else:
                losses['loss_point'] = point_pred.sum() * 0
        return losses