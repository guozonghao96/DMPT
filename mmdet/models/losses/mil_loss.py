# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss

def _expand_onehot_labels(labels, label_weights, label_channels, ignore_index):
    """Expand onehot labels to match the size of prediction."""
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    valid_mask = (labels >= 0) & (labels != ignore_index)
    inds = torch.nonzero(
        valid_mask & (labels < label_channels), as_tuple=False)

    if inds.numel() > 0:
        bin_labels[inds, labels[inds]] = 1

    valid_mask = valid_mask.view(-1, 1).expand(labels.size(0),
                                               label_channels).float()
    if label_weights is None:
        bin_label_weights = valid_mask
    else:
        bin_label_weights = label_weights.view(-1, 1).repeat(1, label_channels)
        bin_label_weights *= valid_mask

    return bin_labels, bin_label_weights, valid_mask


def binary_cross_entropy(pred,
                          label,
                          weight=None,
                          ignore_index=-100,
                          **kwargs):   
    if pred.dim() != label.dim():
        label, weight, _ = _expand_onehot_labels(
            label, weight, pred.size(-1), ignore_index)
    pred = pred.clamp(1e-6, 1 - 1e-6)
    label = label.clamp(0, 1)
    loss = -label * torch.log(pred) - (1 - label) * torch.log(1 - pred)
    return loss

def cross_entropy(pred,
                  label,
                  weight=None,
                  ignore_index=-100,
                  **kwargs):
    if pred.dim() != label.dim():
        label, weight, _ = _expand_onehot_labels(
            label, weight, pred.size(-1), ignore_index)
    pred = pred.clamp(1e-6, 1 - 1e-6)
    label = label.clamp(0, 1)
    loss = -label * torch.log(pred)
    return loss


def binary_cross_entropy_(pred,
                          label,
                          weight=None,
                          ignore_index=-100,
                          **kwargs):   
    if pred.dim() != label.dim():
        label, weight, _ = _expand_onehot_labels(
            label, weight, pred.size(-1), ignore_index)
    pred = pred.clamp(1e-6, 1 - 1e-6)
    label = label.clamp(0, 1)
    loss = -label * torch.log(pred) - (1 - label) * torch.log(1 - pred)
    return loss

# def gfocal_loss(pred,
#                 label,
#                 weight=None,
#                 ignore_index=-100,
#                 **kwargs):
#     if pred.dim() != label.dim():
#         label, weight, _ = _expand_onehot_labels(
#             label, weight, pred.size(-1), ignore_index)
#     pred = pred.clamp(1e-6, 1 - 1e-6)
#     label = label.clamp(0, 1)
#     loss1 = (pred - label) ** 2
#     loss2 = -label * torch.log(pred) - (1 - label) * torch.log(1 - pred)
#     loss = loss1 * loss2
#     return loss

# def gfocal_loss_(pred,
#                 label,
#                 **kwargs): 
#     pred = pred.clamp(1e-6, 1 - 1e-6)
#     label = label.clamp(0, 1)
#     loss1 = (pred - label) ** 2
#     loss2 = -label * torch.log(pred) - (1 - label) * torch.log(1 - pred)
#     loss = loss1 * loss2
#     return loss.sum(-1)

@LOSSES.register_module()
class MILLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 ignore_index=None,
                 loss_weight=1.0,
                 avg_non_ignore=False):
        super(MILLoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.ignore_index = ignore_index
        self.avg_non_ignore = avg_non_ignore
        if ((ignore_index is not None) and not self.avg_non_ignore
                and self.reduction == 'mean'):
            warnings.warn(
                'Default ``avg_non_ignore`` is False, if you would like to '
                'ignore the certain label and average loss over non-ignore '
                'labels, which is the same with PyTorch official '
                'cross_entropy, set ``avg_non_ignore=True``.')

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        else:
            self.cls_criterion = cross_entropy
#             assert False, 'no implemention'

    def extra_repr(self):
        """Extra repr."""
        s = f'avg_non_ignore={self.avg_non_ignore}'
        return s

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if ignore_index is None:
            ignore_index = self.ignore_index

        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(
                self.class_weight, device=cls_score.device)
        else:
            class_weight = None
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            ignore_index=ignore_index,
            avg_non_ignore=self.avg_non_ignore,
            **kwargs)
        return loss_cls
    