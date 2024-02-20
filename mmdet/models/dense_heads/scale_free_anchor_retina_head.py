# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from mmdet.core import (anchor_inside_flags, build_assigner, build_bbox_coder,
                        build_prior_generator, build_sampler, images_to_levels,
                        multi_apply, unmap)
from mmcv.runner import force_fp32
from mmdet.core import bbox_overlaps
from ..builder import HEADS
from .retina_head import RetinaHead
from ..builder import HEADS, build_loss
import torch.nn as nn
EPS = 1e-12
from mmdet.core.bbox.iou_calculators import bbox_overlaps


@HEADS.register_module()
class ScaleFreeAnchorRetinaHead(RetinaHead):
    """FreeAnchor RetinaHead used in https://arxiv.org/abs/1909.02466.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        stacked_convs (int): Number of conv layers in cls and reg tower.
            Default: 4.
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: norm_cfg=dict(type='GN', num_groups=32,
            requires_grad=True).
        pre_anchor_topk (int): Number of boxes that be token in each bag.
        bbox_thr (float): The threshold of the saturated linear function. It is
            usually the same with the IoU threshold used in NMS.
        gamma (float): Gamma parameter in focal loss.
        alpha (float): Alpha parameter in focal loss.
    """  # noqa: W605

    def __init__(self,
                 num_classes,
                 in_channels,
                 score_thr=0.05,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 pre_anchor_topk=50,
                 bbox_thr=0.6,
                 gamma=2.0,
                 alpha=0.5,
                 num_proposals_per_gt=4,
                 loss_mil=None,
                 **kwargs):
        super(ScaleFreeAnchorRetinaHead,
              self).__init__(num_classes, in_channels, stacked_convs, conv_cfg,
                             norm_cfg, **kwargs)

        self.pre_anchor_topk = pre_anchor_topk
        self.bbox_thr = bbox_thr
        self.gamma = gamma
        self.alpha = alpha
        self.num_proposals_per_gt = num_proposals_per_gt
        self.score_thr = score_thr
        # self.conv_cls_ = nn.Conv2d(self.in_channels, 
        #                            self.num_base_priors * self.cls_out_channels,
        #                            1)
#         self.conv_cls__ = nn.Conv2d(self.in_channels, 
#                                    self.num_base_priors * self.cls_out_channels,
#                                    1)
        self.loss_mil = build_loss(loss_mil)
        
        
        
    def forward_single(self, x):
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        # cls_score_ = self.conv_cls_(cls_feat)
#         cls_score__ = self.conv_cls__(cls_feat)
        return cls_score, bbox_pred, #cls_score_ #, cls_score__

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)
    
    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      semantic_scores=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
#         num_gt, num_scale = gt_bboxes[0].size(0), gt_bboxes[0].size(1)
#         gt_labels_ = [labels.reshape(-1, 1).repeat(1, num_scale) for labels in gt_labels]
#         gt_labels = gt_labels_
        
#         # 把目标都当做gt来用, 然而排列方式是 (num_gt * num_scale, 4)
#         # 在于对应分类分数 (num_gt * num_scale)
#         # 其label为 (num_gt * num_scale)
        
#         overall_gt_labels = []
#         overall_gt_bboxes = []
#         overall_scale_scores = []
        
#         for labels, bboxes, scores in zip(gt_labels, gt_bboxes, semantic_scores):
#             overall_gt_labels.append(labels.reshape(-1))
#             overall_gt_bboxes.append(bboxes.reshape(-1, 4))
#             overall_scale_scores.append(scores.reshape(-1))
#         gt_labels = overall_gt_labels
#         gt_bboxes = overall_gt_bboxes
#         semantic_scores = overall_scale_scores
        
        outs = self(x)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def loss(self,
             cls_scores,
             bbox_preds,
             # cls_scores_,
#              cls_scores__,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels
        device = cls_scores[0].device
        anchor_list, _ = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        anchors = [torch.cat(anchor) for anchor in anchor_list]

        # concatenate each level
        cls_scores = [
            cls.permute(0, 2, 3,
                        1).reshape(cls.size(0), -1, self.cls_out_channels)
            for cls in cls_scores
        ]
        bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(bbox_pred.size(0), -1, 4)
            for bbox_pred in bbox_preds
        ]
        cls_scores = torch.cat(cls_scores, dim=1)
        bbox_preds = torch.cat(bbox_preds, dim=1)

        cls_prob = torch.sigmoid(cls_scores)
        box_prob = []
        num_pos = 0
        
        pseudo_gt_bboxes = []
        pseudo_gt_labels = []
        positive_losses = []
        for _, (anchors_, gt_labels_, gt_bboxes_, cls_prob_,
                bbox_preds_) in enumerate(
                    zip(anchors, gt_labels, gt_bboxes, cls_prob, bbox_preds)):

            with torch.no_grad():
                if len(gt_bboxes_) == 0:
                    image_box_prob = torch.zeros(
                        anchors_.size(0),
                        self.cls_out_channels).type_as(bbox_preds_)
                else:
                    # box_localization: a_{j}^{loc}, shape: [j, 4]
                    pred_boxes = self.bbox_coder.decode(anchors_, bbox_preds_)

                    # object_box_iou: IoU_{ij}^{loc}, shape: [i, j]
                    object_box_iou = bbox_overlaps(gt_bboxes_, pred_boxes)

                    # object_box_prob: P{a_{j} -> b_{i}}, shape: [i, j]
                    t1 = self.bbox_thr
                    t2 = object_box_iou.max(
                        dim=1, keepdim=True).values.clamp(min=t1 + 1e-12)
                    object_box_prob = ((object_box_iou - t1) /
                                       (t2 - t1)).clamp(
                                           min=0, max=1)

                    # object_cls_box_prob: P{a_{j} -> b_{i}}, shape: [i, c, j]
                    num_obj = gt_labels_.size(0)
                    indices = torch.stack([
                        torch.arange(num_obj).type_as(gt_labels_), gt_labels_
                    ],
                                          dim=0)
                    object_cls_box_prob = torch.sparse_coo_tensor(
                        indices, object_box_prob)

                    # image_box_iou: P{a_{j} \in A_{+}}, shape: [c, j]
                    """
                    from "start" to "end" implement:
                    image_box_iou = torch.sparse.max(object_cls_box_prob,
                                                     dim=0).t()

                    """
                    # start
                    box_cls_prob = torch.sparse.sum(
                        object_cls_box_prob, dim=0).to_dense()

                    indices = torch.nonzero(box_cls_prob, as_tuple=False).t_()
                    if indices.numel() == 0:
                        image_box_prob = torch.zeros(
                            anchors_.size(0),
                            self.cls_out_channels).type_as(object_box_prob)
                    else:
                        nonzero_box_prob = torch.where(
                            (gt_labels_.unsqueeze(dim=-1) == indices[0]),
                            object_box_prob[:, indices[1]],
                            torch.tensor([
                                0
                            ]).type_as(object_box_prob)).max(dim=0).values

                        # upmap to shape [j, c]
                        image_box_prob = torch.sparse_coo_tensor(
                            indices.flip([0]),
                            nonzero_box_prob,
                            size=(anchors_.size(0),
                                  self.cls_out_channels)).to_dense()
                    # end

                box_prob.append(image_box_prob)

            # construct bags for objects
            match_quality_matrix = bbox_overlaps(gt_bboxes_, anchors_)
            _, matched = torch.topk(
                match_quality_matrix,
                self.pre_anchor_topk,
                dim=1,
                sorted=False)
            del match_quality_matrix

            # matched_cls_prob: P_{ij}^{cls}
            matched_cls_prob = torch.gather(
                cls_prob_[matched], 2,
                gt_labels_.view(-1, 1, 1).repeat(1, self.pre_anchor_topk,
                                                 1)).squeeze(2)

            # matched_box_prob: P_{ij}^{loc}
            matched_anchors = anchors_[matched]
            matched_object_targets = self.bbox_coder.encode(
                matched_anchors,
                gt_bboxes_.unsqueeze(dim=1).expand_as(matched_anchors))
            loss_bbox = self.loss_bbox(
                bbox_preds_[matched],
                matched_object_targets,
                reduction_override='none').sum(-1)
            matched_box_prob = torch.exp(-loss_bbox)
            
            # 把分数按真正的gt 排放 
            # matched_cls_prob = matched_cls_prob.reshape(-1, self.num_proposals_per_gt, self.pre_anchor_topk).flatten(1)
            # matched_box_prob = matched_box_prob.reshape(-1, self.num_proposals_per_gt, self.pre_anchor_topk).flatten(1)
            # positive_losses: {-log( Mean-max(P_{ij}^{cls} * P_{ij}^{loc}) )}
            num_pos += len(gt_bboxes_)
            pos_loss, bag_scores = self.positive_bag_loss(matched_cls_prob, matched_box_prob)
            positive_losses.append(pos_loss)
            
            # 生成pseudo labels
            
            with torch.no_grad():
                topk_merge = 1
                pseudo_proposals = self.bbox_coder.decode(
                    matched_anchors,
                    bbox_preds_[matched]
                ) 
                num_gt = pseudo_proposals.size(0)
                _, pseudo_index = bag_scores.topk(topk_merge, dim=-1)
                pseudo_index = pseudo_index.reshape(-1, topk_merge, 1).repeat(1, 1, 4)
                pseudo_gt_bboxes_ = torch.gather(pseudo_proposals,
                                                dim=1,
                                                index=pseudo_index)[:, 0] #.reshape(-1, 4)
                # pseudo_gt_labels_ = gt_labels_[0::self.num_proposals_per_gt]
                pseudo_gt_bboxes.append(pseudo_gt_bboxes_)
                pseudo_gt_labels.append(gt_labels_)
            
        positive_loss = torch.cat(positive_losses).sum() / max(1, num_pos)

        # box_prob: P{a_{j} \in A_{+}}
        box_prob = torch.stack(box_prob, dim=0)
        # negative_loss:
        # \sum_{j}{ FL((1 - P{a_{j} \in A_{+}}) * (1 - P_{j}^{bg})) } / n||B||
        negative_loss = self.negative_bag_loss(cls_prob, box_prob).sum() / max(
            1, num_pos * self.pre_anchor_topk)

        # avoid the absence of gradients in regression subnet
        # when no ground-truth in a batch
        if num_pos == 0:
            positive_loss = bbox_preds.sum() * 0

        losses = {
            'positive_bag_loss': positive_loss,
            'negative_bag_loss': negative_loss
        }
        
        scores_results = dict(
            pseudo_gt_bboxes=pseudo_gt_bboxes,
            pseudo_gt_labels=pseudo_gt_labels,
        )
            
        return losses, scores_results

    def positive_bag_loss(self, matched_cls_prob, matched_box_prob):
        """Compute positive bag loss.

        :math:`-log( Mean-max(P_{ij}^{cls} * P_{ij}^{loc}) )`.

        :math:`P_{ij}^{cls}`: matched_cls_prob, classification probability of matched samples.

        :math:`P_{ij}^{loc}`: matched_box_prob, box probability of matched samples.

        Args:
            matched_cls_prob (Tensor): Classification probability of matched
                samples in shape (num_gt, pre_anchor_topk).
            matched_box_prob (Tensor): BBox probability of matched samples,
                in shape (num_gt, pre_anchor_topk).

        Returns:
            Tensor: Positive bag loss in shape (num_gt,).
        """  # noqa: E501, W605
        # bag_prob = Mean-max(matched_prob)
        matched_prob = matched_cls_prob * matched_box_prob
        weight = 1 / torch.clamp(1 - matched_prob, 1e-12, None)
        weight /= weight.sum(dim=1).unsqueeze(dim=-1)
        bag_prob = (weight * matched_prob).sum(dim=1)
        pos_labels = torch.ones_like(bag_prob)
        positive_bag_loss = -self.alpha * torch.log(bag_prob)
        # positive_bag_loss = -pos_labels * torch.log(bag_prob) - (1 - pos_labels) * torch.log(1 - bag_prob)
        return positive_bag_loss, (weight * matched_prob).detach()
        # return self.alpha * F.binary_cross_entropy(
        #     bag_prob, torch.ones_like(bag_prob), reduction='none'), (weight * matched_prob).detach()

    def negative_bag_loss(self, cls_prob, box_prob):
        """Compute negative bag loss.

        :math:`FL((1 - P_{a_{j} \in A_{+}}) * (1 - P_{j}^{bg}))`.

        :math:`P_{a_{j} \in A_{+}}`: Box_probability of matched samples.

        :math:`P_{j}^{bg}`: Classification probability of negative samples.

        Args:
            cls_prob (Tensor): Classification probability, in shape
                (num_img, num_anchors, num_classes).
            box_prob (Tensor): Box probability, in shape
                (num_img, num_anchors, num_classes).

        Returns:
            Tensor: Negative bag loss in shape (num_img, num_anchors, num_classes).
        """  # noqa: E501, W605
        prob = cls_prob * (1 - box_prob)
        # There are some cases when neg_prob = 0.
        # This will cause the neg_prob.log() to be inf without clamp.
        prob = prob.clamp(min=EPS, max=1 - EPS)
        neg_labels = torch.zeros_like(prob)
        # negative_bag_loss = prob**self.gamma * F.binary_cross_entropy(
        #     prob, torch.zeros_like(prob), reduction='none')
        negative_bag_loss = -(1 - self.alpha) * prob ** self.gamma * torch.log(1 - prob)
        # return (1 - self.alpha) * negative_bag_loss
        return negative_bag_loss
