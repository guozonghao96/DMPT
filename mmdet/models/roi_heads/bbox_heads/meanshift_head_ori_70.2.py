import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16, force_fp32
from torch.nn.modules.utils import _pair
from mmdet.core.bbox.iou_calculators import bbox_overlaps

from mmdet.core import build_bbox_coder
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.losses import accuracy
from models.utils import trunc_normal_
from mmdet.core import bbox_overlaps
from mmdet.core import bbox2roi
from mmdet.core import (anchor_inside_flags, build_assigner, build_bbox_coder,
                        build_prior_generator, build_sampler, images_to_levels,
                        multi_apply, unmap)
from mmdet.models.builder import HEADS, build_head, build_roi_extractor
from einops import rearrange
from mmdet.models.losses.mil_loss import binary_cross_entropy_

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

from ...losses import weight_reduce_loss
    
from mmdet.models.losses import accuracy

@HEADS.register_module()
class MILMeanShiftHead(nn.Module):
    def __init__(self,
                in_channels=256,
                hidden_channels=1024,
                pooling_type='roi',
                roi_size=7,
                num_classes=20,
                bags_topk=50,
                topk_merge=1,
                pre_anchor_topk=6, 
                instance_extractor=dict(
                    type='SingleRoIExtractor',
                    roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
                    out_channels=256,
                    featmap_strides=[16]),
                anchor_generator=dict(
                    type='AnchorGenerator',
                    octave_base_scale=8,
                    scales_per_octave=1,
                    ratios=[1.0],
                    strides=[8, 16, 32, 64, 128]),
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                loss_mil=dict(
                    type='MILLoss', 
                    use_sigmoid=True, # BCE loss 
                    reduction='mean',
                    loss_weight=1.0),
                loss_reg=dict(type='GIoULoss', loss_weight=1.0),
                loss_shift_reg=dict(type='GIoULoss', loss_weight=1.0),
                ):
        super(MILMeanShiftHead, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.bags_topk = bags_topk
        self.topk_merge = topk_merge
        self.num_classes = num_classes
        
        self.instance_extractor = build_roi_extractor(instance_extractor)
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.prior_generator = build_prior_generator(anchor_generator)
        self.loss_mil = build_loss(loss_mil)
        self.loss_reg = build_loss(loss_reg)
        self.loss_shift_reg = build_loss(loss_shift_reg)
        
        self.hidden_channels = hidden_channels
        self.roi_size = roi_size
        self.pre_anchor_topk = pre_anchor_topk
        
        self.fc1 = nn.Linear(in_channels * roi_size ** 2, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, hidden_channels)
            
        self.score1 = nn.Linear(hidden_channels, num_classes)
        self.score2 = nn.Linear(hidden_channels, num_classes)
        self.reg = nn.Linear(hidden_channels, 4)
        
        self.fusion_fc = nn.Linear(hidden_channels, hidden_channels)
        self.shift_reg = nn.Linear(hidden_channels, 4)
        # self.dynamic_weight = nn.Parameter(torch.ones((1, hidden_channels)), requires_grad=True)
        
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
        
    def forward_train(self,
                      x,  # fpn feature (5, ) (b, c, h, w)
                      vit_feat, # 直接用最后一层特征就行
                      img_metas,
                      gt_bboxes, # mil_pseudo_bboxes
                      gt_labels, # gt_labels
                      semantic_scores, # mil 2 softmax scores (for weight loss)
                      stage=0,
                      bag_proposals_list=None,
                     ):
        if stage == 0:
            # 第一阶段的时候，需要用anchor的box生成roi特征。
            featmap_sizes = [featmap.size()[-2:] for featmap in x]
            assert len(featmap_sizes) == self.prior_generator.num_levels
            device = x[0].device
            anchor_list, _ = self.get_anchors(
                featmap_sizes, img_metas, device=device)
            anchors = [torch.cat(anchor) for anchor in anchor_list]
            # 先选择出后续需要迭代的框集合，并且选择topk用来当作正例的pos proposal
            num_img = x[0].size(0)
            bag_proposals_list = []
            num_bag_topk = []
#             pos_proposals_list = []
#             num_gts_topk = []
            for _, (anchors_, gt_labels_, gt_bboxes_) in enumerate(
                        zip(anchors, gt_labels, gt_bboxes)):
                match_quality_matrix = bbox_overlaps(gt_bboxes_, anchors_)
                _, matched = torch.topk(
                    match_quality_matrix,
                    self.bags_topk,
                    dim=1,
                    sorted=False)
                del match_quality_matrix
                bag_matched_anchors = anchors_[matched] # num_gt, topk, 4
                bag_proposals_list.append(bag_matched_anchors)
                num_bag_topk.append(bag_matched_anchors.size(0) * self.bags_topk)
                
#                 pos_matched_anchors = bag_matched_anchors[:, :self.pre_anchor_topk]
#                 pos_proposals_list.append(pos_matched_anchors)
#                 num_gts_topk.append(pos_matched_anchors.size(0) * self.pre_anchor_topk)

            # 获得匹配后的anchor, 直接做roi，先进行回归
            rois = bbox2roi([proposals.reshape(-1, 4) for proposals in bag_proposals_list]) 
            vit_feat = vit_feat.permute(0, 2, 1).reshape(num_img, -1, *featmap_sizes[2]).contiguous()
            instance_feats = self.instance_extractor(
                    [vit_feat][:self.instance_extractor.num_inputs], rois) # [num_gt1 * topk, 256, 7 * 7], [num_gt2 * topk, 256, 7 * 7]
            bbox_regs = self.forward_reg(instance_feats)
            bbox_regs_list = list(torch.split(bbox_regs, num_bag_topk, dim=0))
            
            losses_reg = 0
            losses_mil = 0
            losses_shift_reg = 0
#             shifted_bboxes = []
#             pred_anchors = []
#             soft_bboxes = []
#             pred_proposal_feats = []
#             cls_max_bboxes = []
#             mil_max_bboxes = []
            
            dy_weights = []
            proposals_list = []
            mean_shift_acc = []
            mean_gt_bboxes = []
            shifted_bboxes = []
#             mil_top2_merge_bboxes = []
#             mil_top3_merge_bboxes = []
#             mil_top4_merge_bboxes = []
#             mil_top5_merge_bboxes = []
#             mil_top6_merge_bboxes = []
#             mil_top7_merge_bboxes = []
#             mil_top8_merge_bboxes = []

            for _, (anchors_, bbox_regs_, gt_labels_, gt_bboxes_, dynamic_weights_) in enumerate(
                        zip(bag_proposals_list, bbox_regs_list, gt_labels, gt_bboxes, semantic_scores)):
                # 做decode,并进行回归,用dynamic_weights_加权
                num_gt = anchors_.size(0)
                bbox_regs_ = bbox_regs_.reshape(num_gt, self.bags_topk, 4)
                pred_boxes = self.bbox_coder.decode(anchors_, bbox_regs_)
                proposals_list.append(pred_boxes)
                # 这里只用 pre_anchor_top来找正例的目标
                pos_pred_boxes = pred_boxes[:, :self.pre_anchor_topk]
                # 只计算正例的loss
                _loss_reg = self.loss_reg(pos_pred_boxes, 
                                          gt_bboxes_.reshape(num_gt, 1, 4).repeat(1, self.pre_anchor_topk, 1),
                                          reduction_override='none') * dynamic_weights_ 
                losses_reg += _loss_reg.sum() / max(1, num_gt * self.pre_anchor_topk)

                # 这些预测的pos_pred_boxes当作新的box来进行wsddn
                pos_pred_boxes_ = pos_pred_boxes.clone().detach() # num_gt, self.pre_anchor_topk, 4
#                 pred_anchors.append(pred_boxes_)
                rois = bbox2roi([pos_pred_boxes_.reshape(-1, 4)])
                instance_feats = self.instance_extractor(
                        [vit_feat][:self.instance_extractor.num_inputs], rois) # [num_gt1 * topk, 256, 7,  7]
#                 pred_proposal_feats.append(instance_feats)
                bag_score, pred_roi_feats_bf_cls = self.forward_mil(instance_feats) # num_gt1, topk, 20,   num_gt, topk, 1024
                # 计算wsddn loss
                _loss_mil = self.loss_mil(bag_score.sum(1), gt_labels_) * dynamic_weights_
                losses_mil += _loss_mil.sum() / max(1, num_gt)
                mil1_pos_acc = accuracy(bag_score.sum(1), gt_labels_)
                mean_shift_acc.append(mil1_pos_acc.reshape(-1))
                
                # 选择topk的分数来加权空间位置
                bag_score_gt_label = torch.gather(
                    bag_score.detach(), 
                    dim=-1, 
                    index=gt_labels_.reshape(num_gt, 1, 1).repeat(1, self.pre_anchor_topk, 1)
                ) # num_gt, topk, 1 
                bag_score_gt_label, topk_index = torch.topk(bag_score_gt_label, k=self.topk_merge, dim=1)
                selected_bboxes = torch.gather(pos_pred_boxes_, dim=1, index=topk_index.repeat(1, 1, 4))
                soft_weight = bag_score_gt_label / bag_score_gt_label.sum(1).unsqueeze(1) # num_gt, topk, 1
                mean_bboxes = (soft_weight * selected_bboxes).sum(1) # num_gt, 4
                mean_gt_bboxes.append(mean_bboxes)
                dy_weights_, _ = torch.topk(bag_score_gt_label, k=1, dim=1)
                dy_weights.append(dy_weights_.reshape(-1, 1))
                
                # pseudo_gt_bboxes 的 roi feature
                rois = bbox2roi([gt_bboxes_.reshape(-1, 4)])
                pseudo_gt_feat = self.instance_extractor(
                        [vit_feat][:self.instance_extractor.num_inputs], rois) # [num_gt1, 256, 7,  7]
                pred_roi_feats_bf_cls = torch.gather(pred_roi_feats_bf_cls, dim=1, index=topk_index.repeat(1, 1, self.hidden_channels))
                gt_bbox_shifts_ = self.forward_shift_reg(pseudo_gt_feat, pred_roi_feats_bf_cls, soft_weight)
                shifted_bboxes_ = self.bbox_coder.decode(gt_bboxes_, gt_bbox_shifts_)
                _loss_shift_reg = self.loss_reg(shifted_bboxes_, 
                                          mean_bboxes.reshape(num_gt, 4),
                                          reduction_override='none') * dynamic_weights_ 
                losses_shift_reg += _loss_shift_reg.sum() / max(1, num_gt)
                shifted_bboxes.append(shifted_bboxes_.detach())
                
            losses = {
                's{:d}.reg_loss'.format(stage): losses_reg,
                's{:d}.mil_loss'.format(stage): losses_mil,
                's{:d}.mil_acc'.format(stage): torch.cat(mean_shift_acc).mean(),
                's{:d}.shift_reg_loss'.format(stage): losses_shift_reg,
                
            }
#             losses = dict(
#                 meanshift_reg_loss=losses_reg,
#                 meanshift_mil_loss=losses_mil,
#                 mean_shift_acc=torch.cat(mean_shift_acc).mean(),
#                 meanshift_shift_reg_loss=losses_shift_reg
#             )
            results = dict(
                mean_gt_bboxes=mean_gt_bboxes,
                dy_weights=dy_weights,
                proposals_list=proposals_list,
                shifted_bboxes=shifted_bboxes
            )
#                 pred_anchors=pred_anchors,
    #             soft_bboxes=soft_bboxes,
    #             anchor_match_ious=anchor_match_ious,
    #             match_gt_ious=match_gt_ious,
#                 pred_proposal_feats=pred_proposal_feats,
#                 mil_max_bboxes=mil_max_bboxes,
#                 cls_max_bboxes=cls_max_bboxes,
#                 mil_top2_merge_bboxes=mil_top2_merge_bboxes,
#                 mil_top3_merge_bboxes=mil_top3_merge_bboxes,
#                 mil_top4_merge_bboxes=mil_top4_merge_bboxes,
#                 mil_top5_merge_bboxes=mil_top5_merge_bboxes,
#                 mil_top6_merge_bboxes=mil_top6_merge_bboxes,
#                 mil_top7_merge_bboxes=mil_top7_merge_bboxes,
#                 mil_top8_merge_bboxes=mil_top8_merge_bboxes,
#             )
            return losses, results
            
#                 # top1
#                 bag_score_gt_label = torch.gather(
#                     bag_score.detach(), 
#                     dim=-1, 
#                     index=gt_labels_.reshape(num_gt, 1, 1).repeat(1, self.pre_anchor_topk, 1)
#                 ) # num_gt, topk, 1 
#                 bag_score_gt_label, topk_index = torch.topk(bag_score_gt_label, k=self.topk_merge, dim=1)
#                 selected_bboxes = torch.gather(pred_boxes_, dim=1, index=topk_index.repeat(1, 1, 4))

#                 # 这两个都表示了 对于这个gt_bboxes_的描述
#                 # anchor的roi特征
#                 soft_weight = bag_score_gt_label / bag_score_gt_label.sum(1).unsqueeze(1) # num_gt, topk, 1 
#                 mean_bboxes = (soft_weight * selected_bboxes).sum(1) # num_gt, 4

#                 mil_max_bboxes.append(mean_bboxes)
#                 soft_bboxes.append(mean_bboxes)


#                 # top2 merge
#                 bag_score_gt_label = torch.gather(
#                     bag_score.detach(), 
#                     dim=-1, 
#                     index=gt_labels_.reshape(num_gt, 1, 1).repeat(1, self.pre_anchor_topk, 1)
#                 ) # num_gt, topk, 1 

#                 # 只用tok来选择
#                 bag_score_gt_label, topk_index = torch.topk(bag_score_gt_label, k=2, dim=1)
#                 selected_bboxes = torch.gather(pred_boxes_, dim=1, index=topk_index.repeat(1, 1, 4))

#                 # 这两个都表示了 对于这个gt_bboxes_的描述
#                 # anchor的roi特征
#                 soft_weight = bag_score_gt_label / bag_score_gt_label.sum(1).unsqueeze(1) # num_gt, topk, 1 
#                 mean_bboxes = (soft_weight * selected_bboxes).sum(1) # num_gt, 4
#                 mil_top2_merge_bboxes.append(mean_bboxes)

#                 # top3 merge
#                 bag_score_gt_label = torch.gather(
#                     bag_score.detach(), 
#                     dim=-1, 
#                     index=gt_labels_.reshape(num_gt, 1, 1).repeat(1, self.pre_anchor_topk, 1)
#                 ) # num_gt, topk, 1 

#                 # 只用tok来选择
#                 bag_score_gt_label, topk_index = torch.topk(bag_score_gt_label, k=3, dim=1)
#                 selected_bboxes = torch.gather(pred_boxes_, dim=1, index=topk_index.repeat(1, 1, 4))

#                 # 这两个都表示了 对于这个gt_bboxes_的描述
#                 # anchor的roi特征
#                 soft_weight = bag_score_gt_label / bag_score_gt_label.sum(1).unsqueeze(1) # num_gt, topk, 1 
#                 mean_bboxes = (soft_weight * selected_bboxes).sum(1) # num_gt, 4
#                 mil_top3_merge_bboxes.append(mean_bboxes)

#                 # top4 merge
#                 bag_score_gt_label = torch.gather(
#                     bag_score.detach(), 
#                     dim=-1, 
#                     index=gt_labels_.reshape(num_gt, 1, 1).repeat(1, self.pre_anchor_topk, 1)
#                 ) # num_gt, topk, 1 

#                 # 只用tok来选择
#                 bag_score_gt_label, topk_index = torch.topk(bag_score_gt_label, k=4, dim=1)
#                 selected_bboxes = torch.gather(pred_boxes_, dim=1, index=topk_index.repeat(1, 1, 4))

#                 # 这两个都表示了 对于这个gt_bboxes_的描述
#                 # anchor的roi特征
#                 soft_weight = bag_score_gt_label / bag_score_gt_label.sum(1).unsqueeze(1) # num_gt, topk, 1 
#                 mean_bboxes = (soft_weight * selected_bboxes).sum(1) # num_gt, 4
#                 mil_top4_merge_bboxes.append(mean_bboxes)


#                 # top5 merge
#                 bag_score_gt_label = torch.gather(
#                     bag_score.detach(), 
#                     dim=-1, 
#                     index=gt_labels_.reshape(num_gt, 1, 1).repeat(1, self.pre_anchor_topk, 1)
#                 ) # num_gt, topk, 1 

#                 # 只用tok来选择
#                 bag_score_gt_label, topk_index = torch.topk(bag_score_gt_label, k=5, dim=1)
#                 selected_bboxes = torch.gather(pred_boxes_, dim=1, index=topk_index.repeat(1, 1, 4))

#                 # 这两个都表示了 对于这个gt_bboxes_的描述
#                 # anchor的roi特征
#                 soft_weight = bag_score_gt_label / bag_score_gt_label.sum(1).unsqueeze(1) # num_gt, topk, 1 
#                 mean_bboxes = (soft_weight * selected_bboxes).sum(1) # num_gt, 4
#                 mil_top5_merge_bboxes.append(mean_bboxes)

#                 # top6 merge
#                 bag_score_gt_label = torch.gather(
#                     bag_score.detach(), 
#                     dim=-1, 
#                     index=gt_labels_.reshape(num_gt, 1, 1).repeat(1, self.pre_anchor_topk, 1)
#                 ) # num_gt, topk, 1 

#                 # 只用tok来选择
#                 bag_score_gt_label, topk_index = torch.topk(bag_score_gt_label, k=6, dim=1)
#                 selected_bboxes = torch.gather(pred_boxes_, dim=1, index=topk_index.repeat(1, 1, 4))

#                 # 这两个都表示了 对于这个gt_bboxes_的描述
#                 # anchor的roi特征
#                 soft_weight = bag_score_gt_label / bag_score_gt_label.sum(1).unsqueeze(1) # num_gt, topk, 1 
#                 mean_bboxes = (soft_weight * selected_bboxes).sum(1) # num_gt, 4
#                 mil_top6_merge_bboxes.append(mean_bboxes)            

#                 # top7 merge
#                 bag_score_gt_label = torch.gather(
#                     bag_score.detach(), 
#                     dim=-1, 
#                     index=gt_labels_.reshape(num_gt, 1, 1).repeat(1, self.pre_anchor_topk, 1)
#                 ) # num_gt, topk, 1 

#                 # 只用tok来选择
#                 bag_score_gt_label, topk_index = torch.topk(bag_score_gt_label, k=7, dim=1)
#                 selected_bboxes = torch.gather(pred_boxes_, dim=1, index=topk_index.repeat(1, 1, 4))

#                 # 这两个都表示了 对于这个gt_bboxes_的描述
#                 # anchor的roi特征
#                 soft_weight = bag_score_gt_label / bag_score_gt_label.sum(1).unsqueeze(1) # num_gt, topk, 1 
#                 mean_bboxes = (soft_weight * selected_bboxes).sum(1) # num_gt, 4
#                 mil_top7_merge_bboxes.append(mean_bboxes)            


#                 # top8 merge
#                 bag_score_gt_label = torch.gather(
#                     bag_score.detach(), 
#                     dim=-1, 
#                     index=gt_labels_.reshape(num_gt, 1, 1).repeat(1, self.pre_anchor_topk, 1)
#                 ) # num_gt, topk, 1 

#                 # 只用tok来选择
#                 bag_score_gt_label, topk_index = torch.topk(bag_score_gt_label, k=8, dim=1)
#                 selected_bboxes = torch.gather(pred_boxes_, dim=1, index=topk_index.repeat(1, 1, 4))
#                 # 这两个都表示了 对于这个gt_bboxes_的描述
#                 # anchor的roi特征


#                 soft_weight = bag_score_gt_label / bag_score_gt_label.sum(1).unsqueeze(1) # num_gt, topk, 1 
#                 mean_bboxes = (soft_weight * selected_bboxes).sum(1) # num_gt, 4
#                 mil_top8_merge_bboxes.append(mean_bboxes)       

#                 dy_weights_, _ = torch.topk(bag_score_gt_label, k=1, dim=1)
#                 print(dy_weights_.size(), dynamic_weights_.size())
#                 exit()


    #             # mean_shift self_refine
    #             top8_soft_weight = soft_weight.clone()
    #             top_instance_feats = torch.gather(instance_feats.reshape(num_gt, self.pre_anchor_topk, -1), dim=1, index=topk_index.repeat(1, 1, self.in_channels * self.roi_size ** 2))
    #             mean_roi_feat = (top_instance_feats * top8_soft_weight).sum(1)
    #                                 # [num_gt1, topk, 256 * 7 * 7] * softmax-> # [num_gt1, 256 * 7 * 7]
    # #             # gt的roi特征
    #             rois = bbox2roi([gt_bboxes_.reshape(-1, 4)])
    #             pseudo_gt_feat = self.instance_extractor(
    #                     [vit_feat][:self.instance_extractor.num_inputs], rois).reshape(num_gt, -1)
    #             pred_roi_feats = pred_roi_feats.reshape(num_gt, self.pre_anchor_topk, -1)
    #             pred_roi_feats = torch.gather(pred_roi_feats, dim=1, index=topk_index.repeat(1, 1, self.hidden_channels))
    #             bbox_shifts = self.forward_shift_reg(pseudo_gt_feat, pred_roi_feats, soft_weight)

    #             shifted_bboxes_ = self.bbox_coder.decode(gt_bboxes_, bbox_shifts) 
    #             _loss_shift_reg = self.loss_shift_reg(shifted_bboxes_, 
    #                                                   mean_bboxes,
    #                                                   reduction_override='none') * dynamic_weights_  #
    #             losses_shift_reg += _loss_shift_reg.sum() / max(1, num_gt)

    #             shifted_bboxes.append(shifted_bboxes_.detach())




    #         anchor_match_ious = bbox_overlaps(
    #             torch.cat(proposals_list).reshape(-1, 4),
    #             torch.cat(gt_bboxes).reshape(-1, 1, 4).repeat(1, self.pre_anchor_topk, 1).reshape(-1, 4),
    #             is_aligned=True
    #         )
    #         match_gt_ious = bbox_overlaps(
    #             torch.cat(pred_anchors).reshape(-1, 4),
    #             torch.cat(gt_bboxes).reshape(-1, 1, 4).repeat(1, self.pre_anchor_topk, 1).reshape(-1, 4),
    #             is_aligned=True
    #         )


        else:
            featmap_sizes = [featmap.size()[-2:] for featmap in x]
            num_img = x[0].size(0)
            num_bag_topk = [proposals.size(0) * self.bags_topk for proposals in bag_proposals_list]
            rois = bbox2roi([proposals.reshape(-1, 4) for proposals in bag_proposals_list]) 
            vit_feat = vit_feat.permute(0, 2, 1).reshape(num_img, -1, *featmap_sizes[2]).contiguous()
            instance_feats = self.instance_extractor(
                    [vit_feat][:self.instance_extractor.num_inputs], rois) # [num_gt1 * topk, 256, 7 * 7], [num_gt2 * topk, 256, 7 * 7]
            bbox_regs = self.forward_reg(instance_feats)
            bbox_regs_list = list(torch.split(bbox_regs, num_bag_topk, dim=0))
            
            losses_reg = 0
            losses_mil = 0
            
            dy_weights = []
            proposals_list = []
            mean_shift_acc = []
            mean_gt_bboxes = []

            for _, (anchors_, bbox_regs_, gt_labels_, gt_bboxes_, dynamic_weights_) in enumerate(
                        zip(bag_proposals_list, bbox_regs_list, gt_labels, gt_bboxes, semantic_scores)):
                # 做decode,并进行回归,用dynamic_weights_加权
                num_gt = anchors_.size(0)
                bbox_regs_ = bbox_regs_.reshape(num_gt, self.bags_topk, 4)
                pred_boxes = self.bbox_coder.decode(anchors_, bbox_regs_)
                proposals_list.append(pred_boxes)
                # 这里只用 pre_anchor_top来找正例的目标
                pos_pred_boxes = pred_boxes[:, :self.pre_anchor_topk]
                # 只计算正例的loss
                _loss_reg = self.loss_reg(pos_pred_boxes, 
                                          gt_bboxes_.reshape(num_gt, 1, 4).repeat(1, self.pre_anchor_topk, 1),
                                          reduction_override='none') * dynamic_weights_ 
                losses_reg += _loss_reg.sum() / max(1, num_gt * self.pre_anchor_topk)

                # 这些预测的pos_pred_boxes当作新的box来进行wsddn
                pos_pred_boxes_ = pos_pred_boxes.clone().detach() # num_gt, self.pre_anchor_topk, 4
#                 pred_anchors.append(pred_boxes_)
                rois = bbox2roi([pos_pred_boxes_.reshape(-1, 4)])
                instance_feats = self.instance_extractor(
                        [vit_feat][:self.instance_extractor.num_inputs], rois) # [num_gt1 * topk, 256, 7,  7]
#                 pred_proposal_feats.append(instance_feats)
                bag_score = self.forward_mil(instance_feats) # num_gt1, topk, 20
                # 计算wsddn loss
                _loss_mil = self.loss_mil(bag_score.sum(1), gt_labels_) * dynamic_weights_
                losses_mil += _loss_mil.sum() / max(1, num_gt)
                mil1_pos_acc = accuracy(bag_score.sum(1), gt_labels_)
                mean_shift_acc.append(mil1_pos_acc.reshape(-1))
                
                # 选择topk的分数来加权空间位置
                bag_score_gt_label = torch.gather(
                    bag_score.detach(), 
                    dim=-1, 
                    index=gt_labels_.reshape(num_gt, 1, 1).repeat(1, self.pre_anchor_topk, 1)
                ) # num_gt, topk, 1 
                bag_score_gt_label, topk_index = torch.topk(bag_score_gt_label, k=self.topk_merge, dim=1)
                selected_bboxes = torch.gather(pos_pred_boxes_, dim=1, index=topk_index.repeat(1, 1, 4))
                soft_weight = bag_score_gt_label / bag_score_gt_label.sum(1).unsqueeze(1) # num_gt, topk, 1
                mean_bboxes = (soft_weight * selected_bboxes).sum(1) # num_gt, 4
                mean_gt_bboxes.append(mean_bboxes)
                dy_weights_, _ = torch.topk(bag_score_gt_label, k=1, dim=1)
                dy_weights.append(dy_weights_.reshape(-1, 1))
            
            losses = {
                's{:d}.reg_loss'.format(stage): losses_reg,
                's{:d}.mil_loss'.format(stage): losses_mil,
                's{:d}.mil_acc'.format(stage): torch.cat(mean_shift_acc).mean(),
            }
#             losses = dict(
#                 meanshift_reg_loss=losses_reg,
#                 meanshift_mil_loss=losses_mil,
#                 mean_shift_acc=torch.cat(mean_shift_acc).mean(),
#                 meanshift_shift_reg_loss=losses_shift_reg
#             )
            results = dict(
                mean_gt_bboxes=mean_gt_bboxes,
                dy_weights=dy_weights,
                proposals_list=proposals_list,
            )
            return losses, results
                
                
    
    def forward_reg(self, x):
        x = x.reshape(-1, self.in_channels * self.roi_size ** 2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # dual-stream 
        bbox_regs = self.reg(x)
        return bbox_regs        
        
    def forward_mil(self, x):
        x = x.reshape(-1, self.pre_anchor_topk, self.in_channels * self.roi_size ** 2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # dual-stream 
        score1 = self.score1(x).softmax(-1) # cls branch 
        score2 = self.score2(x).softmax(-2) # proposal branch
        bag_score = score1 * score2 
        return bag_score, x #, x, score1
    
    def forward_shift_reg(self, pseudo_gt_feat, pred_roi_feats, soft_weight):
        x = pseudo_gt_feat
        x = x.reshape(-1, self.in_channels * self.roi_size ** 2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # 动态融合特征
        fusion_feat = (pred_roi_feats * soft_weight).sum(1) + x
        fusion_feat = self.fusion_fc(fusion_feat)
        bbox_shifts = self.shift_reg(fusion_feat)
        return bbox_shifts
    
    def get_anchors(self, featmap_sizes, img_metas, device='cuda'):
        num_imgs = len(img_metas)
        multi_level_anchors = self.prior_generator.grid_priors(
            featmap_sizes, device=device)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = self.prior_generator.valid_flags(
                featmap_sizes, img_meta['pad_shape'], device)
            valid_flag_list.append(multi_level_flags)
        return anchor_list, valid_flag_list
        