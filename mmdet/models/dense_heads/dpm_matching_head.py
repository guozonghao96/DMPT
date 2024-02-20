# Copyright (c) OpenMMLab. All rights reserved.
import copy
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob, constant_init
from mmcv.runner import force_fp32
from mmcv.cnn import (PLUGIN_LAYERS, Conv2d, ConvModule, caffe2_xavier_init,
                      normal_init, xavier_init)

from mmdet.core import multi_apply
from mmdet.models.utils.transformer import inverse_sigmoid
from ..builder import HEADS
from .point_detr_head import PointDETRHead
from mmdet.models.losses import accuracy
from ..builder import build_roi_extractor
from mmdet.core import (bbox2result, bbox2roi, build_assigner, build_sampler, multi_apply,
                        reduce_mean, bbox_xyxy_to_cxcywh, bbox_cxcywh_to_xyxy)

from mmdet.models.builder import HEADS, build_loss
from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding

from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")
import mmcv
import scipy.interpolate
from mmdet.core import BitmapMasks, PolygonMasks

from models.utils import trunc_normal_
try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None

import itertools as it
import numpy as np
from mmcv.ops import point_sample
import mlflow


@HEADS.register_module()
class DPMMatchingHead(nn.Module):

    def __init__(self,
                 # part based model setting 
                 in_channels=256,
                 hidden_channels=256,
                 out_channels=256,
                 num_classes=21,
                 deformable_cost_weight=1.0,
                 semantic_deform=False,
                 feature_based_deform=False,
                 part_mean_max=False,
                 ablation_cls_weight=False,
                 root_cls_weight=5,
                 dpm_fg_thr=0.0,
                 num_classifier=5,
                 num_neg_sample=5,
                 topk_stride=16,
                 part_points_topk=3,
                 part_proposal_topk=0,
                 iam_pos_thr=0.0,
                 iam_neg_thr=0.0,
                 instance_neg_points_topk=5,
                 neg_loss_weight=0.05,
                 sklearn_matching=False,
                 mask_gt_sets=3, # 1->only attn_shift 2->only dpm 3-> attn_shift+dpm
                 bce_loss=dict(
                     type='MILLoss',
                     use_sigmoid=True, # BCE loss 
                     reduction='mean',
                     loss_weight=0.05),
                 # init point loc setting 
                 random_iam_points=False,
                 num_random_iam_points=10,
                 iam_num_points_init_neg=5, # number of neg and pos sampled points. All number is 2xiam_num_points_init
                 iam_num_points_init_pos=5,
                 iam_thr_pos=0.35, 
                 iam_thr_neg=0.8,
                 iam_refine_times=2, 
                 iam_obj_tau=0.9,
                 pca_dim=64,
                 meanshift_refine_times=5,
                 num_points_for_meanshift=20,
                 map_thr=0.9,
                 num_semantic_points=5,
                 exclude_iam_pos=False,
                 exclude_iam_neg=False,
                 # init point loc setting 
                 **kwargs):
        super(DPMMatchingHead, self).__init__()
        
        assert num_semantic_points == num_classifier
        # instance attention map的参数
        self.random_iam_points = random_iam_points
        self.num_random_iam_points = num_random_iam_points
        self.iam_thr_pos = iam_thr_pos
        self.iam_thr_neg = iam_thr_neg
        self.iam_refine_times = iam_refine_times
        self.iam_obj_tau = iam_obj_tau
        self.iam_num_points_init_pos = iam_num_points_init_pos
        self.iam_num_points_init_neg = iam_num_points_init_neg
        self.pca_dim = pca_dim
        self.meanshift_refine_times = meanshift_refine_times
        self.num_points_for_meanshift = num_points_for_meanshift
        self.point_feat_extractor = None
        self.with_gt_points = False
        self.num_semantic_points = num_semantic_points
        self.num_classifier = num_classifier
        self.num_classes = num_classes
        self.semantic_deform = semantic_deform
        self.map_thr = map_thr
        self.neg_loss_weight = neg_loss_weight
        self.mask_gt_sets = mask_gt_sets
        self.topk_stride = topk_stride
        self.dpm_fg_thr = dpm_fg_thr
        self.feature_based_deform = feature_based_deform
        self.part_mean_max = part_mean_max
        self.exclude_iam_pos = exclude_iam_pos
        self.exclude_iam_neg = exclude_iam_neg
        self.part_proposal_topk = part_proposal_topk
        self.iam_pos_thr = iam_pos_thr
        self.iam_neg_thr = iam_neg_thr
        self.root_cls_weight = root_cls_weight
        self.ablation_cls_weight = ablation_cls_weight
        # 
        self.sklearn_matching = sklearn_matching
        self.num_neg_sample = num_neg_sample
        self.part_points_topk = part_points_topk
        self.instance_neg_points_topk = instance_neg_points_topk
        self.deformable_cost_weight = deformable_cost_weight
        self.bce_loss = build_loss(bce_loss)
        
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        
        self.root_filter = nn.Linear(out_channels, num_classes)
        self.part_filters = nn.ModuleList(
            nn.Linear(out_channels, num_classes) for _ in range(self.num_classifier))
        
        if self.feature_based_deform:
            if not self.semantic_deform:
                self.part_deform = nn.ModuleList(
                    nn.Linear(out_channels, 4) for _ in range(self.num_classifier))
            else:
                self.part_deform = nn.ModuleList(
                    nn.Linear(out_channels, num_classes * 4) for _ in range(self.num_classifier))
                # assert False, 'no implementation'
        else:
            if not self.semantic_deform:
                self.part_deform = nn.Parameter(
                    torch.tensor([0., 0., 1., 1.]).reshape(1, -1).repeat(self.num_classifier, 1), 
                    requires_grad=True)
            else:
                self.part_deform = nn.Parameter(
                    torch.tensor([0., 0., 1., 1.]).reshape(1, 1, -1).repeat(self.num_classifier,
                                                                            self.num_classes, 1), 
                    requires_grad=True)
            
        # self.part_deform = nn.Parameter(
        #     torch.tensor([0., 0., 1., 1.]).reshape(1, -1).repeat(self.num_classifier, 1), 
        #     requires_grad=True)
        # matching proposals
        assert num_semantic_points == num_classifier, 'no implement'
        if not self.sklearn_matching:
            all_matchings = []
            for p in it.permutations(np.arange(num_semantic_points)): 
                all_matchings.append(np.array(p))
            all_matchings = torch.as_tensor(all_matchings).cuda()
            num_matchings = len(all_matchings)
            self.num_matchings = num_matchings
            self.all_row_matchings = all_matchings.clone().reshape(-1, 1, num_semantic_points).repeat(1, num_matchings, 1).reshape(-1, num_semantic_points)
            self.all_col_matchings = all_matchings.clone().reshape(1, -1, num_semantic_points).repeat(num_matchings, 1, 1).reshape(-1, num_semantic_points)
            
    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        if self.feature_based_deform:
            if not self.semantic_deform:
                for m in self.part_deform:
                    m.bias.data = torch.tensor([0, 0, 1, 1], dtype=torch.float)
                    m.bias.requires_grad = False
                    m.weight.data = torch.zeros_like(m.weight)
            else:
                for m in self.part_deform:
                    m.bias.data = torch.tensor([0, 0, 1, 1], 
                                               dtype=torch.float).reshape(1, 4).repeat(self.num_classes, 1).flatten(0)
                    m.bias.requires_grad = False
                    m.weight.data = torch.zeros_like(m.weight)
            
    def forward(self, x):
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        root_scores = self.root_filter(x) #.sigmoid()
        part_scores = []
        for filter_ in self.part_filters:
            scores = filter_(x)
            part_scores.append(scores)
        part_scores = torch.stack(part_scores).permute(1, 0, 2, 3) #.sigmoid()
        deformations = []
        if self.feature_based_deform:
            for deform_ in self.part_deform:
                deformation = deform_(x)
                deformations.append(deformation)
            deformations = torch.stack(deformations).permute(1, 0, 2, 3)
        return root_scores, part_scores, deformations
    
    def forward_train(self,
                      x,  # fpn feature (5, ) (b, c, h, w)
                      vit_feat, # 直接用最后一层特征就行
                      matched_cams,
                      img_metas,
                      gt_bboxes, # mil_pseudo_bboxes
                      gt_labels, # gt_labels
                      gt_points, # gt_points
                      dy_weights,
                      vit_feat_be_norm,
                      imgs_whwh=None,
                      gt_masks=None,
                     ):
        
        # 生成监督label
        device = vit_feat_be_norm.device
        featmap_sizes = [featmap.size()[-2:] for featmap in x]
        patch_h, patch_w = featmap_sizes[2]
        img_h, img_w = patch_h * 16, patch_w * 16
        num_imgs = len(img_metas)
        vit_feat = vit_feat.permute(0, 2, 1).reshape(num_imgs, -1, *featmap_sizes[2]).contiguous()
        vit_feat_be_norm = vit_feat_be_norm.permute(0, 2, 1).reshape(num_imgs, -1, *featmap_sizes[2]).contiguous()
        instance_cos_fg_maps = []
        instance_cos_bg_maps = []
        fg_points = []
        fg_point_feats = []
        bg_point_feats = []
        pseudo_points = []
        pseudo_bin_labels = []
        all_visible_weights = []
        all_semantic_points = []
        all_part_based_masks = []
        fg_masks = []
        bg_masks = []
        inter_bg_masks = []
        inter_neg_points = []
        outer_neg_points = []
        for i_img in range(num_imgs):
            # 第一个mean shift操作，后的instance attention map
            # refine次数，num_gt, img_h, img_w
            # self-attention map的前景点和背景点 -> num_gt, num_point, 2
            neg_points, neg_point_feats, map_cos_fg, map_cos_bg, \
                points_bg_attn, points_fg_attn, bg_mask, outer_neg_points_, inter_bg_mask_, \
                random_points, random_labels = self.instance_attention_generation(matched_cams[i_img], 
                                                                               gt_bboxes[i_img],
                                                                               vit_feat[i_img],
                                                                               vit_feat_be_norm[i_img],
                                                                               pos_thr=self.iam_thr_pos, 
                                                                               neg_thr=self.iam_thr_neg, 
                                                                               num_gt=20, # 就是取20个特征做mean shift
                                                                               refine_times=self.iam_refine_times, 
                                                                               obj_tau=self.iam_obj_tau,
                                                                               gt_points=gt_points[i_img])
            # 在最后一次shift的instance attention上均匀采样得到的前景点，num_gt, num_point, 2 
            # (并且是在feat map size下，即小了16倍)
            # 并且vit feat上直接取（可以用roi algin/grid sample进行插值财经）num_gt, num_point, 256
            sampled_points, pos_point_feats, visible_weights, \
                semantic_points, part_masks, fg_mask = self.get_semantic_centers(map_cos_fg[-1].clone(), 
                                                                    map_cos_bg[-1].clone(), 
                                                                    gt_bboxes[i_img], 
                                                                    vit_feat_be_norm[i_img],
#                                                                     vit_feat[i_img],
                                                                    # pos_thr=0.35,
                                                                    pos_thr=self.iam_thr_pos,
                                                                    n_points_sampled=self.iam_num_points_init_pos,
                                                                    gt_points=gt_points[i_img] if self.with_gt_points else None,
                                                                    gt_labels=gt_labels[i_img])
            if self.random_iam_points:
                pseudo_points_ = random_points
                pseudo_bin_labels_ = random_labels
            else:
                pseudo_points_ = torch.cat([sampled_points, neg_points], dim=1).float()
                pseudo_bin_labels_ = torch.cat([torch.ones_like(sampled_points)[..., 0],
                                            torch.zeros_like(neg_points)[..., 0]], dim=1).long()
            pseudo_points.append(pseudo_points_)
            pseudo_bin_labels.append(pseudo_bin_labels_)
            instance_cos_fg_maps.append(map_cos_fg)
            instance_cos_bg_maps.append(map_cos_bg)
            fg_points.append(sampled_points) # 原图大小的点
            fg_point_feats.append(pos_point_feats)
            bg_point_feats.append(neg_point_feats)
            
            all_visible_weights.append(visible_weights)
            all_semantic_points.append(semantic_points)
            all_part_based_masks.append(part_masks)
            inter_bg_masks.append(inter_bg_mask_)
            fg_masks.append(fg_mask)
            bg_masks.append(bg_mask)
            outer_neg_points.append(outer_neg_points_)
            inter_neg_points.append(neg_points.reshape(-1, 2))
            
        attnshift_results = dict(
            pseudo_points=pseudo_points,
            pseudo_bin_labels=pseudo_bin_labels,
            all_visible_weights=all_visible_weights,
            all_semantic_points=all_semantic_points,
            all_part_based_masks=all_part_based_masks,
            fg_points=fg_points,
            instance_cos_fg_maps=instance_cos_fg_maps,
            fg_masks=fg_masks,
            bg_masks=bg_masks,
            outer_neg_points=outer_neg_points,
            inter_neg_points=inter_neg_points,
            inter_bg_masks=inter_bg_masks # num_gt, h, w
        )
        losses_dpm = dict()
        losses_dpm['dpm_pos_loss'] = 0
        losses_dpm['dpm_neg_loss'] = 0
        losses_dpm['dpm_acc'] = []
        
        all_grids = []
        all_norm_semantic_points = []
        all_aligned_part_centers = []
        all_aligned_part_maps = []
        all_root_scores = []
        all_part_scores = []
        all_deformable_costs = []
        all_dpm_points = []
        all_dpm_visible = []
        part_proposals_maps = []
        all_part_deformables = []
        all_aligned_part_scores = []
        all_part_offset_weights = []
        
        # forward
        vit_feat_dpm = vit_feat.flatten(2).permute(0, 2, 1)
        root_scores, part_scores, part_deformations = self(vit_feat_dpm) # bs, N, 20 # bs, num_classifier, N, 20
        matching_score_maps = []
        instance_part_score_maps = []
        additional_neg_points = []
        # calculate matched feature
        for i_img in range(num_imgs):
            labels = gt_labels[i_img]
            num_gt = len(labels)
            visible_weights = all_visible_weights[i_img] # num_gt, num_parts
            semantic_points = all_semantic_points[i_img] / 16 # num_gt, num_parts, 2
            part_based_masks = all_part_based_masks[i_img].long() # num_gt, num_parts, h, w
            
            inter_neg_mask = inter_bg_masks[i_img].reshape(num_gt, img_h, img_w)
            root_mask = F.interpolate(fg_masks[i_img].unsqueeze(1), 
                                      (patch_h, patch_w),
                                      mode='bilinear').squeeze(1).long() # num_gt, h, w
            root_score = root_scores[i_img].reshape(*featmap_sizes[2], self.num_classes)
            part_score = part_scores[i_img].reshape(-1, *featmap_sizes[2], self.num_classes)
            
            # 反例的分数 
            outer_neg_points_ = outer_neg_points[i_img] # (self.num_classifier + 1, num_gt * num_neg, 2)
            inter_neg_points_ = inter_neg_points[i_img].reshape(num_gt, -1, 2) # num_gt, num_neg_iam, 2
            # num_gt, num_classes, h, w
            cated_score = torch.cat([root_score.unsqueeze(0), 
                                     part_score], dim=0).permute(0, 3, 1, 2)
            
            norm_outer_neg_points_ = outer_neg_points_ / torch.as_tensor([img_w, img_h]).to(device).reshape(1, 1, 2)
            norm_inter_neg_points_ = inter_neg_points_ / torch.as_tensor([img_w, img_h]).to(device).reshape(1, 1, 1, 2)
            
            # num_classifier + 1, num_gt * num_neg, num_classes - > num_classifier + 1, 1, num_classes
            outer_neg_score = point_sample(cated_score, 
                                           norm_outer_neg_points_).permute(0, 2, 1).sum(1).unsqueeze(1) / (norm_outer_neg_points_.shape[-2] + 1)
            # num_classifier + 1, num_gt * num_neg_iam, num_classes - > num_classifier + 1, num_gt, 1, num_classes
            inter_neg_score = point_sample(cated_score, 
                                           norm_inter_neg_points_.repeat(self.num_classifier + 1, 1, 1, 1).flatten(1, 2)) #.permute(0, 2, 1).sum(1).unsqueeze(1)
            inter_neg_score = inter_neg_score.permute(0, 2, 1).reshape(self.num_classifier + 1, num_gt, -1, self.num_classes).sum(-2).unsqueeze(-2) / (norm_inter_neg_points_.shape[-2] + 1)
            inter_neg_score = inter_neg_score.reshape(self.num_classifier + 1, num_gt, self.num_classes)
            # num_classifier + 1, num_gt + 1, num_classes
            neg_score = torch.cat([outer_neg_score, inter_neg_score], dim=1)
            # part offset
            if self.feature_based_deform:
                if self.semantic_deform:
                    part_deformations_ = part_deformations[i_img].reshape(self.num_classifier, 
                                                                          *featmap_sizes[2], self.num_classes, 4)
                    part_deform = part_deformations_.unsqueeze(0).repeat(num_gt, 1, 1, 1, 1, 1) # num_gt, num_classifier, h, w, num_classes, 4
                    # num_gt, num_classifier, ph, pw, 4
                    part_deform = torch.gather(part_deform, 
                                 index=labels.reshape(-1, 1, 1, 1, 1, 1).repeat(1, self.num_classifier, patch_h, patch_w, 1, 4),
                                 dim=2)[..., 0, :]
                else:
                    part_deformations_ = part_deformations[i_img].reshape(self.num_classifier, *featmap_sizes[2], -1)
                    part_deform = part_deformations_.unsqueeze(0).repeat(num_gt, 1, 1, 1, 1)
            else:
                if self.semantic_deform:
                    part_deform = self.part_deform.reshape(1, self.num_classifier, 
                                            self.num_classes, 1, 1, 4).repeat(num_gt, 1, 1, 1, 1, 1)
                    # num_gt, num_classifier, num_classes, 1, 1, 4
                    part_deform = torch.gather(part_deform, 
                                              index=labels.reshape(-1, 1, 1, 1, 1, 1).repeat(1, self.num_classifier, 
                                                                                             1, 1, 1, 4),
                                              dim=2)[:, :, 0, ...] # num_gt, num_classifier, 1, 1, 4
                else:
                    # num_gt, num_classifier, 1, 1, 4
                    part_deform = self.part_deform.reshape(1, self.num_classifier, 1, 1, 4).repeat(num_gt, 1, 1, 1, 1)
            xx, yy = torch.meshgrid(torch.arange(patch_w).to(semantic_points.device), 
                                    torch.arange(patch_h).to(semantic_points.device))
            xx_norm, yy_norm = xx / patch_w, yy / patch_h
            all_grids.append(torch.stack([xx_norm, yy_norm], dim=-1))
            norm_semantic_points = semantic_points / \
                    torch.as_tensor(featmap_sizes[2]).to(semantic_points.device).flip(0).reshape(1, 1, 2)
            all_norm_semantic_points.append(norm_semantic_points)
            deform_cost_term1 = norm_semantic_points.unsqueeze(-2) - \
                                    torch.stack([xx_norm.reshape(-1),
                                                 yy_norm.reshape(-1)], dim=-1).unsqueeze(0).unsqueeze(0)
            deform_cost_term2 = deform_cost_term1 ** 2
            # num_gt, num_parts, h, w, 4
            deform_cost = torch.cat([deform_cost_term1, 
                                     deform_cost_term2], dim=-1).reshape(-1, 
                                                                         self.num_semantic_points, 
                                                                         patch_w, 
                                                                         patch_h,
                                                                         4).permute(0, 1, 3, 2, 4) 
            weighted_deform_cost = self.deformable_cost_weight * deform_cost # num_gt, num_parts, h, w, 4
            # all_deformable_costs.append(weighted_deform_cost.sum(-1))
            all_deformable_costs.append(weighted_deform_cost)
            instance_root_score = root_score.unsqueeze(0).repeat(num_gt, 1, 1, 1) # num_gt, h, w, num_classes
            instance_part_score = part_score.unsqueeze(0).repeat(num_gt, 1, 1, 1, 1) # num_gt, num_classifier, h, w, num_classes
            instance_part_score_maps.append(instance_part_score)
            
            # matching stretegy
            with torch.no_grad():
                if self.sklearn_matching:
                    assert not self.bce_loss.use_sigmoid, 'bce wo sklearn matching' 
                    # 匈牙利匹配之所以和之前cuda tensor排列组合匹配结果存在不同，
                    # 是因为出现invisible的时候，对应的分类器完全可以随机分配，但是loss最小的组合还是不变的，
                    # 毕竟被分配到的训练器不会学习。
                    # 实际上训练loss的样本的匹配完全相同
                    # assert False, 'no sklearn in'
                # 匈牙利匹配
                    part_matching_scores = (instance_part_score.unsqueeze(2) - 
                                            (part_deform.unsqueeze(2) * weighted_deform_cost.unsqueeze(1)).sum(-1).unsqueeze(-1)
                                           ) * part_based_masks.unsqueeze(1).unsqueeze(-1) # num_gt, num_classfier, num_parts, h, w, num_classes
                    matching_score_maps.append(part_matching_scores)
                    # num_gt, 1, num_parts, 1
                    part_visibles = (part_based_masks.flatten(2).sum(-1) > 0).long().reshape(num_gt, 
                                                                                            self.num_semantic_points)
                    if self.part_mean_max:
                        # num_gt, num_classifier, num_part, h, w, 21 -> num_gt, n_cls, n_p, hw, 21
                        _part_matching_scores_ = part_matching_scores.flatten(-3, -2).softmax(-1)
                        _part_based_masks_ = part_based_masks.unsqueeze(1).unsqueeze(-1).flatten(-3, -2)
                        # num_gt, n_cls, n_p, hw, 21
                        weight = (1 / torch.clamp(1 - _part_matching_scores_, 1e-12, None)) * _part_based_masks_
                        # num_gt, n_cls, n_p, hw, 21
                        weight = weight / torch.clamp(weight.sum(dim=-2).unsqueeze(dim=-2), 1e-12, None)
                        part_matching_scores = (weight * _part_matching_scores_ * _part_based_masks_).sum(dim=-2)   
                    else:
                        if self.part_proposal_topk > 0:
                            # num_gt, num_classifier, num_part, h*w, 20
                            # num_gt, num_classifier, num_part, k, 20 - > num_gt, num_classifier, num_part, 20
                            top_part_matching_scores, topk_index_ = part_matching_scores.flatten(-3, -2).topk(dim=-2, k=self.part_proposal_topk)
                            part_matching_scores = top_part_matching_scores.sum(-2)
                            part_based_averaging = part_based_masks.unsqueeze(1).unsqueeze(-1).flatten(-3, -2).repeat(1, self.num_classifier, 1, 1, self.num_classes)
                            part_based_averaging = torch.gather(part_based_averaging, index=topk_index_, dim=-2)[..., 0]
                            part_matching_scores = part_matching_scores / (part_based_averaging.sum(-1).unsqueeze(-1) + 1)
                            part_matching_scores = part_matching_scores.softmax(-1)
                        else:
                            # num_gt, num_classifier, num_part, 20
                            part_matching_scores = part_matching_scores.flatten(-3, -2).sum(-2) / (part_based_masks.unsqueeze(1).flatten(-2, -1).sum(-1).unsqueeze(-1) + 1)
                            # num_gt, num_classifier, num_parts, num_classes
                            part_matching_scores = part_matching_scores.softmax(-1)
                        
                    part_matching_costs = -part_matching_scores[torch.arange(num_gt).to(device), ..., labels]
                    part_matching_costs = part_matching_costs * part_visibles.reshape(num_gt, 1, self.num_semantic_points)
                    matched_row_inds = []
                    matched_col_inds = []
                    for i_gt_, matching_costs in enumerate(part_matching_costs.cpu().numpy()):
                        matched_row_inds_, matched_col_inds_ = linear_sum_assignment(matching_costs)
                        matched_row_inds_ = torch.as_tensor(matched_row_inds_).to(device)
                        matched_col_inds_ = torch.as_tensor(matched_col_inds_).to(device)
                        matched_row_inds.append(matched_row_inds_)
                        matched_col_inds.append(matched_col_inds_)
                    matched_row_inds = torch.stack(matched_row_inds)
                    matched_col_inds = torch.stack(matched_col_inds)
                else:
                    # 使用排列组合的cuda tensor进行，在part数量少的时候且gt少的，排列组合少，占用显存小可以加快
                    # 但像coco有时候很大的gt数量，然后part又多，显存都爆掉了，
                    part_matching_scores = (instance_part_score.unsqueeze(2) - 
                                            (part_deform.unsqueeze(2) * weighted_deform_cost.unsqueeze(1)).sum(-1).unsqueeze(-1)
                                           ) * part_based_masks.unsqueeze(1).unsqueeze(-1) # num_gt, num_classfier, num_parts, h, w, num_classes
                    matching_score_maps.append(part_matching_scores)
                    # 
                    part_matching_scores = part_matching_scores.flatten(-3, -2).sum(-2) / (part_based_masks.unsqueeze(1).flatten(-2, -1).sum(-1).unsqueeze(-1) + 1)
                    # num_gt, num_classifier, num_part, 20
                    part_matching_scores = part_matching_scores[:, self.all_row_matchings.reshape(-1), self.all_col_matchings.reshape(-1), :].reshape(num_gt, self.num_matchings ** 2, self.num_semantic_points, self.num_classes)
                    # num_gt, num_matchings ^ 2, num_parts(只是part 的索引index的维度), 20
                    if self.bce_loss.use_sigmoid: # BCE loss
                        assert False, 'bce without sklearn matching'
                        # num_gt, num_matchings ^ 2, num_parts, num_classes
                        part_matching_scores = part_matching_scores.sigmoid()
                        # num_gt, 1, num_parts, 1
                        part_visibles = (part_based_masks.flatten(2).sum(-1) > 0).long().reshape(num_gt, 
                                                                                                self.num_semantic_points)
                        # num_gt, num_matchings ^ 2, num_parts, 1
                        part_visibles = part_visibles[:, self.all_col_matchings.reshape(-1)].reshape(num_gt, 
                                                                                                     self.num_matchings ** 2,
                                                                                                     self.num_semantic_points,
                                                                                                     1)
                        # 因为invisible存在0的值，softmax会变为0.05，因此需要invisble乘上去，保证没有分数
                        # 实际上，不乘也可以，因为对于同一个gt来说，invisible的part是相同的，由于是相加，invisible不会影响最后分数
                        part_matching_scores = part_matching_scores * part_visibles

                        add_instance_scores = (instance_root_score * root_mask.unsqueeze(-1)).flatten(1, 2).sum(1).unsqueeze(1) / (root_mask.flatten(1, 2).sum(1).unsqueeze(-1).unsqueeze(1) + 1) 
                        # num_gt, 1, 20
                        add_instance_scores = add_instance_scores.sigmoid()
                        part_pos_cost = -(part_matching_scores + 1e-6).log()
                        part_neg_cost = -(1 - part_matching_scores + 1e-6).log()
                        root_pos_cost = -(add_instance_scores + 1e-6).log()
                        root_neg_cost = -(1 - add_instance_scores + 1e-6).log()
                        # num_gt, num_matchings ^ 2, num_parts
                        part_cost = part_pos_cost[torch.arange(num_gt).to(device), :, :, labels] - \
                                part_neg_cost[torch.arange(num_gt).to(device), :, :, labels]
                        # num_gt, 1
                        root_cost = root_pos_cost[torch.arange(num_gt).to(device), :, labels] - \
                            root_neg_cost[torch.arange(num_gt).to(device), :, labels]

                        # 我们找loss最小，就是对应softmax分数最大，就是负的分数cost，最小
                        instance_cost = (part_cost + root_cost.unsqueeze(1)).sum(-1)
                    else: # CE loss
                        # num_gt, num_matchings ^ 2, num_parts, num_classes
                        part_matching_scores = part_matching_scores.softmax(-1)
                        # num_gt, 1, num_parts, 1
                        part_visibles = (part_based_masks.flatten(2).sum(-1) > 0).long().reshape(num_gt, 
                                                                                                self.num_semantic_points)
                        # num_gt, num_matchings ^ 2, num_parts, 1
                        part_visibles = part_visibles[:, self.all_col_matchings.reshape(-1)].reshape(num_gt, 
                                                                                                     self.num_matchings ** 2,
                                                                                                     self.num_semantic_points,
                                                                                                     1)
                        # 因为invisible存在0的值，softmax会变为0.05，因此需要invisble乘上去，保证没有分数
                        # 实际上，不乘也可以，因为对于同一个gt来说，invisible的part是相同的，由于是相加，invisible不会影响最后分数
                        part_matching_scores = part_matching_scores * part_visibles

                        add_instance_scores = (instance_root_score * root_mask.unsqueeze(-1)).flatten(1, 2).sum(1).unsqueeze(1) / (root_mask.flatten(1, 2).sum(1).unsqueeze(-1).unsqueeze(1) + 1) 
                        # num_gt, 1, 20
                        add_instance_scores = add_instance_scores.softmax(-1)

                        # num_gt, num_matchings ^ 2, 21
                        final_instance_scores = part_matching_scores.sum(-2) + add_instance_scores
                        # 我们找loss最小，就是对应softmax分数最大，就是负的分数cost，最小
                        instance_cost = -final_instance_scores[torch.arange(num_gt).to(device), :, labels]

                    # 取loss最小的 num_gt(每个最小的匹配index)
                    matching_index = instance_cost.min(1)[1]
                    # print(instance_cost.min(1)[0])
                    # 分类器匹配顺序 num_gt, num_parts
                    matched_row_inds = self.all_row_matchings[matching_index]
                    # part 特征匹配顺序 num_gt, num_parts
                    matched_col_inds = self.all_col_matchings[matching_index]
                    
            # part 分类器分数
            masked_instance_part_score = []
            masked_part_proposals_map = []
            aligned_part_masks = []
            aligned_part_centers = []
            aligned_part_deforms = []
            aligned_part_scores = []
            part_offset_weights = []
            for i_gt_, (row_inds_, col_inds_, part_score_, deform_cost_, part_mask_, label_, part_centers_) in \
                enumerate(zip(matched_row_inds, matched_col_inds, instance_part_score, 
                    weighted_deform_cost, part_based_masks.unsqueeze(-1),
                    labels, semantic_points)):
                part_score_ = part_score_[row_inds_] # num_part, h, w, 20
                part_deform_ = part_deform[i_gt_][row_inds_] # num_part, 1, 1, 4
                deform_cost_ = deform_cost_[col_inds_] # num_part, h, w, 4
                part_mask_ = part_mask_[col_inds_] # num_part, h, w, 1
                final_part_score = (part_score_ - (part_deform_ * 
                                                   deform_cost_).sum(-1).unsqueeze(-1)) * part_mask_ # num_part, h, w, 20
                aligned_part_scores.append((part_score_ * part_mask_)[..., label_])
                aligned_part_deforms.append((part_deform_ * deform_cost_ * part_mask_).sum(-1))
                part_offset_weights.append(part_deform_)
                aligned_part_masks.append(part_mask_)
                aligned_part_centers.append(part_centers_[col_inds_] * 16)
                
                # 直接logits * mask 来获取map，然后在mask=0处加上 mask！=0范围中最小值，保证在选择的时候不被选到
                # num_parts, H, W
                proposal_maps = F.interpolate(((part_score_ - (part_deform_ * deform_cost_).sum(-1).unsqueeze(-1))
                                   * part_mask_).permute(0, 3, 1, 2), 
                                  size=(img_h, img_w),
                                  mode='bilinear')[:, label_, ...]
                proposal_maps_mins = proposal_maps.reshape(self.num_classifier, -1).min(-1)[0][:, None, None, None] # num_gt, 
                min_maps = proposal_maps_mins * (~(part_mask_.bool())).long()
                min_maps = F.interpolate(min_maps.permute(0, 3, 1, 2), (img_h, img_w), mode='bilinear')[:, 0]
                proposal_maps = proposal_maps + min_maps
                masked_part_proposals_map.append(proposal_maps)
                if self.part_mean_max:
                    # final_part_score -> num_part, h, w, num_classes  
                    # part_mask_ -> num_part, h, w, 1
                    _final_part_score_ = final_part_score.flatten(1, 2).softmax(-1)
                    _part_mask_ = part_mask_.flatten(1, 2)
                    # num_gt, n_cls, n_p, hw, 21
                    weight = (1 / torch.clamp(1 - _final_part_score_, 1e-12, None)) * _part_mask_
                    # num_gt, n_cls, n_p, hw, 21
                    weight = weight / torch.clamp(weight.sum(dim=1).unsqueeze(dim=1), 1e-12, None)
                    _final_part_score_ = (weight * _final_part_score_ * _part_mask_).sum(dim=1)
                    masked_instance_part_score.append(_final_part_score_) # num_parts, num_classes
                else:
                    if self.part_proposal_topk > 0:
                        top_scores, topk_index_ = final_part_score.flatten(1, 2).topk(dim=1, k=self.part_proposal_topk)
                        top_scores = top_scores.sum(1) # num_part, num_classes
                        part_mask_averaging = part_mask_.flatten(1, 2).repeat(1, 1, self.num_classes)
                        part_mask_averaging = torch.gather(part_mask_averaging, index=topk_index_, dim=1)[..., 0] # num_part, topk
                        top_scores = top_scores / (part_mask_averaging.sum(-1).unsqueeze(-1) + 1)
                        masked_instance_part_score.append(top_scores)
                    else:
                        masked_instance_part_score.append(
                            final_part_score.flatten(1, 2).sum(1) / (part_mask_.flatten(1, 2).sum(1) + 1)
                        ) # num_parts, num_classes
            aligned_part_scores = F.interpolate(torch.stack(aligned_part_scores),
                                                (img_h, img_w),
                                                mode='bilinear') # num_gt, num_parts, H, W
            all_aligned_part_scores.append(aligned_part_scores)
            
            aligned_part_deforms = F.interpolate(torch.stack(aligned_part_deforms),
                                                 (img_h, img_w),
                                                 mode='bilinear') # num_gt, num_parts, H, W
            part_offset_weights = torch.stack(part_offset_weights)
            part_offset_weights = F.interpolate(part_offset_weights.permute(0, 1, 4, 2, 3).flatten(1, 2),
                                               (img_h, img_w),
                                               mode='bilinear').unflatten(1, [self.num_classifier, 4]).permute(0, 1, 3, 4, 2)
            
            all_part_deformables.append(aligned_part_deforms)
            all_part_offset_weights.append(part_offset_weights)
            
            masked_part_proposals_map = torch.stack(masked_part_proposals_map) #num_gt, num_parts, H, W
            part_proposals_maps.append(masked_part_proposals_map)
            masked_instance_part_score = torch.stack(masked_instance_part_score) # num_gt, num_parts, num_classes
            aligned_part_masks = torch.stack(aligned_part_masks) # num_gt, num_parts, h, w
            aligned_part_centers = torch.stack(aligned_part_centers) # num_gt, num_parts, 2
            all_aligned_part_maps.append(aligned_part_masks)
            all_aligned_part_centers.append(aligned_part_centers)
            
            all_root_scores.append(instance_root_score)
            all_part_scores.append(instance_part_score)
            
            # root 分类器分数  
            masked_instance_root_score = (instance_root_score * root_mask.unsqueeze(-1)).flatten(1, 2).sum(1).unsqueeze(1) / (root_mask.flatten(1).sum(1).unsqueeze(1).unsqueeze(-1) + 1) # num_gt, 1, num_classes
            
            # # ce losses pos_losses->softmax(sum) -> loss -> ap=50.5
            # # num_gt, num_parts
            # part_visibles = aligned_part_masks.flatten(2).sum(-1) > 0
            # # num_gt, (num_parts + 1), num_classes -> num_gt, num_classes
            # # 用part invisble确定哪些part不用参加训练
            # object_score = torch.cat([
            #     masked_instance_part_score.softmax(-1) * part_visibles.unsqueeze(-1),
            #     masked_instance_root_score.softmax(-1)], dim=1).sum(1)
            # # 由于相加，我们还需要让其分数取平均
            # object_score = object_score / (part_visibles.sum(1) + 1).unsqueeze(-1)
            # # 因为这个算了softmax，所以loss里面不能引入softmax进行计算哦
            # dpm_pos_loss = self.bce_loss(object_score, labels).sum() / num_gt
            # # 这里面要除以num img 防止batchsize不同的时候loss不同
            # losses_dpm['dpm_pos_loss'] += dpm_pos_loss / num_imgs
            # losses_dpm['dpm_acc'].append(accuracy(object_score, labels))
            
            # ce losses pos_losses->   softmax -> loss1; softmax -> loss2 -> ap=51.1
            # num_gt, num_parts
            part_visibles = aligned_part_masks.flatten(2).sum(-1) > 0
            if self.bce_loss.use_sigmoid: # BCE loss
                root_loss = self.bce_loss(
                    masked_instance_root_score.sigmoid().reshape(-1, self.num_classes), 
                    labels).unsqueeze(1) # num_gt, 1, num_classes
                part_loss = self.bce_loss(
                    # num_gt, num_part, num_classes
                    masked_instance_part_score.sigmoid().reshape(-1, self.num_classes),
                    labels.reshape(-1, 1).repeat(1, self.num_classifier).reshape(-1)
                ).reshape(-1, self.num_classifier, self.num_classes)
            else: # CE loss
                root_loss = self.bce_loss(
                    masked_instance_root_score.softmax(-1).reshape(-1, self.num_classes), 
                    labels).unsqueeze(1) # num_gt, 1, num_classes
                if self.part_mean_max:
                    part_loss = self.bce_loss(
                        # num_gt, num_part, num_classes
                        masked_instance_part_score.reshape(-1, self.num_classes),
                        labels.reshape(-1, 1).repeat(1, self.num_classifier).reshape(-1)
                    ).reshape(-1, self.num_classifier, self.num_classes) 
                else:
                    part_loss = self.bce_loss(
                        # num_gt, num_part, num_classes
                        masked_instance_part_score.softmax(-1).reshape(-1, self.num_classes),
                        labels.reshape(-1, 1).repeat(1, self.num_classifier).reshape(-1)
                    ).reshape(-1, self.num_classifier, self.num_classes) 
                
            # 把invisible的特征摒弃掉
            part_loss = part_loss * part_visibles.unsqueeze(-1)
            
            if not self.ablation_cls_weight:
                # 这个应该有错，加了classifier数量倍的root，root loss权重多了数倍
                dpm_pos_loss = (root_loss + part_loss).flatten(1).sum(-1) / (part_visibles.sum(1) + 1)
            else:
                # assert self.root_cls_weight != 0 # 不能等于0，否则取平均有问题
                # 下面这个是对的，因为原来的ablation都是上面，所以我们先弄上面的
                dpm_pos_loss = torch.cat([self.root_cls_weight * root_loss, part_loss], dim=1).sum(1) / (part_visibles.sum(1) + 1).unsqueeze(-1)
            dpm_pos_loss = dpm_pos_loss.sum() / num_gt
            losses_dpm['dpm_pos_loss'] += dpm_pos_loss / num_imgs
            losses_dpm['dpm_acc'].append(accuracy((root_loss + part_loss).sum(1).exp(), labels))
            # neg losses
            if self.bce_loss.use_sigmoid: # BCE loss
                assert False, 'no implement'
                # # num_classifier + 1, num_all_negs, num_classes
                # neg_score = neg_score.sigmoid()
                # num_negs = neg_score.shape[0] * neg_score.shape[1]
                # dpm_neg_loss = self.bce_loss(
                #     neg_score.reshape(-1, self.num_classes), 
                #     (torch.ones_like(neg_score[..., 0]).long() \
                #      * self.num_classes).reshape(-1)).sum() / (num_negs * num_gt)
                # losses_dpm['dpm_neg_loss'] += self.neg_loss_weight * (dpm_neg_loss / num_imgs)
            else:
                # num_classifier + 1, num_gt + 1, num_classes
                neg_score = neg_score.softmax(-1)
                num_neg_samples = (num_gt + 1) * (self.num_classifier + 1)
                dpm_neg_loss = self.bce_loss(
                    neg_score.reshape(-1, self.num_classes), 
                    (torch.ones_like(neg_score[..., 0]).long() \
                     * (self.num_classes - 1)).reshape(-1)).sum() / num_neg_samples
                losses_dpm['dpm_neg_loss'] += self.neg_loss_weight * (dpm_neg_loss / num_imgs)
            
            # 为mask head提供点
            # max_flatten_index = torch.argmax(masked_part_proposals_map.flatten(-2), dim=-1)
            # num_gt, num_parts, num_topk
            temp_masked_part_proposals_map = F.interpolate(
                masked_part_proposals_map,      
                (img_h // self.topk_stride, img_w // self.topk_stride),
                mode='bilinear')
            # print(temp_masked_part_proposals_map.shape)
            max_flatten_index = torch.topk(temp_masked_part_proposals_map.flatten(-2), 
                                           k=self.part_points_topk, dim=-1)[1].reshape(num_gt, -1)
            # max_flatten_index = torch.topk(masked_part_proposals_map.flatten(-2), 
            #                                k=self.part_points_topk, dim=-1)[1].reshape(num_gt, -1)
            # max_flatten_index = torch.argmax(masked_part_proposals_map.flatten(-2), dim=-1)
            y_coords = (max_flatten_index // (img_w // self.topk_stride)  + 0.5) * self.topk_stride
            x_coords = (max_flatten_index % (img_w // self.topk_stride)+ 0.5) * self.topk_stride
            part_locations = torch.stack([x_coords, y_coords], dim=-1)
            # num_gt, num_parts
            visible_weights = (aligned_part_masks.flatten(2).sum(-1) > 0)
            # num_gt, num_parts, num_topk
            visible_weights = visible_weights.unsqueeze(-1).repeat(1, 1, 
                                                                   self.part_points_topk).reshape(num_gt, -1)
            part_locations[~visible_weights] = -1000
            # 这边必须用invisible的点把目标的位置值给置-1000，用于后面的mask sup.的过滤
            all_dpm_points.append(part_locations)
            all_dpm_visible.append(visible_weights)
            
            if self.instance_neg_points_topk > 0:
                # 反例的确定点
                # num_gt, h, w, num_classes [cat] num_gt, num_parts, h, w, num_classes
                # num_gt, h, w, num_classes -> num_gt, num_classes, h, w,  
                # overall_instance_score = torch.cat([instance_root_score.softmax(-1).unsqueeze(1),
                #                                    instance_part_score.softmax(-1)], dim=1).mean(1).permute(0, 3, 1, 2)[:, -1, ...]
                overall_instance_score = instance_root_score.softmax(-1).permute(0, 3, 1, 2) #[:, -1, ...]
                overall_instance_score = F.interpolate(
                    overall_instance_score,
                    (img_h // self.topk_stride, img_w // self.topk_stride), 
                    mode='bilinear')[:, -1, ...]
                masked_overall_instance_neg_score = overall_instance_score * \
                        F.interpolate(inter_neg_mask.unsqueeze(1).float(), 
                                      (img_h // self.topk_stride, img_w // self.topk_stride), 
                                      mode='bilinear').squeeze(1).bool()
                neg_max_flatten_index = torch.topk(masked_overall_instance_neg_score.flatten(1), 
                                               k=self.instance_neg_points_topk, dim=-1)[1].reshape(num_gt, -1)
                # for vis
                overall_instance_score_ = F.interpolate(overall_instance_score.unsqueeze(1),
                                                          (img_h, img_w), mode='bilinear').squeeze(1)
                masked_overall_instance_neg_score_ = overall_instance_score_ * inter_neg_mask
                # for vis
                neg_y_coords = (neg_max_flatten_index // (img_w // self.topk_stride) + 0.5) * self.topk_stride
                neg_x_coords = (neg_max_flatten_index % (img_w // self.topk_stride) + 0.5) * self.topk_stride
                # num_gt, instance_neg_points_topk, 2
                neg_locations = torch.stack([neg_x_coords, neg_y_coords], dim=-1)
                additional_neg_points.append(neg_locations)
                
                attnshift_results['masked_overall_instance_neg_score'] = masked_overall_instance_neg_score_
                attnshift_results['neg_locations'] = neg_locations
        
        # 引入所有dpm 与 attn shift的点
        all_mask_sup_points = []
        all_mask_sup_visibles = []
        for dpm_p, dpm_v, attn_p, attn_v in \
                zip(all_dpm_points, all_dpm_visible, all_semantic_points, all_visible_weights):
            if self.mask_gt_sets == 1: # 1->only attn_shift 2->only dpm 3-> attn_shift+dpm
                mask_sup_points_per_img = attn_p.clone()
                mask_sup_visibles_per_img = attn_v.clone()
            elif self.mask_gt_sets == 2:
                mask_sup_points_per_img = dpm_p.clone()
                mask_sup_visibles_per_img = dpm_v.clone()
            elif self.mask_gt_sets == 3:
                mask_sup_points_per_img = torch.cat([attn_p, dpm_p], dim=1)
                mask_sup_visibles_per_img = torch.cat([attn_v, dpm_v], dim=1)
                
            all_mask_sup_points.append(mask_sup_points_per_img)
            all_mask_sup_visibles.append(mask_sup_visibles_per_img)
            
        if self.instance_neg_points_topk > 0:
            # 在随机采样正反例上，引入额外的反例点
            pseudo_points = attnshift_results['pseudo_points']
            pseudo_bin_labels = attnshift_results['pseudo_bin_labels']
            
            new_pseudo_points = []
            new_pseudo_bin_labels = []
            for iam_points, iam_labels, add_neg_points in \
                zip(pseudo_points, pseudo_bin_labels, additional_neg_points):
                iam_points = torch.cat([iam_points, add_neg_points], dim=1)
                iam_labels = torch.cat([
                    iam_labels, torch.zeros_like(add_neg_points[..., 0]).long()], dim=1)
                new_pseudo_points.append(iam_points)
                new_pseudo_bin_labels.append(iam_labels)
            attnshift_results['pseudo_points'] = new_pseudo_points
            attnshift_results['pseudo_bin_labels'] = new_pseudo_bin_labels
            
        if self.exclude_iam_pos:
            # 直接将pseudo_points的正例点去掉（减少噪声）
            # 利用设计进行去除，因为pseudo_points 是 cat[pos + neg]
            pseudo_points = attnshift_results['pseudo_points']
            pseudo_bin_labels = attnshift_results['pseudo_bin_labels']
            new_pseudo_points = []
            new_pseudo_bin_labels = []
            for iam_points, iam_labels, root_score_map, labels_ in \
                zip(pseudo_points, pseudo_bin_labels, all_root_scores, gt_labels):
                final_points = iam_points[:, self.iam_num_points_init_pos:] # 只取反例点
                final_labels = iam_labels[:, self.iam_num_points_init_pos:]

                if self.exclude_iam_neg:
                    # num_gt, h, w, num_classes
                    num_gt = final_points.shape[0]
                    root_score_map = root_score_map.softmax(-1)[torch.arange(num_gt).to(device), ..., labels_]
                    neg_points_norm = final_points / torch.as_tensor([img_w, img_h]).to(device).reshape(1, 1, 2)
                    neg_root_scores = point_sample(root_score_map.unsqueeze(1),
                                                  neg_points_norm)[:, 0] # num_gt, num_part
                    no_keep = (neg_root_scores >= self.iam_neg_thr).bool() # num_gt, num_part
                    # import pdb
                    # pdb.set_trace()
                    
                    final_points[no_keep] = -1e3 # 应该距离不在框内就不会算loss -> num_gt, num_part, 2
                    final_labels[no_keep] = -1 # 表示不算loss
                
                new_pseudo_points.append(final_points)
                new_pseudo_bin_labels.append(final_labels)
            attnshift_results['pseudo_points'] = new_pseudo_points
            attnshift_results['pseudo_bin_labels'] = new_pseudo_bin_labels
            
        if self.iam_pos_thr > 0: # 利用root 分数 筛选 iam 的label
            pseudo_points = attnshift_results['pseudo_points']
            pseudo_bin_labels = attnshift_results['pseudo_bin_labels']
            new_pseudo_points = []
            new_pseudo_bin_labels = []
            for iam_points, iam_labels, root_score_map, part_score_map, labels_ in \
                zip(pseudo_points, pseudo_bin_labels, all_root_scores, all_part_scores, gt_labels):
                neg_points = iam_points[:, self.iam_num_points_init_pos:]
                neg_labels = iam_labels[:, self.iam_num_points_init_pos:]
                
                pos_points = iam_points[:, :self.iam_num_points_init_pos] # num_gt, num_part, 2
                pos_labels = iam_labels[:, :self.iam_num_points_init_pos]
                # num_gt, h, w, num_classes
                num_gt = pos_points.shape[0]
                root_score_map = root_score_map.softmax(-1)[torch.arange(num_gt).to(device), ..., labels_]
                pos_points_norm = pos_points / torch.as_tensor([img_w, img_h]).to(device).reshape(1, 1, 2)
                pos_root_scores = point_sample(root_score_map.unsqueeze(1),
                                              pos_points_norm)[:, 0] # num_gt, num_part
                no_keep = (pos_root_scores <= self.iam_pos_thr).bool() # num_gt, num_part
                
                pos_points[no_keep] = -1e3 # 应该距离不在框内就不会算loss -> num_gt, num_part, 2
                pos_labels[no_keep] = -1 # 表示不算loss
                
                final_points = torch.cat([pos_points, neg_points], dim=1)
                final_labels = torch.cat([pos_labels, neg_labels], dim=1)
                new_pseudo_points.append(final_points)
                new_pseudo_bin_labels.append(final_labels)
            attnshift_results['pseudo_points'] = new_pseudo_points
            attnshift_results['pseudo_bin_labels'] = new_pseudo_bin_labels
            
        losses_dpm['dpm_acc'] = torch.as_tensor(losses_dpm['dpm_acc']).to(device).sum() / max(1, num_imgs)
        # losses_dpm['dpm_aux_acc'] = torch.as_tensor(losses_dpm['dpm_aux_acc']).to(device).mean()
        attnshift_results['all_mask_sup_points'] = all_mask_sup_points
        attnshift_results['all_mask_sup_visibles'] = all_mask_sup_visibles  
        attnshift_results['all_dpm_points'] = all_dpm_points
        attnshift_results['all_dpm_visible'] = all_dpm_visible
        attnshift_results['matching_score_maps'] = matching_score_maps
        attnshift_results['instance_part_score_maps'] = instance_part_score_maps
        attnshift_results['part_proposals_maps'] = part_proposals_maps
        attnshift_results['all_aligned_part_maps'] = all_aligned_part_maps
        attnshift_results['all_deformable_costs'] = all_deformable_costs
        attnshift_results['all_aligned_part_centers'] = all_aligned_part_centers
        attnshift_results['all_root_scores'] = all_root_scores
        attnshift_results['all_grids'] = all_grids
        attnshift_results['all_norm_semantic_points'] = all_norm_semantic_points
        attnshift_results['all_part_deformables'] = all_part_deformables
        attnshift_results['all_aligned_part_scores'] = all_aligned_part_scores
        attnshift_results['additional_neg_points'] = additional_neg_points
        attnshift_results['all_part_offset_weights'] = all_part_offset_weights
        
#         # evaluate 前景/背景性能
#         # attn shift 前景/背景的性能
#         miou_attn = []
#         for i_img, (masks, pred_masks) in enumerate(zip(gt_masks, attnshift_results['fg_masks'])):
#             for i_gt, (mask, pred_mask) in enumerate(zip(masks, pred_masks)):
#                 pred_mask = pred_mask.bool()
#                 inter_ = (pred_mask & mask).sum()
#                 union_ = (pred_mask | mask).sum()
#                 iou = inter_ / (union_ + 1e-6)
#                 miou_attn.append(iou)
#         # print('attn_miou', miou_attn)
#         miou_attn = (torch.as_tensor(miou_attn).sum() / max(1, len(miou_attn))).cuda()
#         losses_dpm['miou_attn'] = miou_attn
        
#         # dpm 前景/背景的性能
#         dpm_cls_thr = [0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.0]
#         for thr in dpm_cls_thr:
#             losses_dpm['miou_dpm_{:0.2f}'.format(thr)] = []
#         dpm_masks = []
#         dpm_pseudo_points = []
#         dpm_pseudo_bin_labels = []
#         # dpm_
#         for i_img, (masks, root_logits_maps, part_logits_maps, bboxes, labels) in \
#             enumerate(zip(gt_masks, attnshift_results['all_root_scores'], 
#                           attnshift_results['instance_part_score_maps'], gt_bboxes, gt_labels)):
#             num_gt = len(bboxes)
#             root_logits_maps = F.interpolate(root_logits_maps.permute(0, 3, 1, 2),
#                                             (img_h, img_w),
#                                             mode='bilinear')
#             part_logits_maps = part_logits_maps[0].permute(0, 3, 1, 2)
#             part_logits_maps = F.interpolate(part_logits_maps, size=(img_h, img_w), mode='bilinear')
#             ins_root_scores = root_logits_maps.unsqueeze(1).softmax(2)
#             ins_part_scores = part_logits_maps.unsqueeze(0).repeat(num_gt, 1, 1, 1, 1).softmax(2)
#             overall_score = torch.cat([ins_root_scores, ins_part_scores], dim=1).mean(1)
#             # num_gt, img_h, img_w
#             overall_score = overall_score[torch.arange(num_gt).to(overall_score.device), labels, ...]
            
#             _dpm_pseudo_points = []
#             _dpm_pseudo_bin_labels = []
#             for i, (o_s, bb, g_m) in enumerate(zip(overall_score, bboxes, masks)):
#                 bbox_mask = torch.zeros_like(o_s)
#                 bbox_mask[int(bb[1]):int(bb[3]), int(bb[0]):int(bb[2])] = 1
#                 o_s = o_s * bbox_mask
#                 g_m = g_m.bool()
#                 for thr in dpm_cls_thr:
#                     # o_s_temp = o_s.clone()
#                     # non_zeros = o_s_temp[o_s_temp != 0]
#                     # o_s_temp = (o_s_temp - non_zeros.min()) / (non_zeros.max() - non_zeros.min())
#                     # o_s_temp = o_s_temp > (o_s_temp.max() * thr)
#                     o_s_temp = o_s > thr
#                     o_s_temp_neg = o_s <= 0.05
#                     inter_ = (g_m & o_s_temp).sum()
#                     union_ = (g_m | o_s_temp).sum()
#                     iou = inter_ / (union_ + 1e-6)
#                     losses_dpm['miou_dpm_{:0.2f}'.format(thr)].append(iou)
#                     if thr == self.dpm_fg_thr:
#                         coor, label = get_mask_points_single_box_cls_fg_bg(
#                                 o_s_temp, ~o_s_temp * bbox_mask, 
#                                 num_gt=self.iam_num_points_init_neg + self.iam_num_points_init_pos)
#                         dpm_masks.append(o_s_temp)
#                         coor = coor.flip(1).float()
#                         _dpm_pseudo_points.append(coor)
#                         _dpm_pseudo_bin_labels.append(label)
#             _dpm_pseudo_points = torch.stack(_dpm_pseudo_points)
#             _dpm_pseudo_bin_labels = torch.stack(_dpm_pseudo_bin_labels)
#             dpm_pseudo_points.append(_dpm_pseudo_points)
#             dpm_pseudo_bin_labels.append(_dpm_pseudo_bin_labels)
#         # 所有都取mean
#         for thr in dpm_cls_thr:
#             # print('cls_{:0.2f}'.format(thr), losses_dpm['miou_dpm_{:0.2f}'.format(thr)])
#             _miou_ = torch.as_tensor(losses_dpm['miou_dpm_{:0.2f}'.format(thr)])
#             losses_dpm['miou_dpm_{:0.2f}'.format(thr)] = (_miou_.sum() / max(1, len(_miou_))).cuda()
#             # if torch.isnan(losses_dpm['miou_dpm_{:0.2f}'.format(thr)]).sum() != 0:
#             #     print(losses_dpm['miou_dpm_{:0.2f}'.format(thr)])
#             #     print(_miou_, _miou_.sum(), max(1, len(_miou_)))
#             #     exit()
#         attnshift_results['dpm_masks'] = dpm_masks
#         if self.dpm_fg_thr > 0:
#             print('dpm cls thr false')
#             # 说明使用cls 分类作为前景点
#             attnshift_results['pseudo_points'] = dpm_pseudo_points
#             attnshift_results['pseudo_bin_labels'] = dpm_pseudo_bin_labels
#         # 利用class classifier map 获得前景点
        return attnshift_results, losses_dpm
    
    def uniform_sample_grid(self, maps, vit_feat, 
                            rois=None, thr=0.35, 
                            n_points=20, gt_points=None):
        select_coords = []
        for i_obj, map_ in enumerate(maps):
            pos_map = map_ >= thr
            num_pos = pos_map.sum()
            pos_idx = pos_map.nonzero() # 这里用到的nonzero获得坐标是 y,x 而不是x,y
            if num_pos >= n_points: # 所需要采样的点，比点数多时，对其军用采样
                grid = torch.arange(0, num_pos, step=num_pos//n_points)[:n_points]
                coords = pos_idx[grid]
            elif num_pos > 0: # 点数少于采样点时候，递归取点做填充
                coords = pos_idx
                coords = fill_in_idx(coords, n_points)

            else: # num_pos == 0 # 没有点时候分两种情况
                if rois is not None: # 有roi输入时只中心点
                    coords = ((rois[i_obj][:2] + rois[i_obj][2:]) // (2 * 16)).long().view(1, 2).flip(1)
                    coords = coords.repeat(n_points, 1)
                else:
                    pos_map = map_ >= 0 # 或者直接把阈值调到0来获取正例点
                    num_pos = pos_map.sum()
                    pos_idx = pos_map.nonzero()
                    grid = torch.arange(0, num_pos, step=num_pos//n_points)[:n_points]
                    coords = pos_idx[grid]

            select_coords.append(coords)
        select_coords = torch.stack(select_coords).float()
        
        if gt_points is not None: # 引入标注点来进行dedetr特征query,注意这里面gt point是原图，并且是xy格式(需要flip之后除16)
#             select_coords = torch.cat([select_coords, (gt_points.unsqueeze(1).flip(-1) / 16).long()], dim=1)
            select_coords = torch.cat([select_coords, gt_points.unsqueeze(1).flip(-1)], dim=1)
            
        prototypes = idx_by_coords(vit_feat[None].permute(0,2,3,1).expand(select_coords.shape[0],-1,-1,-1), 
                                   (select_coords[..., 0] // 16).long(), 
                                   (select_coords[..., 1] // 16).long()).clone() # 这利用可以用roi align 插值来做提高精度
        # prototypes = point_align(vit_feat[None], select_coords, self.point_feat_extractor) # 这利用可以用roi align 插值来做提高精度
        return select_coords, prototypes
            
    def get_semantic_centers(self, 
                            map_cos_fg, 
                            map_cos_bg,
                            rois, 
                            vit_feat, 
                            pos_thr=0.35,
                            n_points_sampled=20,
                            merge_thr=0.85,
                            gt_points=None,
                            gt_labels=None,
                            ):
        vit_feat = vit_feat.clone().detach()
        map_cos_fg_corr = corrosion_batch(torch.where(map_cos_fg > pos_thr, 
                                                      torch.ones_like(map_cos_fg), 
                                                      torch.zeros_like(map_cos_fg))[None], corr_size=11)[0]
        
        fg_inter = map_cos_fg_corr # 后面直接是原图大小，为了保持baseline性能而已
        map_fg = torch.where(fg_inter > pos_thr, torch.ones_like(fg_inter), torch.zeros_like(fg_inter))
        sampled_points, point_feats = self.uniform_sample_grid(map_fg, vit_feat, rois, 
                                                               thr=pos_thr, n_points=n_points_sampled,
                                                               gt_points=gt_points)
        sampled_points = sampled_points.flip(-1) # 变xy为yx
        num_gt = sampled_points.size(0)
        fg_mask = map_fg.clone()
        
        # 后面用mean shift获得稳定的目标特征点
        fg_inter = F.interpolate(map_cos_fg_corr.unsqueeze(0), vit_feat.shape[-2:], mode='bilinear')[0]
        bg_inter = F.interpolate(map_cos_bg.unsqueeze(0).max(dim=1, keepdim=True)[0], vit_feat.shape[-2:], mode='bilinear')[0]
        map_fg = torch.where(fg_inter > pos_thr, torch.ones_like(fg_inter), torch.zeros_like(fg_inter))
        
        # disable mlflow.sklearn 主要可能再微软的机器上难以跑通
        mlflow.sklearn.autolog(disable=True)
        pca = PCA(n_components=self.pca_dim)
        pca.fit(vit_feat.flatten(1).permute(1, 0).cpu().numpy())
        vit_feat_pca = torch.from_numpy(pca.fit_transform(vit_feat.flatten(1).permute(1, 0).cpu().numpy())).to(vit_feat.device)
        vit_feat_pca = vit_feat_pca.permute(1, 0).unflatten(1, vit_feat.shape[-2:])
        mlflow.sklearn.autolog(disable=False)
        
        # vit_feat_pca = vit_feat.clone()
        map_fg = torch.where(fg_inter > pos_thr, torch.ones_like(fg_inter), torch.zeros_like(fg_inter))
        prototypes_fg, sim_fg = self.mean_shift_grid_prototype(map_fg, vit_feat_pca, rois, 
                                                               tau=0.1, temp=0.1, 
                                                               n_shift=self.meanshift_refine_times, 
                                                               n_points=self.num_points_for_meanshift)
        sim_fg, idx_pos = filter_maps(sim_fg.unflatten(0, (sim_fg.shape[0]//self.num_points_for_meanshift,
                                                           self.num_points_for_meanshift)), fg_inter, bg_inter)
        split_size = idx_pos.sum(dim=-1).tolist()
        prototypes_fg = merge_maps(prototypes_fg[idx_pos.flatten()].split(split_size, dim=0), thr=merge_thr)
        # prototypes_fg = torch.split(prototypes_fg, map_fg.shape[0], dim=0)
        sim_fg = [cal_similarity(prot, vit_feat_pca.permute(1,2,0)) for prot in prototypes_fg]
        _, coord_semantic_center_split, part_masks_split = get_center_coord(sim_fg, rois, 
                                                          gt_labels, num_max_obj=self.num_semantic_points, 
                                                                           map_thr=self.map_thr)
        # 生成 visable weights 和 coords
        visible_weights = torch.zeros(num_gt, self.num_semantic_points).to(sampled_points.device).long()
        semantic_points = -1e4 * torch.ones(num_gt, self.num_semantic_points, 2).to(sampled_points.device)
        part_based_masks = torch.zeros(num_gt, self.num_semantic_points, *vit_feat.shape[-2:]).to(sampled_points.device).bool()
        if len(coord_semantic_center_split) == 0: # 一个semantic center都没有
            pass
        else:
            assert num_gt == len(coord_semantic_center_split)
            for i_gt, (points, part_masks) in enumerate(zip(coord_semantic_center_split, part_masks_split)):
                num_vis = points.size(0)
                visible_weights[i_gt][:num_vis] = 1
                semantic_points[i_gt][:num_vis] = points
                part_based_masks[i_gt][:num_vis] = part_masks
                
#         gt_visible_weights = torch.ones(num_gt, 1).long().to(sampled_points.device)
#         visible_weights = torch.cat([gt_visible_weights, visible_weights], dim=1)
#         semantic_points = torch.cat([gt_points.unsqueeze(1), semantic_points], dim=1)
        return sampled_points, point_feats, visible_weights, semantic_points, part_based_masks, fg_mask
    
    def mean_shift_grid_prototype(self, maps, vit_feat, rois=None, 
                                  thr=0.35, n_shift=5, output_size=(4,4), 
                                  tau=0.1, temp=0.1, n_points=20):
        # TODO: debug
        # get prototypes of earch instance
        # maps = F.interpolate(maps[None], scale_factor=1/16, mode='bilinear')[0]
        # vit_feat_pca = torch.pca_lowrank(vit_feat.flatten(1).permute(1, 0), q=64)

        prototypes = []
        select_coords = []
        for i_obj, map_ in enumerate(maps):
            pos_map = map_ >= thr
            num_pos = pos_map.sum()
            pos_idx = pos_map.nonzero()
            if num_pos >= n_points:
                grid = torch.arange(0, num_pos, step=num_pos//n_points)[:n_points]
                coords = pos_idx[grid]
            elif num_pos > 0:
                coords = pos_idx
                coords = fill_in_idx(coords, n_points)

            else: # num_pos == 0
                if rois is not None:
                    coords = ((rois[i_obj][:2] + rois[i_obj][2:]) // (2 * 16)).long().view(1, 2).flip(1)
                    coords = coords.repeat(n_points, 1)
                else:
                    pos_map = map_ >= 0
                    num_pos = pos_map.sum()
                    pos_idx = pos_map.nonzero()
                    grid = torch.arange(0, num_pos, step=num_pos//n_points)[:n_points]
                    coords = pos_idx[grid]

            select_coords.append(coords)
        select_coords = torch.stack(select_coords)
        prototypes = idx_by_coords(vit_feat[None].permute(0,2,3,1).expand(select_coords.shape[0],-1,-1,-1), 
                                   select_coords[..., 0], select_coords[..., 1]).clone()
        prot_objs = []
        sims_objs = []
        if rois is not None:
            maps_bbox = box2mask(rois//16, vit_feat.shape[-2:], default_val=0)
            prototypes, sim = cosine_shift_batch(prototypes.clone(), 
                                                 (vit_feat[None] * maps_bbox[:, None]).flatten(-2).transpose(1, 2).clone(),
                                                 vit_feat.flatten(-2).transpose(0,1).clone(), 
                                                 tau=tau, temp=temp, n_shift=n_shift)
        else:
            prototypes, sim = cosine_shift_self(prototypes[0], 
                                                (vit_feat).flatten(-2).transpose(0,1).clone(),
                                                vit_feat.flatten(-2).transpose(0,1).clone(),
                                                tau=tau, n_shift=n_shift)
        return prototypes, sim.unflatten(-1, vit_feat.shape[-2:]).clamp(0)
            
    def instance_attention_generation(self, 
                                  attn, # num_gt, 1, img_h, img_w
                                  rois, # num_gt, 4
                                  vit_feat, # C, H, W
                                  vit_feat_be_norm, # C, H, W
                                  pos_thr=0.6, #
                                  neg_thr=0.1, 
                                  num_gt=20, 
                                  corr_size=21, # 腐蚀的kernel 大小
                                  refine_times=2, 
                                  obj_tau=0.85, 
                                  gt_points=None):
        num_points = attn.shape[0]
        attn_map = attn.detach().clone()[:, 0]
        img_h, img_w = attn_map.shape[-2:]
        map_cos_fg, map_cos_bg, points_bg, points_fg = get_cosine_similarity_refined_map(attn_map, 
                                                                                         vit_feat_be_norm, 
                                                                                         rois, 
                                                                                         thr_pos=pos_thr, 
                                                                                         thr_neg=neg_thr, 
                                                                                         num_points=num_gt, 
                                                                                         thr_fg=0.7, 
                                                                                         refine_times=refine_times, 
                                                                                         obj_tau=obj_tau,
                                                                                         gt_points=gt_points,
                                                                                         point_feat_extractor=self.point_feat_extractor
                                                                                        )
        inter_bg_masks = []
        bg_masks = []
        neg_coords = []
        coord_chosen = []
        label_chosen = []
        num_objs = map_cos_fg[0].shape[0]
        for i_p, map_fg, map_bg in zip(range(num_objs), map_cos_fg[-1], map_cos_bg[-1]):
            xmin, ymin, xmax, ymax = rois[i_p].int().tolist()
            H, W = map_fg.shape[-2:]
            map_crop_fg = map_fg[ymin:ymax, xmin:xmax]
            map_crop_bg = map_bg[ymin:ymax, xmin:xmax]
            # 将bg map过thr,并且在box内mask置于False，最后取多个bg mask的与
            bg_masks_ = (map_bg > map_bg.max() * neg_thr)
            bg_masks_[ymin:ymax, xmin:xmax] = False
            bg_masks.append(bg_masks_)
            # 将inter bg 提取出来
            inter_bg_masks_ = torch.zeros_like(bg_masks_)
            inter_bg_masks_[ymin:ymax, xmin:xmax] = (map_crop_bg > neg_thr)
            inter_bg_masks.append(inter_bg_masks_)
            
            # 只用来选反例
            coor = get_mask_points_single_box_cos_map_bg(map_crop_fg, map_crop_bg,
                                                            pos_thr=pos_thr, neg_thr=neg_thr, 
                                                            num_gt=self.iam_num_points_init_neg, i=i_p, 
                                                            corr_size=corr_size)
            coor[:, 0] += ymin
            coor[:, 1] += xmin
            coor = coor.flip(1)
            neg_coords.append(coor)
            
            # 随机选择正反例
            coor, label = get_mask_points_single_box_cos_map_fg_bg(map_crop_fg, map_crop_bg, pos_thr=pos_thr, neg_thr=neg_thr, num_gt=self.num_random_iam_points, i=i_p, corr_size=corr_size)
            coor[:, 0] += ymin
            coor[:, 1] += xmin
            coor = coor.flip(1)
            coord_chosen.append(coor)
            label_chosen.append(label)
        coords_chosen = torch.stack(coord_chosen).float()
        labels_chosen= torch.stack(label_chosen)
        
        neg_coords = torch.stack(neg_coords).float() # 图像尺度大小xy格式        
        neg_prototypes = idx_by_coords(vit_feat[None].permute(0,2,3,1).expand(neg_coords.shape[0],-1,-1,-1), 
                                   (neg_coords[..., 1] // 16).long(), # 特征图尺度大小的点的y
                                   (neg_coords[..., 0] // 16).long()).clone() # 特征图尺度大小的点的x
        # 最后取多个bg mask的与
        bg_mask_exclue_bbox = bg_masks[0] # img_h, img_w
        for mask in bg_masks[1:]:
            bg_mask_exclue_bbox &= mask
        # 选择bg point来进行反例监督
        all_num_negs = num_objs * self.num_neg_sample
        all_neg_points_outer = bg_mask_exclue_bbox.nonzero(as_tuple=False).flip(-1)
        all_rand_neg_points_outer = []
        for _ in range(self.num_semantic_points + 1): # part + root 所有随机点
            neg_points_outer_index = torch.randperm(all_neg_points_outer.shape[0])[:all_num_negs].to(bg_mask_exclue_bbox.device)
            neg_points_outer = all_neg_points_outer[neg_points_outer_index]
            all_rand_neg_points_outer.append(neg_points_outer)
        all_rand_neg_points_outer = torch.stack(all_rand_neg_points_outer)
        return neg_coords, neg_prototypes, map_cos_fg, map_cos_bg, points_bg, points_fg, \
            bg_mask_exclue_bbox, all_rand_neg_points_outer, torch.stack(inter_bg_masks), coords_chosen, labels_chosen
    
# 用于做instance attention map的函数
def corrosion(cam, corr_size=11):
    if cam.ndim < 4:
        H, W = cam.shape[-2:]
        cam = cam.view(1, 1, H, W)
        return -F.max_pool2d(-cam, corr_size, 1, corr_size//2).reshape(H, W)
    return -F.max_pool2d(-cam, corr_size, 1, corr_size//2).reshape(H, W)

def corrosion_batch(cam, corr_size=11):
    return -F.max_pool2d(-cam, corr_size, 1, corr_size//2)

def norm_attns(attns):
    N, H, W = attns.shape
    max_val, _ = attns.view(N,-1,1).max(dim=1, keepdim=True)
    min_val, _ = attns.view(N,-1,1).min(dim=1, keepdim=True)
    return (attns - min_val) / (max_val - min_val)

def decouple_instance(map_bg, map_fg):
    map_bg = normalize_map(map_bg)
    map_fg = normalize_map(map_fg)
    map_fack_bg = 1 - (map_fg*0.5 + map_bg*0.5)
    return map_bg + map_fack_bg

def box2mask(bboxes, img_size, default_val=0.5):
    N = bboxes.shape[0]
    mask = torch.zeros(N, img_size[0], img_size[1], device=bboxes.device, dtype=bboxes.dtype) + default_val
    for n in range(N):
        box = bboxes[n]
        mask[n, int(box[1]):int(box[3]+1), int(box[0]):int(box[2]+1)] = 1.0
    return mask

def normalize_map(map_):
    max_val = map_.flatten(-2).max(-1,keepdim=True)[0].unsqueeze(-1)
    map_ = (map_ / (max_val + 1e-8))
    return map_

def fill_in_idx(idx_chosen, num_gt):
    assert idx_chosen.shape[0] != 0, '不能一个点都不选!'
    if idx_chosen.shape[0] >= num_gt / 2:
        idx_chosen = torch.cat((idx_chosen, idx_chosen[:num_gt-idx_chosen.shape[0]]), dim=0)
    else:
        repeat_times = num_gt // idx_chosen.shape[0]
        idx_chosen = idx_chosen.repeat(repeat_times, 1)
        idx_chosen = fill_in_idx(idx_chosen, num_gt)
    return idx_chosen

def idx_by_coords(map_, idx0, idx1, dim=0):
    # map.shape: N, H, W, C
    # idx0.shape: N, k
    # idx1.shape: N, k
    N = idx0.shape[0]
    k = idx0.shape[-1]
    idx_N = torch.arange(N, dtype=torch.long, device=idx0.device).unsqueeze(1).expand_as(idx0)
    idx_N = idx_N.flatten(0)
    return map_[idx_N, idx0.flatten(0), idx1.flatten(0)].unflatten(dim=dim, sizes=[N, k])

def point_align(map_, coords, point_feat_extractor): # coords 是 yx的格式 # 1, C, H, W    num_gt, num_point, 2  
    # output num_gt, num_point, C
    num_gt, num_points = coords.size()[:2]
    coords = coords.flip(-1).reshape(-1, 2)
    point_boxes = torch.cat([coords, coords + 16], dim=-1).float()
    rois = bbox2roi([point_boxes.reshape(-1, 4)])
    point_feats = point_feat_extractor(
                [map_][:point_feat_extractor.num_inputs], rois)
    point_feats = point_feats.reshape(num_gt, num_points, -1)
    return point_feats

def update_density_batch(prototypes, feats, mask_weight):
    similarity = F.cosine_similarity(prototypes[:, :, None], feats[:, None], dim=-1)
    density =(similarity * mask_weight).sum(-1)
    density = 1 - torch.where(mask_weight.sum(-1)>=1, density / mask_weight.sum(-1), torch.zeros_like(density))
    return density.clamp(1e-10).unsqueeze(-1)

def sample_point_grid(maps, num_points=10, thr=0.2, is_pos=False, gt_points=None):
    ret_coords = []
    for i_obj, map_ in enumerate(maps):
        factor = 1.0
        if is_pos:
            coords = (map_ >= thr*factor).nonzero(as_tuple=False).view(-1, 2)
        else:
            coords = (map_ < thr*factor).nonzero(as_tuple=False).view(-1, 2)
        # coords = coords[:0]
        num_pos_pix = coords.shape[0] 
        
        if num_pos_pix < num_points:
            if is_pos:
                coords_chosen = torch.cat((coords, gt_points[i_obj].repeat(num_points-num_pos_pix, 1)), dim=0)
                ret_coords.append(coords_chosen)
                continue
            else:
                while num_pos_pix < num_points:
                    # print(f'factor adjusted from {thr * factor} to {thr * factor * 2}')
                    factor *= 2
                    coords = (map_ < thr*factor).nonzero(as_tuple=False)
                    num_pos_pix = coords.shape[0]

        step = num_pos_pix // num_points
        idx_chosen = torch.arange(0, num_pos_pix, step=step)
        idx_chosen = torch.randint(num_pos_pix, idx_chosen.shape) % num_pos_pix
        coords_chosen = coords[idx_chosen][:num_points]
        ret_coords.append(coords_chosen)
    return torch.stack(ret_coords).flip(-1)

def cal_similarity(prototypes, feat, dim=-1):
    if isinstance(prototypes, list):
        return torch.zeros(0, 0)
    sim = F.cosine_similarity(prototypes[:, None, None,:], feat[None], dim=-1)
    return sim

def get_refined_similarity(point_coords, feats, bboxes, ratio=1, refine_times=1, tau=0.85, is_select=False, point_feat_extractor=None):
    cos_map = get_point_cos_similarity_map(point_coords, feats, ratio=ratio, point_feat_extractor=point_feat_extractor)
    # fg_map = cos_map.argmax(dim=0, keepdim=True)  < bboxes.shape[0]
    # cos_map *= fg_map
    cos_map1 = cos_map.clone()
    cos_rf = []
    bbox_mask = box2mask(bboxes//16, cos_map.shape[-2:], default_val=0)
    # cos_map[:bboxes.shape[0]] = cos_map[:bboxes.shape[0]] * bbox_mask
    if is_select:
        # cos_map_select = torch.where(idx_max_aff==range_obj[:,None,None], cos_map, torch.zeros_like(cos_map))
        cos_map[:bboxes.shape[0]] = cos_map[:bboxes.shape[0]] * bbox_mask
        # max_val = cos_map.flatten(1).max(1, keepdim=True)[0].unsqueeze(-1)
        # cos_map = cos_map / (max_val + 1e-8)
        idx_max_aff = cos_map.argmax(0, keepdim=True).expand_as(cos_map)
        range_obj = torch.arange(cos_map.shape[0], device=cos_map.device)
        cos_rf.append(torch.where(idx_max_aff==range_obj[:,None,None], cos_map.clone(), torch.zeros_like(cos_map)))
    else:
        cos_rf.append(cos_map.clone())

    for i in range(refine_times):
        # fg_map = cos_map1.argmax(dim=0, keepdim=True)  < bboxes.shape[0]
        # cos_map1 *= fg_map
        # cos_map1[:bboxes.shape[0]] = bbox_mask * cos_map1[:bboxes.shape[0]]
        max_val = cos_map1.flatten(1).max(1, keepdim=True)[0].unsqueeze(-1)
        thr = max_val * tau
        cos_map1[cos_map1 < thr] *= 0
        feats_mask = feats * cos_map1.unsqueeze(1)
        feats_mask = feats_mask.sum([2,3], keepdim=True) / (cos_map1.unsqueeze(1).sum([2,3], keepdim=True).clamp(1e-8))
        cos_map1 = F.cosine_similarity(feats, feats_mask, dim=1)
        if is_select:
            # cos_map_select = torch.where(idx_max_aff==range_obj[:,None,None], cos_map1, torch.zeros_like(cos_map1))
            # cos_map1[:bboxes.shape[0]] = bbox_mask * cos_map1[:bboxes.shape[0]]
            cos_map1[:bboxes.shape[0]] = cos_map1[:bboxes.shape[0]] * bbox_mask
            idx_max_aff = cos_map1.argmax(0, keepdim=True).expand_as(cos_map1)
            range_obj = torch.arange(cos_map1.shape[0], device=cos_map1.device)
            cos_rf.append(torch.where(idx_max_aff==range_obj[:,None,None], cos_map1.clone(), torch.zeros_like(cos_map1)))
        else:
            cos_rf.append(cos_map1.clone())

    return torch.stack(cos_rf)

def get_point_cos_similarity_map(point_coords, feats, ratio=1, point_feat_extractor=None):
    feat_expand = feats.permute(0,2,3,1).expand(point_coords.shape[0], -1, -1, -1)
    point_feats = idx_by_coords(feat_expand, 
                                (point_coords[...,1] // 16 * ratio).long().clamp(0, feat_expand.shape[1]),
                                (point_coords[...,0] // 16 * ratio).long().clamp(0, feat_expand.shape[2]))
    # point_feats = point_align(feats, point_coords, point_feat_extractor)
    point_feats_mean = point_feats.mean(dim=1, keepdim=True)
    sim = F.cosine_similarity(feat_expand.flatten(1,2), point_feats_mean, dim=2)
    # sim = torch.cdist(feat_expand.flatten(1,2), point_feats_mean, p=2).squeeze(-1)
    return sim.unflatten(1, (feat_expand.shape[1], feat_expand.shape[2]))

def get_cosine_similarity_refined_map(attn_maps, vit_feat, bboxes, 
                                      thr_pos=0.2, thr_neg=0.1, num_points=20, 
                                      thr_fg=0.7, refine_times=1, obj_tau=0.85, 
                                      gt_points=None, point_feat_extractor=None):
    # attn_maps是上采样16倍之后的，vit_feat是上采样前的，实验表明，上采样后的不太好，会使cos_sim_fg ~= cos_sim_bg
    attn_norm = norm_attns(attn_maps)
    points_bg = sample_point_grid(attn_norm, thr=thr_neg, num_points=num_points)
    points_fg = sample_point_grid(attn_norm, thr=thr_pos, num_points=num_points, is_pos=True, gt_points=gt_points)
    points_bg_supp = sample_point_grid(attn_norm.mean(0, keepdim=True), thr=thr_neg, num_points=num_points)
    # points_bg_supp = torch.cat([sample_point_grid(attn_norm[0].mean(0,keepdim=True)<thr_neg, num_points=num_points) for _ in range(3)],dim=0)
    points_fg = torch.cat((points_fg, points_bg_supp), dim=0)
    cos_sim_fg = F.interpolate(get_refined_similarity(points_fg, vit_feat[None], bboxes=bboxes, refine_times=refine_times, tau=obj_tau, is_select=True, point_feat_extractor=point_feat_extractor), attn_maps.shape[-2:], mode='bilinear')[:,:attn_norm.shape[0]]
    cos_sim_bg = F.interpolate(get_refined_similarity(points_bg, vit_feat[None], bboxes=bboxes, refine_times=refine_times, tau=obj_tau, point_feat_extractor=point_feat_extractor), attn_maps.shape[-2:], mode='bilinear')
    ret_map = (1 - cos_sim_bg) * cos_sim_fg
    map_val = ret_map.flatten(-2, -1).max(-1, keepdim=True)[0].unsqueeze(-1).clamp(1e-8)
    
    cos_sim_bg = decouple_instance(cos_sim_bg.clone(), ret_map.clone())
    max_val_bg = cos_sim_bg.flatten(-2, -1).max(-1, keepdim=True)[0].unsqueeze(-1).clamp(1e-8)
#    map_fg = torch.where(ret_map < map_val * thr_fg, torch.zeros_like(ret_map), torch.ones_like(ret_map))
    # map_bg = torch.where(ret_map > map_val * 0.1, torch.zeros_like(ret_map), torch.ones_like(ret_map))
    return ret_map / map_val, cos_sim_bg / max_val_bg, points_fg, points_bg

def cosine_shift_batch(prototypes, feats, feats_org=None, tau=0.1, temp=0.1, n_shift=5):
    # prototypes.shape: n_obj, n_block, n_dim
    # feat.shape: n_patches, n_dim
    for i_s in range(n_shift):
        sim_map = F.cosine_similarity(prototypes[:, :, None], feats[:, None], dim=-1)
        weight = F.softmax(sim_map/(temp*tau), dim=-1)
        feat_idx = weight.argmax(1, keepdim=True)
        prot_range = torch.arange(prototypes.shape[1], device=feat_idx.device, dtype=feat_idx.dtype)[None, :, None].expand(prototypes.shape[0], prototypes.shape[1], -1)
        mask_weight = torch.where(prot_range==feat_idx, torch.ones_like(weight), torch.zeros_like(weight))
        prototypes = torch.matmul(weight * mask_weight, feats)
        tau = update_density_batch(prototypes, feats, mask_weight)
    # prototypes = merge_pototypes(prototypes, thr=1-tau)
    if feats_org is not None:
        sim_map = F.cosine_similarity(prototypes[:, :, None, :], feats_org[None, None, :, :], dim=-1)
    else:
        sim_map = F.cosine_similarity(prototypes[:, :, None, :], feats[None, None, :, :], dim=-1)
    # sim_map = F.cosine_similarity(prototypes[:, None, :], feats[None, :, :], dim=-1)
    # weight = F.softmax(sim_map/(tau*0.1), dim=-1)
    return prototypes.flatten(0,1), sim_map.flatten(0,1)

def get_mask_points_single_box_cls_fg_bg(map_fg, map_bg, num_gt=10):
    device = map_fg.device
    # attn_pos = corrosion(attn_map.float(), corr_size=corr_size)
    coord_pos = map_fg.nonzero(as_tuple=False)
    coord_neg = map_bg.nonzero(as_tuple=False)
    coord_pos_neg = torch.cat((coord_pos, coord_neg), dim=0)
    # print(f'coord_pos.shape[0] / coord_neg.shape[0]: {coord_pos.shape[0] / coord_neg.shape[0]}')
    idx_chosen = torch.randperm(coord_pos_neg.shape[0])[:num_gt].to(device)
    labels_pos_neg = torch.cat((torch.ones(coord_pos.shape[0], device=device, dtype=torch.bool),
                                torch.zeros(coord_neg.shape[0], device=device, dtype=torch.bool)), dim=0)
    if idx_chosen.shape[0] < num_gt:
        if idx_chosen.shape[0] == 0:
            coords_chosen = -torch.ones(num_gt, 2, dtype=torch.float, device=device)
            print(f'**************一个点都没有找到!**************')
            # 这些-1的点会在point ignore里被处理掉
            return coords_chosen, torch.zeros(num_gt, dtype=torch.bool, device=device)
        else:
            idx_chosen = fill_in_idx(idx_chosen, num_gt)
    coords_chosen = coord_pos_neg[idx_chosen]
    labels_chosen = labels_pos_neg[idx_chosen]

    return coords_chosen, labels_chosen

def get_mask_points_single_box_cos_map_fg_bg(map_fg, map_bg, pos_thr=0.6, neg_thr=0.6, num_gt=10, i=0, corr_size=21):
    # Parameters:
    #     coords: num_pixels, 2
    #     attn:H, W
    #     cls: scalar,
    # Return:
    #     coords_chosen: num_gt, 2
    #     labels_chosen: num_gt
    device = map_fg.device
    # attn_pos = corrosion(attn_map.float(), corr_size=corr_size)
    coord_pos = corrosion((map_fg > map_fg.max()*pos_thr).float(), corr_size=corr_size).nonzero(as_tuple=False)
    coord_neg = (map_bg > map_bg.max()*neg_thr).nonzero(as_tuple=False)
    coord_pos_neg = torch.cat((coord_pos, coord_neg), dim=0)
    # print(f'coord_pos.shape[0] / coord_neg.shape[0]: {coord_pos.shape[0] / coord_neg.shape[0]}')
    idx_chosen = torch.randperm(coord_pos_neg.shape[0])[:num_gt].to(device)
    labels_pos_neg = torch.cat((torch.ones(coord_pos.shape[0], device=device, dtype=torch.bool),
                                torch.zeros(coord_neg.shape[0], device=device, dtype=torch.bool)), dim=0)
    if idx_chosen.shape[0] < num_gt:
        if idx_chosen.shape[0] == 0:
            coords_chosen = -torch.ones(num_gt, 2, dtype=torch.float, device=device)
            print(f'**************一个点都没有找到!**************')
            # 这些-1的点会在point ignore里被处理掉
            return coords_chosen, torch.zeros(num_gt, dtype=torch.bool, device=device)
        else:
            idx_chosen = fill_in_idx(idx_chosen, num_gt)
    coords_chosen = coord_pos_neg[idx_chosen]
    labels_chosen = labels_pos_neg[idx_chosen]

    return coords_chosen, labels_chosen

# 只用来选择反例
def get_mask_points_single_box_cos_map_bg(map_fg, map_bg, pos_thr=0.6, neg_thr=0.6, num_gt=10, i=0, corr_size=21):
    # Parameters:
    #     coords: num_pixels, 2
    #     attn:H, W
    #     cls: scalar,
    # Return:
    #     coords_chosen: num_gt, 2
    #     labels_chosen: num_gt
    device = map_fg.device
    # attn_pos = corrosion(attn_map.float(), corr_size=corr_size)
    # coord_pos = corrosion((map_fg > map_fg.max()*pos_thr).float(), corr_size=corr_size).nonzero(as_tuple=False)
    coord_neg = (map_bg > map_bg.max()*neg_thr).nonzero(as_tuple=False)
    # coord_pos_neg = torch.cat((coord_pos, coord_neg), dim=0)
    # print(f'coord_pos.shape[0] / coord_neg.shape[0]: {coord_pos.shape[0] / coord_neg.shape[0]}')
    # idx_chosen = torch.randperm(coord_pos_neg.shape[0], device=device)[:num_gt]
    idx_chosen = torch.randperm(coord_neg.shape[0], device=device)[:num_gt]
    # labels_pos_neg = torch.cat((torch.ones(coord_pos.shape[0], device=device, dtype=torch.bool),
    #                             torch.zeros(coord_neg.shape[0], device=device, dtype=torch.bool)), dim=0)
    if idx_chosen.shape[0] < num_gt:
        if idx_chosen.shape[0] == 0:
            coords_chosen = -torch.ones(num_gt, 2, dtype=torch.float, device=device)
            # print(f'**************一个点都没有找到!**************')
            # 这些-1的点会在point ignore里被处理掉
            # return coords_chosen, torch.zeros(num_gt, dtype=torch.bool, device=device)
            return coords_chosen
        else:
            idx_chosen = fill_in_idx(idx_chosen, num_gt)
    coords_chosen = coord_neg[idx_chosen]
    # labels_chosen = labels_pos_neg[idx_chosen]

    return coords_chosen #, labels_chosen


def filter_maps(maps, pos_maps, neg_maps, pos_thr=0.85, neg_thr=0.8):
    map_pos = []
    map_neg = []
    maps_fore = torch.where(maps>0.8, torch.ones_like(maps), torch.zeros_like(maps))
    
    pos_score = (pos_maps[:, None] * maps_fore).sum(dim=[-2, -1]) / maps_fore.sum(dim=[-2, -1]).clamp(1e-6)
    neg_score = (neg_maps[:, None] * maps_fore).sum(dim=[-2, -1]) / maps_fore.sum(dim=[-2, -1]).clamp(1e-6)
    pos_idx = (pos_score >= pos_thr)
    # neg_idx = (neg_score >= neg_thr) & (pos_score < 0.5)
    split_size = pos_idx.sum(dim=-1).tolist()
    maps_fore = maps.flatten(0,1)[pos_idx.flatten()].split(split_size, dim=0)
    # maps_back = maps.flatten(0,1)[neg_idx.flatten()]
    return maps_fore, pos_idx

def merge_maps(prototypes, thr=0.95):
    prot_ret = []
    for prot in prototypes:
        prot_obj = []
        if prot.shape[0] == 0:
            prot_ret.append([])
            continue
        sim = F.cosine_similarity(prot[None], prot[:, None], dim=-1)
        sim_triu = torch.where(torch.triu(sim, diagonal=0) >= thr, torch.ones_like(sim), torch.zeros_like(sim))
        for i_p in range(sim_triu.shape[0]):
            weight = sim_triu[i_p]
            if weight.sum() > 0:
                prot_merge = torch.matmul(weight, prot) / (weight.sum() + 1e-8)
                prot_obj.append(prot_merge)
            sim_triu[weight>0] *= 0 
        prot_ret.append(torch.stack(prot_obj))
    return prot_ret

def get_center_coord(maps, rois, obj_label, num_max_keep=50, num_max_obj=3, map_thr=0.9):
    part_masks = []
    coords = []
    labels = []
    split_size = [0 for _ in range(len(maps))]
    for i_obj, map_ in enumerate(maps):
        if map_.shape[0] == 0:
            continue
        top2 = map_.flatten(1).topk(dim=1, k=1)[0][:, -1, None, None]
        coord_top2 = (map_ >= top2).nonzero().float() # 可能出现多个个极值点的
        xmin, ymin, xmax, ymax = rois[i_obj]
        label = obj_label[i_obj]
        map_area_idxsort = (map_>map_thr).sum(dim=[-2,-1]).argsort(descending=True, dim=0)
        for i_prot in range(map_.shape[0]):
            if i_prot == num_max_obj - 1:
                break
            coord = (coord_top2[coord_top2[:, 0]==map_area_idxsort[i_prot]].mean(dim=0)[1:].flip(0) + 0.5) * 16 # patch坐标应该位于中心， 上采样16倍
            part_mask = map_[(coord_top2[:, 0]==map_area_idxsort[i_prot])[:map_.shape[0]]].mean(dim=0) > map_thr
            if (coord[0] >= xmin) & (coord[0] <= xmax) & (coord[1] >= ymin) & (coord[1] <= ymax):
                part_masks.append(part_mask)
                coords.append(coord)
                labels.append(label)
                split_size[i_obj] += 1

    if len(coords) == 0:
        return (torch.zeros(0, 2, dtype=rois[0].dtype, device=rois[0].device), torch.zeros(0, dtype=obj_label[0].dtype, device=obj_label[0].device)), [], []
    else:
        coords = torch.stack(coords)
        labels = torch.stack(labels)
        part_masks = torch.stack(part_masks)
        coord_split = coords.split(split_size, dim=0) # coord_split是用来监督Mask Decoder的，所以不需要有数量限制
        masks_split = part_masks.split(split_size, dim=0)
        if coords.shape[0] > num_max_keep:
            idx_chosen = torch.randperm(coords.shape[0], device=coords.device)[:num_max_keep]
            coords = coords[idx_chosen]
            labels = labels[idx_chosen]
        return (coords, labels), coord_split, masks_split