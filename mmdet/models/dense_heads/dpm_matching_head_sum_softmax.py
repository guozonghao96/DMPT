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


@HEADS.register_module()
class DPMMatchingHead(nn.Module):

    def __init__(self,
                 # part based model setting 
                 in_channels=256,
                 hidden_channels=1024,
                 out_channels=256,
                 num_classes=20,
                 deformable_cost_weight=0.2,
                 semantic_deform=False,
                 num_classifier=5,
                 bce_loss=dict(
                     type='MILLoss',
                     use_sigmoid=True, # BCE loss 
                     reduction='mean',
                     loss_weight=1.0),
                 aux_bce_loss=dict(
                     type='MILLoss',
                     use_sigmoid=True, # BCE loss 
                     reduction='mean',
                     loss_weight=1.0),
                 # init point loc setting 
                 iam_num_points_init=5, # number of neg and pos sampled points. All number is 2xiam_num_points_init
                 iam_thr_pos=0.35, 
                 iam_thr_neg=0.8,
                 iam_refine_times=2, 
                 iam_obj_tau=0.9,
                 pca_dim=64,
                 meanshift_refine_times=5,
                 num_points_for_meanshift=20,
                 num_semantic_points=5,
                 # init point loc setting 
                 **kwargs):
        super(DPMMatchingHead, self).__init__()
        
        # instance attention map的参数
        self.iam_thr_pos = iam_thr_pos
        self.iam_thr_neg = iam_thr_neg
        self.iam_refine_times = iam_refine_times
        self.iam_obj_tau = iam_obj_tau
        self.iam_num_points_init = iam_num_points_init
        self.pca_dim = pca_dim
        self.meanshift_refine_times = meanshift_refine_times
        self.num_points_for_meanshift = num_points_for_meanshift
        self.point_feat_extractor = None
        self.with_gt_points = False
        self.num_semantic_points = num_semantic_points
        self.num_classifier = num_classifier
        self.num_classes = num_classes
        self.semantic_deform = semantic_deform
        # 
        self.deformable_cost_weight = deformable_cost_weight
        self.bce_loss = build_loss(bce_loss)
        self.aux_bce_loss = build_loss(aux_bce_loss)
        
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        
        self.root_filter = nn.Linear(out_channels, num_classes)
        self.part_filters = nn.ModuleList(
            nn.Linear(out_channels, num_classes) for _ in range(self.num_classifier))
        
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
        all_matchings = []
        for p in it.permutations(np.arange(num_semantic_points)): 
            all_matchings.append(np.array(p))
        all_matchings = torch.as_tensor(all_matchings).cuda()
        num_matchings = len(all_matchings)
        self.num_matchings = num_matchings
        self.all_row_matchings = all_matchings.clone().reshape(-1, 1, num_semantic_points).repeat(1, num_matchings, 1).reshape(-1, num_semantic_points)
        self.all_col_matchings = all_matchings.clone().reshape(1, -1, num_semantic_points).repeat(num_matchings, 1, 1).reshape(-1, num_semantic_points)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, x):
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        root_scores = self.root_filter(x) #.sigmoid()
        part_scores = []
        for filter_ in self.part_filters:
            scores = filter_(x)
            part_scores.append(scores)
        part_scores = torch.stack(part_scores).permute(1, 0, 2, 3) #.sigmoid()
        return root_scores, part_scores
    
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
        for i_img in range(num_imgs):
            # 第一个mean shift操作，后的instance attention map
            # refine次数，num_gt, img_h, img_w
            # self-attention map的前景点和背景点 -> num_gt, num_point, 2
            neg_points, neg_point_feats, map_cos_fg, map_cos_bg, \
                points_bg_attn, points_fg_attn = self.instance_attention_generation(matched_cams[i_img], 
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
                                                                    pos_thr=0.35,
                                                                    n_points_sampled=self.iam_num_points_init,
                                                                    gt_points=gt_points[i_img] if self.with_gt_points else None,
                                                                    gt_labels=gt_labels[i_img])
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
            fg_masks.append(fg_mask)
        
        attnshift_results = dict(
            pseudo_points=pseudo_points,
            pseudo_bin_labels=pseudo_bin_labels,
            all_visible_weights=all_visible_weights,
            all_semantic_points=all_semantic_points,
            all_part_based_masks=all_part_based_masks,
            fg_points=fg_points,
            instance_cos_fg_maps=instance_cos_fg_maps,
            fg_masks=fg_masks,
        )
        losses_dpm = dict()
        losses_dpm['dpm_bce_loss'] = 0
        # losses_dpm['dpm_aux_bce_loss'] = 0
        losses_dpm['dpm_acc'] = []
        # losses_dpm['dpm_aux_acc'] = []
        
        all_aligned_part_centers = []
        all_aligned_part_maps = []
        all_root_scores = []
        all_part_scores = []
        all_deformable_costs = []
        all_dpm_points = []
        all_dpm_visible = []
        part_proposals_maps = []
        
        # forward
        vit_feat_dpm = vit_feat.flatten(2).permute(0, 2, 1)
        root_scores, part_scores = self(vit_feat_dpm) # bs, N, 20 # bs, num_classifier, N, 20
        matching_score_maps = []
        instance_part_score_maps = []
        # calculate matched feature
        for i_img in range(num_imgs):
            labels = gt_labels[i_img]
            num_gt = len(labels)
            visible_weights = all_visible_weights[i_img] # num_gt, num_parts
            semantic_points = all_semantic_points[i_img] / 16 # num_gt, num_parts, 2
            part_based_masks = all_part_based_masks[i_img].long() # num_gt, num_parts, h, w
            
            root_mask = F.interpolate(fg_masks[i_img].unsqueeze(1), 
                                      (patch_h, patch_w),
                                      mode='bilinear').squeeze(1).long() # num_gt, h, w
            root_score = root_scores[i_img].reshape(*featmap_sizes[2], self.num_classes)
            part_score = part_scores[i_img].reshape(-1, *featmap_sizes[2], self.num_classes)
            
            # part offset
            
            if self.semantic_deform:
                part_deform = self.part_deform.reshape(1, self.num_classifier, 
                                        self.num_classes, 1, 1, 4).repeat(num_gt, 1, 1, 1, 1, 1)
                # num_gt, num_classifier, num_classes, 1, 1, 4
                part_deform = torch.gather(part_deform, 
                                          index=labels.reshape(-1, 1, 1, 1, 1, 1).repeat(1, self.num_classifier, 1, 1, 1, 4),
                                          dim=2)[:, :, 0, ...] # num_gt, num_classifier, 1, 1, 4
            else:
                part_deform = self.part_deform.reshape(1, self.num_classifier, 1, 1, 4) # num_gt, num_classifier, 1, 1, 4
                
            xx, yy = torch.meshgrid(torch.arange(patch_w).to(semantic_points.device), 
                                    torch.arange(patch_h).to(semantic_points.device))
            xx_norm, yy_norm = xx / patch_w, yy / patch_h
            norm_semantic_points = semantic_points / \
                    torch.as_tensor(featmap_sizes[2]).to(semantic_points.device).flip(0).reshape(1, 1, 2)
            
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
            # weighted_deform_cost = self.deformable_cost_weight * \
            #         (deform_cost * part_deform).sum(-1) # num_gt, num_parts, h, w
            weighted_deform_cost = self.deformable_cost_weight * deform_cost # num_gt, num_parts, h, w, 4
            all_deformable_costs.append(weighted_deform_cost.sum(-1))
            instance_root_score = root_score.unsqueeze(0).repeat(num_gt, 1, 1, 1) # num_gt, h, w, num_classes
            instance_part_score = part_score.unsqueeze(0).repeat(num_gt, 1, 1, 1, 1) # num_gt, num_classifier, h, w, num_classes
            # aux_instance_part_score = instance_part_score * root_mask.unsqueeze(1).unsqueeze(-1)
            # aux_instance_part_score = aux_instance_part_score.flatten(2, 3).sum(2)
            instance_part_score_maps.append(instance_part_score)
            
            # matching stretegy
            with torch.no_grad():
                part_matching_scores = (instance_part_score.unsqueeze(2) - 
                                        (part_deform.unsqueeze(2) * weighted_deform_cost.unsqueeze(1)).sum(-1).unsqueeze(-1)
                                       ) * part_based_masks.unsqueeze(1).unsqueeze(-1) # num_gt, num_classfier, num_parts, h, w, num_classes
                matching_score_maps.append(part_matching_scores)
                # 
                part_matching_scores = part_matching_scores.flatten(-3, -2).sum(-2) / (part_based_masks.unsqueeze(1).flatten(-2, -1).sum(-1).unsqueeze(-1) + 1)
                # num_gt, num_classifier, num_part, 20
                part_matching_scores = part_matching_scores[:, self.all_row_matchings.reshape(-1), self.all_col_matchings.reshape(-1), :].reshape(num_gt, self.num_matchings ** 2, self.num_semantic_points, self.num_classes)
                # num_gt, num_matchings ^ 2, num_parts(只是part 的索引index的维度), 20

                add_instance_scores = (instance_root_score * root_mask.unsqueeze(-1)).flatten(1, 2).sum(1).unsqueeze(1) / root_mask.flatten(1, 2).sum(1).unsqueeze(-1).unsqueeze(1) # num_gt, 1, 20
                # 寻找bce loss最小！=(分类分数最大)的那个匹配 xxx完全错，有反例的softmax才能用这个
                # 
                # num_gt, num_matchings ^ 2, 20
                part_matching_scores = (part_matching_scores.sum(-2) + 
                                        add_instance_scores).softmax(-1).reshape(-1, self.num_classes)
                # num_gt, num_matchings ^ 2
                matching_labels = labels.reshape(-1, 1).repeat(1, self.num_matchings ** 2).reshape(-1)
                matching_bce_losses = self.bce_loss(part_matching_scores, 
                                                    matching_labels).reshape(num_gt, -1, self.num_classes).mean(-1)
                # 取loss最小的
                matching_index = matching_bce_losses.min(1)[1]
                # 分类器匹配顺序 num_gt, num_parts
                matched_row_inds = self.all_row_matchings[matching_index]
                # part 特征匹配顺序 num_gt, num_parts
                matched_col_inds = self.all_col_matchings[matching_index]
                
            # part 分类器分数
            masked_instance_part_score = []
            masked_part_proposals_map = []
            aligned_part_masks = []
            aligned_part_centers = []
            for row_inds_, col_inds_, part_score_, deform_cost_, part_mask_, label_, part_centers_ in \
                zip(matched_row_inds, matched_col_inds, instance_part_score, 
                    weighted_deform_cost, part_based_masks.unsqueeze(-1),
                    labels, semantic_points):
                part_score_ = part_score_[row_inds_] # num_part, h, w, 20
                part_deform_ = part_deform[0][row_inds_] # num_part, 1, 1, 4
                deform_cost_ = deform_cost_[col_inds_] # num_part, h, w, 4
                part_mask_ = part_mask_[col_inds_] # num_part, h, w, 1
                final_part_score = (part_score_ - (part_deform_ * 
                                                   deform_cost_).sum(-1).unsqueeze(-1)) * part_mask_ # num_part, h, w, 20
                aligned_part_masks.append(part_mask_)
                aligned_part_centers.append(part_centers_[col_inds_] * 16)
                masked_part_proposals_map.append(
                    F.interpolate(((part_score_ - (part_deform_ * deform_cost_).sum(-1).unsqueeze(-1)).softmax(-1)
                                   * part_mask_).permute(0, 3, 1, 2), 
                                  size=(img_h, img_w),
                                  mode='bilinear')[:, label_, ...] # num_parts, H, W
                )
                # final_part_score 内部有负数/正数/0，因为我们mask乘在特征上，
                # 导致大部分是0，其实对于分类分数来说已经有贡献了，除非是-inf
                # num_part, num_classes
                # final_part_score_ = []
                # for mask_, score_ in zip(part_mask_, final_part_score):
                #     # 可能存在mask的0，直接取mean会产生inf
                #     if mask_.sum() == 0:
                #         mean_score = score_.reshape(-1, self.num_classes)[mask_.reshape(-1) > 0].sum(0)
                #     else:
                #         mean_score = score_.reshape(-1, self.num_classes)[mask_.reshape(-1) > 0].mean(0)
                #     final_part_score_.append(mean_score)
                # final_part_score_ = torch.stack(final_part_score_)
                # masked_instance_part_score.append(final_part_score_) # num_gt, num_part, num_classes
                masked_instance_part_score.append(
                    final_part_score.flatten(1, 2).sum(1) / (part_mask_.flatten(1, 2).sum(1) + 1)
                ) # num_parts, num_classes
                
            masked_part_proposals_map = torch.stack(masked_part_proposals_map) #num_gt, num_parts, H, W
            masked_instance_part_score = torch.stack(masked_instance_part_score) # num_gt, num_parts, num_classes
            aligned_part_masks = torch.stack(aligned_part_masks) # num_gt, num_parts, h, w
            aligned_part_centers = torch.stack(aligned_part_centers) # num_gt, num_parts, 2
            all_aligned_part_maps.append(aligned_part_masks)
            all_aligned_part_centers.append(aligned_part_centers)
            
            all_root_scores.append(instance_root_score)
            # root 分类器分数
            # masked_instance_root_score = []
            # for score_, mask_ in zip(instance_root_score, root_mask):
            #     # 可能存在mask的0，直接取mean会产生inf
            #     if mask_.sum() == 0:
            #         mean_score = score_.reshape(-1, self.num_classes)[mask_.reshape(-1) > 0].sum(0).unsqueeze(0)
            #     else:
            #         mean_score = score_.reshape(-1, self.num_classes)[mask_.reshape(-1) > 0].mean(0).unsqueeze(0)
            #     masked_instance_root_score.append(mean_score)
            # masked_instance_root_score = torch.stack(masked_instance_root_score) # num_gt, 1, 20    
            masked_instance_root_score = (instance_root_score * root_mask.unsqueeze(-1)).flatten(1, 2).sum(1).unsqueeze(1) / (root_mask.flatten(1).sum(1).unsqueeze(1).unsqueeze(-1) + 1) # num_gt, 1, 20
            # losses
            object_score = torch.cat([masked_instance_root_score, masked_instance_part_score], dim=1).sum(1)
            losses_dpm['dpm_bce_loss'] += self.bce_loss(object_score.softmax(-1), labels).mean() + instance_part_score.sum() * 0
            losses_dpm['dpm_acc'].append(accuracy(object_score.softmax(-1), labels))
            # losses_dpm['dpm_aux_bce_loss'] += self.aux_bce_loss(
            #     aux_instance_part_score.softmax(-1).reshape(-1, self.num_classes), 
            #     labels.reshape(-1, 1).repeat(1, self.num_classifier).reshape(-1)).mean() + instance_part_score.sum() * 0
            # losses_dpm['dpm_aux_acc'].append(accuracy(
            #     aux_instance_part_score.softmax(-1).reshape(-1, self.num_classes),
            #     labels.reshape(-1, 1).repeat(1, self.num_classifier).reshape(-1)
            # ))

            part_proposals_maps.append(masked_part_proposals_map)
            max_flatten_index = torch.argmax(masked_part_proposals_map.flatten(-2), dim=-1)
            y_coords = max_flatten_index // img_w + 0.5
            x_coords = max_flatten_index % img_w + 0.5
            part_locations = torch.stack([x_coords, y_coords], dim=-1)
            visible_weights = (aligned_part_masks.flatten(2).sum(-1) > 0)
            part_locations[~visible_weights] = -1000
            # 这边必须用invisible的点把目标的位置值给置-1000，用于后面的mask sup.的过滤
            all_dpm_points.append(part_locations)
            all_dpm_visible.append(visible_weights)
            
        losses_dpm['dpm_acc'] = torch.as_tensor(losses_dpm['dpm_acc']).to(device).mean()
        # losses_dpm['dpm_aux_acc'] = torch.as_tensor(losses_dpm['dpm_aux_acc']).to(device).mean()
        attnshift_results['all_dpm_points'] = all_dpm_points
        attnshift_results['all_dpm_visible'] = all_dpm_visible
        attnshift_results['matching_score_maps'] = matching_score_maps
        attnshift_results['instance_part_score_maps'] = instance_part_score_maps
        attnshift_results['part_proposals_maps'] = part_proposals_maps
        attnshift_results['all_aligned_part_maps'] = all_aligned_part_maps
        attnshift_results['all_deformable_costs'] = all_deformable_costs
        attnshift_results['all_aligned_part_centers'] = all_aligned_part_centers
        attnshift_results['all_root_scores'] = all_root_scores
        
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
        pca = PCA(n_components=self.pca_dim)
        pca.fit(vit_feat.flatten(1).permute(1, 0).cpu().numpy())
        vit_feat_pca = torch.from_numpy(pca.fit_transform(vit_feat.flatten(1).permute(1, 0).cpu().numpy())).to(vit_feat.device)
        vit_feat_pca = vit_feat_pca.permute(1, 0).unflatten(1, vit_feat.shape[-2:])
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
                                                          gt_labels, num_max_obj=self.num_semantic_points)
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
        neg_coords = []
        num_objs = map_cos_fg[0].shape[0]
        for i_p, map_fg, map_bg in zip(range(num_objs), map_cos_fg[-1], map_cos_bg[-1]):
            xmin, ymin, xmax, ymax = rois[i_p].int().tolist()
            H, W = map_fg.shape[-2:]
            map_crop_fg = map_fg[ymin:ymax, xmin:xmax]
            map_crop_bg = map_bg[ymin:ymax, xmin:xmax]
            # 只用来选反例
            coor = get_mask_points_single_box_cos_map_fg_bg(map_crop_fg, map_crop_bg,
                                                            pos_thr=pos_thr, neg_thr=neg_thr, 
                                                            num_gt=self.iam_num_points_init, i=i_p, 
                                                            corr_size=corr_size)
            coor[:, 0] += ymin
            coor[:, 1] += xmin
            coor = coor.flip(1)
            neg_coords.append(coor)
            # label_chosen.append(label)
        neg_coords = torch.stack(neg_coords).float() # 图像尺度大小xy格式        
        neg_prototypes = idx_by_coords(vit_feat[None].permute(0,2,3,1).expand(neg_coords.shape[0],-1,-1,-1), 
                                   (neg_coords[..., 1] // 16).long(), # 特征图尺度大小的点的y
                                   (neg_coords[..., 0] // 16).long()).clone() # 特征图尺度大小的点的x
        
        return neg_coords, neg_prototypes, map_cos_fg, map_cos_bg, points_bg, points_fg
    
# 用于做instance attention map的函数
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


# 只用来选择反例
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