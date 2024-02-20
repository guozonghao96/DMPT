import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16, force_fp32
from torch.nn.modules.utils import _pair

from mmdet.models.builder import HEADS, build_loss
from mmdet.models.losses import accuracy
from models.utils import trunc_normal_

from einops import rearrange

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

from ...losses import weight_reduce_loss
    
@HEADS.register_module()
class PointMILHead(nn.Module):

    def __init__(self,
                in_channels=256,
                hidden_channels=1024,
                num_classes=20,
                cls_threshold=0.1,
                neg_cls_threshold=0.2,
                neg_loss_weight=0.75,
                pos_gt_weight=0.125,
                loss_mil=dict(
                    type='CrossEntropyLoss', 
                    use_sigmoid=True, # BCE loss 
                    reduction='mean',
                    loss_weight=1.0),
                ):
        super(PointMILHead, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.cls_threshold = cls_threshold
        self.neg_cls_threshold = neg_cls_threshold
        self.neg_loss_weight = neg_loss_weight
        self.pos_gt_weight = pos_gt_weight
        self.num_classes = num_classes
        self.loss_mil = build_loss(loss_mil)
        
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, hidden_channels)
            
        self.score1 = nn.Linear(hidden_channels, num_classes)
        self.score2 = nn.Linear(hidden_channels, num_classes)
        
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
    
    def forward(self, pos_x, neg_x, pos_gt_x):
        # pos_x  [num_gt, num_points, C]
        pos_x = F.relu(self.fc1(pos_x), inplace=True)
        pos_x = F.relu(self.fc2(pos_x), inplace=True)
        
        neg_x = F.relu(self.fc1(neg_x), inplace=True)
        neg_x = F.relu(self.fc2(neg_x), inplace=True)
        
        pos_gt_x = F.relu(self.fc1(pos_gt_x), inplace=True)
        pos_gt_x = F.relu(self.fc2(pos_gt_x), inplace=True)
        
        # dual-stream 
        score1 = self.score1(pos_x).sigmoid() # cls branch # num_gt, 7, 20
        score2 = self.score2(pos_x).softmax(-2) # scale branch
        bag_score = score1 * score2 #* loc_scores.unsqueeze(-1) # num_gt, 7, 1
        neg_score = self.score1(neg_x).sigmoid()
        pos_gt_score = self.score1(pos_gt_x).sigmoid()
        return score1, bag_score, neg_score, pos_gt_score

    @force_fp32(apply_to=('bag_score'))
    def loss(self,
             bag_score,
             neg_score,
             pos_gt_score,
             gt_labels,
             **kwargs):
        losses = dict()
        num_gt = gt_labels.shape[0]
        if num_gt == 0:
            losses['loss_pos_gt_cls'] = pos_gt_score.sum() * 0
            losses['acc_pos_gt_cls'] = pos_gt_score.sum() * 0
            losses['loss_mil'] = bag_score.sum() * 0
            losses['acc_mil'] = bag_score.sum() * 0
            losses['loss_neg_cls'] = neg_score.sum() * 0
        else:
            # pos_gt_loss
            losses['loss_pos_gt_cls'] = self.pos_gt_weight / num_gt * \
                                        self.loss_mil.cls_criterion(pos_gt_score, gt_labels).sum()
            losses['acc_pos_gt_cls'] = accuracy(pos_gt_score, gt_labels)
            # bag loss
            losses['loss_mil'] = self.loss_mil(bag_score.sum(1), gt_labels).sum() / num_gt
            losses['acc_mil'] = accuracy(bag_score.sum(1), gt_labels)
            # neg loss
            neg_binary_labels = torch.zeros_like(neg_score)
            num_neg = neg_binary_labels.size(0)
            losses['loss_neg_cls'] = self.neg_loss_weight * self.loss_mil.cls_criterion(neg_score, 
                                                                 neg_binary_labels).sum() / num_neg
        return losses
    
    def points2bbox(self, pts):
        pts_x = pts[..., 0]
        pts_y = pts[..., 1]
        bbox_left = pts_x.min(dim=-1, keepdim=True)[0]
        bbox_right = pts_x.max(dim=-1, keepdim=True)[0]
        bbox_up = pts_y.min(dim=-1, keepdim=True)[0]
        bbox_bottom = pts_y.max(dim=-1, keepdim=True)[0]
        bbox = torch.cat([bbox_left, bbox_up, bbox_right, bbox_bottom], dim=-1)
        return bbox
    
    def forward_select_pseudo_point_gts(self,
                                        pos_scores, 
                                        neg_scores,
                                        pos_gt_scores,
                                        pos_points,
                                        neg_points,
                                        pos_pred_points,
                                        pos_gt_features,
                                        neg_features,
                                        gt_labels_,
                                        imgs_wh_=None):
        # pos points
        # 1.分类分数大于cls threshold
        num_pos_gt, num_pos_points, num_class = pos_scores.size()
        pos_scores = torch.gather(pos_scores, dim=-1, 
                                  index=gt_labels_.reshape(-1, 1, 1).repeat(1, num_pos_points, 1))[..., 0]
        pos_flag_1 = pos_scores > self.cls_threshold
        # 2.正例点分数大于gt点分数
        pos_gt_scores = torch.gather(pos_gt_scores, dim=-1, 
                                     index=gt_labels_.reshape(-1, 1))[..., 0]
        pos_flag_2 = pos_scores > pos_gt_scores.unsqueeze(-1)
        # 3.通过空间几何距离将多者分开
        if num_pos_gt == 1:
            pos_flag_3 = torch.ones_like(pos_flag_2)
        else:
            spatial_dist = torch.cdist(pos_points.reshape(-1, 2), pos_pred_points, p=2)
            _, closest_gt_idx = spatial_dist.min(dim=1)
            closest_gt_idx = closest_gt_idx.reshape(num_pos_gt, num_pos_points)
            cur_gt_idx = torch.arange(len(closest_gt_idx)).reshape(-1, 1).to(closest_gt_idx.device)
            pos_flag_3 = (closest_gt_idx == cur_gt_idx).reshape(num_pos_gt, num_pos_points)
        # 4.inside img
        pos_flag_4 = (pos_points[..., 0] >= 0) & (pos_points[..., 0] <= 1) & \
                        (pos_points[..., 1] >= 0) & (pos_points[..., 1] <= 1)
        # 5.merge
        foreground_flags = (pos_flag_1 & pos_flag_2 & pos_flag_3 & pos_flag_4)
        refined_pos_points = -torch.ones_like(pos_points)
        valid_flags = torch.zeros_like(pos_points[..., 0])
        for i_instance, (pos_flag, pos_points_) in enumerate(zip(foreground_flags, pos_points)):
            refined_pos_points[i_instance][pos_flag] = pos_points_[pos_flag]
            valid_flags[i_instance][pos_flag] = 1
        
        # neg points
        # 1.分类分数大于cls threshold, 并且看哪个类分数大
        num_neg_points, num_class = neg_scores.size()
        containing_labels = gt_labels_.unique()
        neg_scores, assigned_cls_ids = neg_scores[..., containing_labels].max(-1)
        assigned_cls_ids = containing_labels[assigned_cls_ids]
        
        neg_pos_flag = neg_scores > self.neg_cls_threshold
        assigned_cls_ids = assigned_cls_ids[neg_pos_flag]
        neg_pos_scores = neg_scores[neg_pos_flag]
        neg_pos_features = neg_features[neg_pos_flag]
        neg_pos_points = neg_points[neg_pos_flag]
        neg_neg_points = neg_points[~neg_pos_flag]
        
        # 2.与哪个gt特征相似分配给哪个gt, 用cosin similarity
        dot_matric = torch.mm(neg_pos_features, pos_gt_features.transpose(1, 0))
        norm = torch.norm(neg_pos_features, dim=-1).unsqueeze(1) * \
                torch.norm(pos_gt_features, dim=-1).unsqueeze(0)
        cosin_similarity = dot_matric / norm
        _, assigned_gt_ids = cosin_similarity.max(-1)
        assigned_gt_labels = gt_labels_[assigned_gt_ids]
        
        # 3.类别对了，同时分配gt对了就分为哪个gt
        neg_pos_flag = (assigned_gt_labels == assigned_cls_ids)
        refined_assigned_gt_ids = assigned_gt_ids[neg_pos_flag]
        neg_pos_points = neg_pos_points[neg_pos_flag]
        
        refined_neg_pos_points = -torch.ones((num_pos_gt, num_neg_points, 2)).to(neg_scores.device)
        neg_pos_valid_flags = torch.zeros((num_pos_gt, num_neg_points)).to(neg_scores.device)
        
        for i_index in refined_assigned_gt_ids.unique():
            refined_neg_pos_points[i_index][:(refined_assigned_gt_ids == i_index).sum()] = neg_pos_points[refined_assigned_gt_ids == i_index]
            neg_pos_valid_flags[i_index][:(refined_assigned_gt_ids == i_index).sum()] = 1
        
        final_pos_points = torch.cat([refined_pos_points, refined_neg_pos_points], dim=1)
        final_valid_flags = torch.cat([valid_flags, neg_pos_valid_flags], dim=1)
        
        final_neg_points = -torch.ones((num_neg_points, 2)).to(neg_scores.device)
        final_neg_valid_flags = torch.zeros((num_neg_points)).to(neg_scores.device)
        final_neg_points[:len(neg_neg_points)] = neg_neg_points
        final_neg_valid_flags[:len(neg_neg_points)] = 1
        
        return final_pos_points, final_valid_flags, final_neg_points, final_neg_valid_flags
        
        