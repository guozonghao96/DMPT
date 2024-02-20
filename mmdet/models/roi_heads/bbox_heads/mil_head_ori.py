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
class MILHead(nn.Module):

    def __init__(self,
                in_channels=256,
                hidden_channels=1024,
                pooling_type='roi',
                roi_size=7,
                num_classes=20,
                topk_merge=1,
                loss_mil=dict(
                    type='CrossEntropyLoss', 
                    use_sigmoid=True, # BCE loss 
                    reduction='mean',
                    loss_weight=1.0),
                ):
        super(MILHead, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.topk_merge = topk_merge
        self.num_classes = num_classes
        self.loss_mil = build_loss(loss_mil)
        
        self.pooling_type = pooling_type
        self.roi_size = roi_size
        
        if pooling_type == 'attn':
            self.fc1 = nn.Linear(in_channels, hidden_channels)
            self.fc2 = nn.Linear(hidden_channels, hidden_channels)
#             self.fc1_e = nn.Linear(in_channels, hidden_channels)
#             self.fc2_e = nn.Linear(hidden_channels, hidden_channels)
        elif pooling_type == 'roi':
            self.fc1 = nn.Linear(in_channels * roi_size ** 2, hidden_channels)
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
    
    def forward(self, x, x_edge=None, num_scale=None):
        if self.pooling_type == 'attn':
            # x -> (num_gt, num_scale, c)
            if x_edge is not None: # edge detection
                x = F.relu(self.fc1(x), inplace=True)
                x = F.relu(self.fc2(x), inplace=True)
                score1 = self.score1(x).softmax(-1) # cls branch
                
                x_edge = F.relu(self.fc1_e(x_edge), inplace=True)
                x_edge = F.relu(self.fc2_e(x_edge), inplace=True)
                score2 = (self.score2(x) - self.score2(x_edge)).softmax(-2) # scale branch
                bag_score = score1 * score2
                return bag_score
            else:
                x = F.relu(self.fc1(x), inplace=True)
                x = F.relu(self.fc2(x), inplace=True)
                # dual-stream 
                score1 = self.score1(x).softmax(-1) # cls branch
                score2 = self.score2(x).softmax(-2) # scale branch
                bag_score = score1 * score2
                return bag_score
            
        elif self.pooling_type == 'roi':
            x = x.reshape(-1, num_scale, self.in_channels * self.roi_size ** 2)
            x = F.relu(self.fc1(x), inplace=True)
            x = F.relu(self.fc2(x), inplace=True)
            # dual-stream 
            score1 = self.score1(x).softmax(-1) # cls branch
            score2 = self.score2(x).softmax(-2) # scale branch
            bag_score = score1 * score2
            return bag_score

    @force_fp32(apply_to=('bag_score'))
    def loss(self,
             bag_score,
             gt_labels,
             **kwargs):
        losses = dict()
        gt_labels = torch.cat(gt_labels).reshape(-1)
        losses['loss_mil'] = self.loss_mil(bag_score.sum(-2), gt_labels)
        return losses
        