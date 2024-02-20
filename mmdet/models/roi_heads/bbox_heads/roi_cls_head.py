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
class RoIClsHead(nn.Module):

    def __init__(self,
                in_channels=256,
                hidden_channels=1024,
                roi_size=7,
                num_classes=20,
                loss_roi_cls=dict(
                    type='CrossEntropyLoss', 
                    use_sigmoid=True, # BCE loss 
                    reduction='mean',
                    loss_weight=1.0),
                ):
        super(RoIClsHead, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        self.loss_roi_cls = build_loss(loss_roi_cls)
        
        self.fc1 = nn.Linear(in_channels * roi_size ** 2, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, hidden_channels)
        self.score = nn.Linear(hidden_channels, self.num_classes)
        
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
        '''
            x: roi feat -> num_gt, channel, S, S
        '''
        x = x.flatten(1)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        scores = self.score(x)
        return scores

    @force_fp32(apply_to=('scores'))
    def loss(self,
             scores,
             gt_labels,
             valid_weights,
             **kwargs):
        losses = dict()
        gt_labels = torch.cat(gt_labels).reshape(-1)
        losses['loss_roi_cls'] = self.loss_roi_cls(scores, 
                                                   gt_labels,
                                                   valid_weights)
        return losses
        