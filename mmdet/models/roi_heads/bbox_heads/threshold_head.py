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
class ThresholdHead(nn.Module):

    def __init__(self,
                in_channels=256,
                hidden_channels=1024
                ):
        super(ThresholdHead, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, hidden_channels)
        self.score = nn.Linear(hidden_channels, 1)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        
        self.score.weight.data.zero_()
        self.score.bias.data.copy_(torch.tensor([0.2], dtype=torch.float))
        
    def init_weights(self):
        self.apply(self._init_weights)
    
    def forward(self, x, num_gt_per_batch=None):
        '''
            x: vit_feat -> batch_size, num_patch, channel
            num_gt_per_batch: list > [num_gt1, num_gt2]
        '''
        threshold_feat = []
        for feat, num_gt in zip(x.unsqueeze(1), num_gt_per_batch):
            threshold_feat.append(feat.repeat(num_gt, 1, 1))
        threshold_feat = torch.cat(threshold_feat) # num_gt, num_patch, channel
        threshold_feat = F.relu(self.fc1(threshold_feat), inplace=True)
        threshold_feat = F.relu(self.fc2(threshold_feat), inplace=True)
        threshold = self.score(threshold_feat.mean(1))
        print(threshold.reshape(-1))
        return threshold