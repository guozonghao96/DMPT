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
class MLPClsHead(nn.Module):

    def __init__(self,
                 in_channels=256,
                 num_classes=20,
                 cls_mlp_depth=3,
                 loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False,
                                reduction='mean', loss_weight=2.0),
                ):
        super(MLPClsHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.loss_cls = build_loss(loss_cls)
        self.cls_embed = MLP(in_channels, in_channels, num_classes, cls_mlp_depth)

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
        return cls_score

    @force_fp32(apply_to=('cls_score'))
    def loss(self,
             cls_score,
             labels):
        losses = dict()
        cls_score = cls_score.reshape(-1, self.num_classes)
        losses['loss_point_cls'] = self.loss_cls(cls_score, labels)
        return losses