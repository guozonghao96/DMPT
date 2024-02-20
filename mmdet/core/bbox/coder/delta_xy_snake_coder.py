# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import mmcv
import numpy as np
import torch

from ..builder import BBOX_CODERS
from .base_bbox_coder import BaseBBoxCoder


@BBOX_CODERS.register_module()
class DeltaXYSnakeCoder(BaseBBoxCoder):

    def __init__(self,
                 target_means=(0., 0., 0., 0.),
                 target_stds=(1., 1., 1., 1.),
                 clip_border=True,
                 add_ctr_clamp=False,
                 ctr_clamp=32):
        super(BaseBBoxCoder, self).__init__()
        self.means = target_means
        self.stds = target_stds
        self.clip_border = clip_border
        self.add_ctr_clamp = add_ctr_clamp
        self.ctr_clamp = ctr_clamp

    def encode(self, bboxes, gt_bboxes):
        pass
    
    def decode(self,
               snakes,
               snake_offsets,
               pseudo_gt_bboxes,
               max_shape=None,
               wh_ratio_clip=16 / 1000):
        decoded_snakes = delta2snake(snakes, snake_offsets, pseudo_gt_bboxes,
                                     self.means, self.stds, max_shape, wh_ratio_clip,
                                     self.clip_border, self.add_ctr_clamp,
                                     self.ctr_clamp)

        return decoded_snakes

@mmcv.jit(coderize=True)
def delta2snake(snakes,
                snake_offsets,
                pseudo_gt_bboxes,
                means=(0., 0.),
                stds=(1., 1.),
                max_shape=None,
                wh_ratio_clip=16 / 1000,
                clip_border=True,
                add_ctr_clamp=False,
                ctr_clamp=32):
    snakes = snakes.reshape(-1, 2)
    num_point = snake_offsets.size(1)
    snake_offsets = snake_offsets.reshape(-1, 2)
    means = snake_offsets.new_tensor(means).view(1, -1)
    stds = snake_offsets.new_tensor(stds).view(1, -1)
    snake_offsets = snake_offsets * stds + means
    proposal_ws = (pseudo_gt_bboxes[:, 2] - pseudo_gt_bboxes[:, 0]).reshape(-1, 1, 1).repeat(1, num_point, 1)
    proposal_hs = (pseudo_gt_bboxes[:, 3] - pseudo_gt_bboxes[:, 1]).reshape(-1, 1, 1).repeat(1, num_point, 1)
    proposal_wh = torch.cat([proposal_ws, proposal_hs], dim=-1).reshape(-1, 2)
    snake_offsets = snake_offsets * proposal_wh
    decoded_snakes = snakes + snake_offsets
    
    if clip_border and max_shape is not None:
        decoded_snakes[..., 0::2].clamp_(min=0, max=max_shape[1])
        decoded_snakes[..., 1::2].clamp_(min=0, max=max_shape[0])
    decoded_snakes = decoded_snakes.reshape(-1, num_point, 2)
    return decoded_snakes