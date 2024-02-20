# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
import torch
from ..builder import LOSSES
from .utils import weighted_loss


def energy(decoded_snakes,
           energy_map, #[2, h, w]
           points,
           snake_targets,
           snake_weights,
           alpha=0.5,
           beta=0.5,
           gamma=1.0,
           sigma=1.0,
           weight=None,
           reduction='mean',
           avg_factor=None):
    
    num_gt, num_point = decoded_snakes.size(0), decoded_snakes.size(1)
    
#     left_decoded_snakes = torch.roll(decoded_snakes, 1, dims=(1))
#     right_decoded_snakes = torch.roll(decoded_snakes, -1, dims=(1))
#     # 弹性势能  #右边减去左边 roll 正数是左边  负数是右边
#     diff_snakes = (right_decoded_snakes - left_decoded_snakes) ** 2 + (left_decoded_snakes - right_decoded_snakes) ** 2
#     energy1 = alpha * diff_snakes.sum()
    
#     # 弯曲势能 #右边减去左边
#     diff_diff_snakes = (right_decoded_snakes + left_decoded_snakes - 2 * decoded_snakes) ** 2 + (left_decoded_snakes + right_decoded_snakes - 2 * decoded_snakes) ** 2
#     energy2 = beta * diff_diff_snakes.sum()
    
    # 内力 l1 loss
    snake_weights_ = snake_weights.reshape(num_gt, 1, 1).repeat(1, num_point, 2)
    energy_1 = alpha * ((decoded_snakes - snake_targets) * snake_weights_).abs().sum()
    # # 梯度势能 
    # coordinates = decoded_snakes.long().reshape(-1, 2) - 1
    # # x_offset
    # energy3_1 = energy_map[0][coordinates[:, 1], coordinates[:, 0]]
    # # y_offset
    # energy3_2 = energy_map[1][coordinates[:, 1], coordinates[:, 0]]
    # # distance energy
    # energy3 = -gamma * (energy3_1 ** 2 + energy3_2 ** 2).sqrt().sum()
    
#     # 约束snake的质心只在 gt附近运动，不能跑太远
#     # energy4 = sigma * ((decoded_snakes.mean(1) - points) ** 2).sum()
#     # loss = (energy1 + energy2 + energy3 + energy4) / (num_gt * num_point)
#     loss = (energy1 + energy2 + energy3) / (num_gt * num_point)
    # loss = (energy_1 + energy3) / (snake_weights.sum() * num_point + 1e-7)
    loss = (energy_1) / (snake_weights.sum() * num_point + 1e-7)

    return loss


@LOSSES.register_module()
class SnakeEnergyLoss(nn.Module):

    def __init__(self, 
                 alpha=0.5,
                 beta=0.5,
                 gamma=1.0,
                 sigma=1.0,
                 reduction='mean', 
                 loss_weight=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.sigma = sigma
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                decoded_snakes,
                pred_energy_maps,
                points,
                snake_targets,
                snake_weights,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * energy(
            decoded_snakes, pred_energy_maps, points, snake_targets, snake_weights,
            self.alpha, self.beta, self.gamma, self.sigma,
            weight, reduction=reduction, avg_factor=avg_factor)
        return loss
