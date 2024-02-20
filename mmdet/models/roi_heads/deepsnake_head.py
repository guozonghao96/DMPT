import torch
import torch.nn as nn
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import HEADS, build_head, build_roi_extractor
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
import torch.nn.functional as F
import numpy as np
import cv2
from cc_torch import connected_components_labeling
from mmdet.core.bbox.iou_calculators import bbox_overlaps
import mmcv
try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None

@HEADS.register_module()
class DeepSnakeHead(nn.Module):
    def __init__(self,
                 instance_extractor=None,
                 decoder_head=None,
                 num_points=32,
                ):
        super(DeepSnakeHead, self).__init__()
        self.init_decoder_head(instance_extractor, decoder_head)
        self.num_points = num_points
        
    def init_decoder_head(self, instance_extractor, decoder_head):
        self.instance_extractor = build_roi_extractor(instance_extractor)
        self.decoder_head = build_head(decoder_head)

    def init_weights(self):
        self.instance_extractor.init_weights()
        self.decoder_head.init_weights()

    def forward_train(self,
                      x,
                      pseudo_gt_bboxes,
                      vit_feat,
                      offset_targets,
                      pred_offset_map,
                      gt_points,
                      snake_targets_mask):
        split_lengths = [len(pseudo) for pseudo in pseudo_gt_bboxes]
        snake_targets_mask = list(torch.split(snake_targets_mask, split_lengths, dim=0))
        losses = dict()
        patch_h, patch_w = x[2].size(-2), x[2].size(-1)
        B, N, C = vit_feat.size()
        high_feat = vit_feat.permute(0, 2, 1).reshape(B, C, patch_h, patch_w)
        # 设置snake的采样点
        t = np.linspace(0, 2 * np.pi, self.num_points, endpoint=True)
        t = torch.from_numpy(t).to(torch.float32).to(vit_feat.device)
        
        # 获得椭圆形snake初始contours
        snakes = []
        for pseudo_bboxes in pseudo_gt_bboxes:
            snakes_per_img = []
            for pseudo in pseudo_bboxes:
                a, b = pseudo[2] - pseudo[0], pseudo[3] - pseudo[1]
                xc, yc = pseudo[0::2].mean(), pseudo[1::2].mean()
                x = xc + a / 2 * torch.sin(t)
                y = yc + b / 2 * torch.cos(t)
                snake = torch.stack([x, y]).transpose(1, 0)
                snakes_per_img.append(snake)
            snakes_per_img = torch.stack(snakes_per_img)
            snakes.append(snakes_per_img)
        
        # # 利用roi align 采样目标特征
        # rois = bbox2roi([torch.cat([snakes_per_img.int(), 
        #                             snakes_per_img + 16], dim=-1).reshape(-1, 4)
        #                  for snakes_per_img in snakes])
        # snake_feats = self.instance_extractor(
        #         [high_feat][:self.instance_extractor.num_inputs], rois)
        # snake_feats = snake_feats.reshape(-1, self.num_points, C)
        
        
        # 送入deep snake进行特征提取并预测offset
        snake_outputs = self.decoder_head(high_feat, snakes)
        decoded_snakes = snake_outputs['py_pred']
        decoded_snakes = list(torch.split(decoded_snakes, split_lengths, dim=0))
        # 先获得 snake target TODO
        snake_targets, snake_weights = self.get_snake_targets(snake_targets_mask, 
                                                              decoded_snakes, 
                                                              self.num_points)
        
        # decoded_snakes = self.decoder_head.bbox_coder.decode(
        #                 torch.cat(snakes),
        #                 snake_offsets,
        #                 torch.cat(pseudo_gt_bboxes),
        #                 max_shape=(patch_h * 16, patch_w * 16))
        # decoded_snakes = list(torch.split(decoded_snakes, split_lengths, dim=0))
        # num_points = decoded_snakes[0].size(1)
        
        losses = dict()
        loss_energy = self.decoder_head.loss(decoded_snakes, 
                                             pred_offset_map, 
                                             gt_points,
                                             snake_targets,
                                             snake_weights
                                            )
        
        # 能量函数怎么定义, offset 以什么样的scale decoder到 contour上进行偏移
        # 能量函数的loss：轮廓能量如何定义，梯度能量如何定义
        # snake_results = dict(snake_offsets=snake_offsets,
        snake_results = dict(snakes=snakes,
                             decoded_snakes=decoded_snakes,
                             # rois=[torch.cat([snakes_per_img.int(), 
                             #        snakes_per_img + 16], dim=-1).reshape(-1, 4)
                             #     for snakes_per_img in snakes],
                             pred_offset_map=pred_offset_map,
                             loss_energy=loss_energy)
        
        return snake_results
    
# def get_snake_targets(snake_targets_mask, decoded_snakes, num_sample):
#     num_sample = 32
#     #　初始化单位向量来选择点, 预测的decoded 之后的snake相同,需要用匹配获得距离最小的作为gt, 
#     #　因为是直接用匹配,所以不用有顺序
#     sobel_x = torch.as_tensor([[-1, 0, 1], 
#                            [-2, 0, 2], 
#                            [-1, 0, 1]], 
#                           dtype=decoded_snakes[0].dtype).to(decoded_snakes[0].device).reshape(1, 1, 3, 3)
#     sobel_y = torch.as_tensor([[1, 2, 1], 
#                                [0, 0, 0], 
#                                [-1, -2, -1]], 
#                               dtype=decoded_snakes[0].dtype).to(decoded_snakes[0].device).reshape(1, 1, 3, 3)

#     snake_targets = []
#     for i_batch, (masks, snakes) in enumerate(zip(snake_targets_mask, decoded_snakes)):
#         snake_targets_per_batch = []
#         for k, (mask, snake) in enumerate(zip(masks, snakes)):
#             # 1. 将目标进行闭运算,补齐空洞区域和凹区域
#             convex_mask = erode(dilate(mask.unsqueeze(0).unsqueeze(0), ksize=21), ksize=21).squeeze(0).squeeze(0)
#             # convex_mask = mask
#             convex_mask = convex_mask.unsqueeze(0).unsqueeze(0).float()
#             diff_x = F.conv2d(convex_mask, sobel_x, stride=1, padding=1)
#             diff_y = F.conv2d(convex_mask, sobel_y, stride=1, padding=1)
#             convex_mask = (diff_x ** 2 + diff_y ** 2).sqrt().squeeze(0).squeeze(0)
            
#             # 2. 直接获得目标边界点
#             coords = torch.nonzero(convex_mask).float()
#             coordinates = torch.cat([coords[:, 1:2],
#                                      coords[:, 0:1],
#                                     ], dim=-1).float() # x, y
#             plt.subplot(1, num_gt, k + 1)
#             plt.imshow(convex_mask.detach().cpu().numpy())
#             # 3. 利用l2 norm 来找最近距离的targe进行匹配
#             dist_cost = torch.cdist(snake, coordinates, p=2)
#             _, matched_indexs = linear_sum_assignment(dist_cost.detach().cpu().numpy())
#             matched_indexs = torch.from_numpy(matched_indexs).to(snake.device)
#             matched_points = coordinates[matched_indexs]
#             plt.scatter(matched_points[:, 0].detach().cpu().numpy(), matched_points[:, 1].detach().cpu().numpy(), color='r')
#             plt.scatter(snake[:, 0].detach().cpu().numpy(), snake[:, 1].detach().cpu().numpy(), color='b')
#             snake_targets_per_batch.append(matched_points)
#         snake_targets_per_batch = torch.stack(snake_targets_per_batch)
#     snake_targets.append(snake_targets_per_batch)
#     return snake_targets

    def get_snake_targets(self, snake_targets_mask, decoded_snakes, num_sample):
        #　初始化单位向量来选择点, 预测的decoded 之后的snake相同,需要用匹配获得距离最小的作为gt, 
        #　因为是直接用匹配,所以不用有顺序
        sobel_x = torch.as_tensor([[-1, 0, 1], 
                               [-2, 0, 2], 
                               [-1, 0, 1]], 
                              dtype=decoded_snakes[0].dtype).to(decoded_snakes[0].device).reshape(1, 1, 3, 3)
        sobel_y = torch.as_tensor([[1, 2, 1], 
                                   [0, 0, 0], 
                                   [-1, -2, -1]], 
                                  dtype=decoded_snakes[0].dtype).to(decoded_snakes[0].device).reshape(1, 1, 3, 3)

        snake_targets = []
        snake_weights = []
        for i_batch, (masks, snakes) in enumerate(zip(snake_targets_mask, decoded_snakes)):
            snake_targets_per_batch = []
            snake_weights_per_batch = []
            for k, (mask, snake) in enumerate(zip(masks, snakes)):
                weight = torch.ones(1).reshape(-1).to(snake.device)
                # 1. 将目标进行闭运算,补齐空洞区域和凹区域
                convex_mask = erode(dilate(mask.unsqueeze(0).unsqueeze(0), ksize=21), ksize=21).squeeze(0).squeeze(0)
                # convex_mask = mask
                # 找到边界
                convex_mask = convex_mask.unsqueeze(0).unsqueeze(0).float()
                diff_x = F.conv2d(convex_mask, sobel_x, stride=1, padding=1)
                diff_y = F.conv2d(convex_mask, sobel_y, stride=1, padding=1)
                convex_mask = (diff_x ** 2 + diff_y ** 2).sqrt().squeeze(0).squeeze(0)

                # 2. 找到contours均匀采样
                convex_mask_numpy = convex_mask.detach().cpu().byte().numpy()
                contours, _ = cv2.findContours(convex_mask_numpy,
                                       cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) != 0:
                    estimated_bbox = []
                    areas = list(map(cv2.contourArea, contours))
                    area_idx = sorted(range(len(areas)), key=areas.__getitem__, reverse=True)
                    remained_contour = contours[area_idx[0]].reshape(-1, 2)
                    length = remained_contour.shape[0]
                    sample_length = length // num_sample
                    sample_indexs = np.arange(num_sample) * sample_length
                    sampled_contour = remained_contour[sample_indexs]
                    sampled_contour_tensor = torch.from_numpy(sampled_contour).to(snake.device).float()
                    dist_cost = torch.cdist(snake, sampled_contour_tensor, p=2)
                    _, matched_indexs = linear_sum_assignment(dist_cost.detach().cpu().numpy())

                    matched_points = sampled_contour_tensor[matched_indexs]

                    if len(remained_contour) < num_sample:
                        matched_points = torch.zeros(num_sample, 2).to(snake.device)
                        weight[0] = 0
                else:
                    matched_points = torch.zeros(num_sample, 2).to(snake.device)
                    weight[0] = 0

                snake_targets_per_batch.append(matched_points)
                snake_weights_per_batch.append(weight)
            snake_targets_per_batch = torch.stack(snake_targets_per_batch)
            snake_weights_per_batch = torch.stack(snake_weights_per_batch)

            snake_targets.append(snake_targets_per_batch)
            snake_weights.append(snake_weights_per_batch)
        
        return snake_targets, snake_weights
    
def erode(bin_img, ksize=11):
    # 首先为原图加入 padding，防止腐蚀后图像尺寸缩小
    B, C, H, W = bin_img.shape
    bin_img = bin_img
    pad = (ksize - 1) // 2
    bin_img = F.pad(bin_img, [pad, pad, pad, pad], mode='constant', value=0)

    # 将原图 unfold 成 patch
    patches = bin_img.unfold(dimension=2, size=ksize, step=1)
    patches = patches.unfold(dimension=3, size=ksize, step=1)
    # B x C x H x W x k x k

    # 取每个 patch 中最小的值，i.e., 0
    eroded, _ = patches.reshape(B, C, H, W, -1).min(dim=-1)
    return eroded

def dilate(bin_img, ksize=11):
    # 首先为原图加入 padding，防止腐蚀后图像尺寸缩小
    B, C, H, W = bin_img.shape
    bin_img = bin_img
    pad = (ksize - 1) // 2
    bin_img = F.pad(bin_img, [pad, pad, pad, pad], mode='constant', value=0)

    # 将原图 unfold 成 patch
    patches = bin_img.unfold(dimension=2, size=ksize, step=1)
    patches = patches.unfold(dimension=3, size=ksize, step=1)
    # B x C x H x W x k x k

    # 取每个 patch 中最小的值，i.e., 0
    dilated, _ = patches.reshape(B, C, H, W, -1).max(dim=-1)
    return dilated
    
        