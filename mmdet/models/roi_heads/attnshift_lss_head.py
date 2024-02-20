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


import numpy as np 
try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None
    

@HEADS.register_module()
class AttnLatentScaleSelectionHead(nn.Module):
    def __init__(self,
                 instance_extractor=None,
                 point_head=None,
                 mil_head=None,
                 train_cfg=None,
                 test_cfg=None,
                ):
        super(AttnLatentScaleSelectionHead, self).__init__()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        # 设置pooling的pre_process
        if self.train_cfg is None:
            self.train_cfg = self.test_cfg
        self.pooling = self.train_cfg.get('pooling', None)
        if self.pooling is None:
            assert False, 'should identificate "pooling" in train_cfg'
        self.pooling_type = self.pooling.type
        self.scale = self.pooling.get('scale')
        self.scale_method = self.pooling.get('scale_method')
        self.filter = self.pooling.get('filter')
        self.multiple = self.pooling.get('multiple')
        self.discard = self.pooling.get('discard')
        
        if mil_head is not None:
            self.init_mil_head(instance_extractor, mil_head)
        if point_head is not None:
            self.init_point_head(point_head)
            
        self.init_assigner_sampler()

    @property
    def with_mil(self):
        return hasattr(self, 'mil_head') and self.mil_head is not None
    
    @property
    def with_point(self):
        return hasattr(self, 'point_head') and self.point_head is not None
    
    def init_mil_head(self, instance_extractor, mil_head):
        self.instance_extractor = build_roi_extractor(instance_extractor)
        self.mil_head = build_head(mil_head)
        
    def init_point_head(self, point_head):
        self.point_head = build_head(point_head)
    
    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.point_assigner = None
        self.point_sampler = None
        if self.train_cfg:
            self.point_assigner = build_assigner(self.train_cfg.assigner)
            self.point_sampler = build_sampler(
                self.train_cfg.sampler, context=self)

    def init_weights(self):
        if self.with_mil:
            self.instance_extractor.init_weights()
            self.mil_head.init_weights()
        if self.with_point:
            self.point_head.init_weights()
            
    def generate_multi_scale_cams(self, attns_maps, point_results, 
                                  align_label=True):
        '''
        input:
            attns_maps -> (blocks, batch_size, N, N)
            #只取需要的层数
            -> attns_maps -> (scale, batch_size, N, N)
            #得到多尺度的attns
            -> joint_attentions -> (scale, batch_size, N, N)
            -> joint_attentions -> (batch_size, scale, N, N)
            #取出match的object的attention maps,并resize成maps
            -> point_attentions -> (batch_size, scale, num_gt, patch_h * patch_w)
        output:
            point_attentions -> (batch_size, scale, num_gt, patch_h * patch_w)
        '''
        # 1. filter the noise 
        if self.filter:
            bs = attns_maps.size(1)
            num_patches = attns_maps.size(-1)
            for i in range(len(attns_maps)):
                flat = attns_maps[i].reshape(bs, -1)
                    # 取topk的参数需要排序算法,这个过滤噪声,速度太慢
                _, indices = flat.topk(int(flat.size(-1) * self.discard), -1, False)
                for j in range(len(flat)):
                    flat[j, indices[j]] = 0
                attns_maps[i] = flat.reshape(bs, num_patches, num_patches)
    
        if self.multiple:
        # 2. multiple matrics
            residual_att = torch.eye(attns_maps.size(2), dtype=attns_maps.dtype, device=attns_maps.device)
            aug_att_mat = attns_maps + residual_att
            aug_att_mat = aug_att_mat / aug_att_mat.sum(-1).unsqueeze(-1)
            joint_attentions = torch.zeros(aug_att_mat.size(), dtype=aug_att_mat.dtype, device=aug_att_mat.device)
            joint_attentions[-1] = aug_att_mat[-1]
            for i in range(2, len(attns_maps) + 1):
                joint_attentions[-i] = torch.matmul(joint_attentions[-(i - 1)], aug_att_mat[-i])
            joint_attentions = joint_attentions.permute(1, 0, 2, 3)
        else:
            joint_attentions = attns_maps.permute(1, 0, 2, 3)
            
        # 3. selection function
        if self.scale_method == 'average':
            joint_attentions = joint_attentions.mean(1).unsqueeze(1)
        elif self.scale_method == 'single':
            joint_attentions = joint_attentions[:, 12 - self.scale, ...].unsqueeze(1)
        elif self.scale_method == 'auto':
            joint_attentions = joint_attentions[:, -self.scale:, ...]
        elif self.scale_method == 'scale':
            assert self.scale == 4
            joint_attentions = torch.cat([joint_attentions[:, 3:4, ...],
                                          joint_attentions[:, 5:6, ...],
                                          joint_attentions[:, 7:8, ...],
                                          joint_attentions[:, 11:12, ...]
                                         ], dim=1)
        # 4. get point matched attention maps
        pos_inds = point_results['pos_inds']        
        num_imgs = len(pos_inds)
        num_points = point_results['cls_score'].size(1)
        
        multiple_cams = []
        for i in range(num_imgs):
            pos_inds_per_img = pos_inds[i]
            point_attn_maps_per_img = joint_attentions[i, :, -num_points:, 1:-num_points]
            matched_point_attn_maps_per_img = point_attn_maps_per_img[:, pos_inds_per_img] # (scale, num_gt, num_patches)
            matched_point_attn_maps_per_img = matched_point_attn_maps_per_img.permute(1, 0, 2) # (num_gt, scale, num_patches)
            multiple_cams.append(matched_point_attn_maps_per_img)
        multiple_cams = torch.cat(multiple_cams)
        # 4. align oders of "gt_labels" and "gt_points" to that of "pos_inds"
        gt_labels = []
        gt_points = []
        labels = point_results['point_targets'][0].reshape(-1, num_points)
        points = point_results['point_targets'][2].reshape(-1, num_points, 2)
        
        for i in range(num_imgs):
            pos_inds_per_img = pos_inds[i]
            labels_per_img = labels[i]
            points_per_img = points[i]
            
            gt_labels.append(labels_per_img[pos_inds_per_img])
            gt_points.append(points_per_img[pos_inds_per_img])        
        # 5. align oders of "gt_bboxes" to that of "pos_inds"
        gt_bboxes_ = []
        pos_assigned_gt_inds = point_results['pos_assigned_gt_inds']
        gt_bboxes = point_results['gt_bboxes']
        for i in range(num_imgs):
            gt_inds = pos_assigned_gt_inds[i]
            bboxes = gt_bboxes[i]
            gt_bboxes_.append(bboxes[gt_inds])
            
        return multiple_cams, gt_labels, gt_points, gt_bboxes_
    
    def _generate_multi_scale_cams_(self, attns_maps, point_results, 
                                  align_label=True):
        '''
        input:
            attns_maps -> (blocks, batch_size, N, N) (h, w)
            #只取需要的层数
            -> attns_maps -> (scale, batch_size, N, N)
            #得到多尺度的attns
            -> joint_attentions -> (scale, batch_size, N, N)
            -> joint_attentions -> (batch_size, scale, N, N)
            #取出match的object的attention maps,并resize成maps
            -> point_attentions -> (batch_size, scale, num_gt, patch_h * patch_w)
        output:
            point_attentions -> (batch_size, scale, num_gt, patch_h * patch_w)
        '''
        patch_h, patch_w = attns_maps.size()[-2:]
#         # 1. filter the noise 
#         if self.filter:
#             bs = attns_maps.size(1)
#             num_patches = attns_maps.size(-1)
#             for i in range(len(attns_maps)):
#                 flat = attns_maps[i].reshape(bs, -1)
#                     # 取topk的参数需要排序算法,这个过滤噪声,速度太慢
#                 _, indices = flat.topk(int(flat.size(-1) * self.discard), -1, False)
#                 for j in range(len(flat)):
#                     flat[j, indices[j]] = 0
#                 attns_maps[i] = flat.reshape(bs, num_patches, num_patches)
    
#         if self.multiple:
#         # 2. multiple matrics
#             residual_att = torch.eye(attns_maps.size(2), dtype=attns_maps.dtype, device=attns_maps.device)
#             aug_att_mat = attns_maps + residual_att
#             aug_att_mat = aug_att_mat / aug_att_mat.sum(-1).unsqueeze(-1)
#             joint_attentions = torch.zeros(aug_att_mat.size(), dtype=aug_att_mat.dtype, device=aug_att_mat.device)
#             joint_attentions[-1] = aug_att_mat[-1]
#             for i in range(2, len(attns_maps) + 1):
#                 joint_attentions[-i] = torch.matmul(joint_attentions[-(i - 1)], aug_att_mat[-i])
#             joint_attentions = joint_attentions.permute(1, 0, 2, 3)
#         else:
#             joint_attentions = attns_maps.permute(1, 0, 2, 3)
            
#         # 3. selection function
#         if self.scale_method == 'average':
#             joint_attentions = joint_attentions.mean(1).unsqueeze(1)
#         elif self.scale_method == 'single':
#             joint_attentions = joint_attentions[:, 12 - self.scale, ...].unsqueeze(1)
#         elif self.scale_method == 'auto':
#             joint_attentions = joint_attentions[:, -self.scale:, ...]
#         elif self.scale_method == 'scale':
#             assert self.scale == 4
#             joint_attentions = torch.cat([joint_attentions[:, 3:4, ...],
#                                           joint_attentions[:, 5:6, ...],
#                                           joint_attentions[:, 7:8, ...],
#                                           joint_attentions[:, 11:12, ...]
#                                          ], dim=1)
        # 4. get point matched attention maps
        num_stages, num_imgs, num_points = point_results['cls_score'].size()[:3]
        pos_inds = point_results['pos_inds'] #num_stage,  
        point_targets = point_results['point_targets']
        
        pos_assigned_gt_inds = point_results['pos_assigned_gt_inds']
        gt_align_inds = pos_assigned_gt_inds[-1]
        # num_stages = len(pos_inds)
        
        multiple_cams = []
        for i in range(num_stages):
            for j in range(num_imgs):
                pos_inds_per_img = pos_inds[i][j]
                point_attn_maps_per_img = attns_maps[i][j][pos_inds_per_img]
                matched_cost = torch.cdist(gt_align_inds[j].reshape(-1, 1).float(), 
                                           pos_assigned_gt_inds[i][j].reshape(-1, 1).float(),
                                           p=1).cpu()
                matched_row_inds, matched_col_inds = linear_sum_assignment(matched_cost)
                point_attn_maps_per_img[matched_col_inds]
                attn_cams = torch.zeros_like(point_attn_maps_per_img)
                attn_cams = point_attn_maps_per_img[matched_col_inds]
                multiple_cams.append(attn_cams)
        multiple_cams = torch.cat(multiple_cams).reshape(num_stages, -1, patch_h, patch_w).permute(1, 0, 2, 3)
        
        # 4. align oders of "gt_labels" and "gt_points" to that of "pos_inds"
        gt_labels = []
        gt_points = []
        labels = point_targets[-1][0].reshape(-1, num_points)
        points = point_targets[-1][2].reshape(-1, num_points, 2)
        
        for i in range(num_imgs):
            pos_inds_per_img = pos_inds[-1][i]
            labels_per_img = labels[i]
            points_per_img = points[i]
            
            gt_labels.append(labels_per_img[pos_inds_per_img])
            gt_points.append(points_per_img[pos_inds_per_img])        
        # 5. align oders of "gt_bboxes" to that of "pos_inds"
        gt_bboxes_ = []
        # pos_assigned_gt_inds = point_results['pos_assigned_gt_inds'][-1]
        gt_bboxes = point_results['gt_bboxes']
        for i in range(num_imgs):
            gt_inds = gt_align_inds[i]
            bboxes = gt_bboxes[i]
            gt_bboxes_.append(bboxes[gt_inds])
            
        return multiple_cams, gt_labels, gt_points, gt_bboxes_
    
    def forward_train_point(self,
                            x,
                            vit_feat,
                            point_tokens,
                            attns,
                            img_metas,
                            gt_bboxes,
                            gt_labels,
                            gt_points,
                            imgs_whwh=None,
                            gt_bboxes_ignore=None):
        
        patch_h, patch_w = x[2].size(-2), x[2].size(-1)
        if self.with_point:
            # point_results = self._point_forward_train(point_tokens, gt_points, 
            point_results = self._point_forward_train(vit_feat, gt_points, 
                                                      gt_labels, img_metas, 
                                                      attns,
                                                      imgs_whwh=imgs_whwh, 
                                                      gt_bboxes=gt_bboxes,
                                                      patch_size=(patch_h, patch_w)
                                                     )
            return point_results
        
    def _calculate_loss(self, img_metas, gt_points, gt_labels,
                        key_points, visible_flags, point_results, imgs_whwh=None):
        
        cls_score = point_results['cls_score'].detach()
        point_pred = point_results['point_pred'].detach()
        
        num_imgs = len(img_metas)
        num_proposals = cls_score.size(1)
        imgs_whwh = imgs_whwh.repeat(1, num_proposals, 1)
        
        all_gt_points = []
        all_gt_labels = []
        # pseudo_points 是 下采样16倍的xy坐标
        for points, labels, pseudo_points, visible in zip(gt_points, gt_labels, key_points, visible_flags):
            visible = visible.bool()
            num_points = pseudo_points.size(1)
            pseudo_labels_ = labels.reshape(-1, 1).repeat(1, num_points)
            # 只取key point 除第一个之外的所有，因为第一个就为gt,
            # 或者不用cat，直接用key point当作gt points
            visible_gt_points = pseudo_points[visible]
            visible_labels = pseudo_labels_[visible]
            all_gt_points.append(visible_gt_points)
            all_gt_labels.append(visible_labels)
            # all_gt_points.append(torch.cat([points, 
            #                                 pseudo_points.reshape(-1, 2).float()], dim=0))
            # all_gt_labels.append(torch.cat([labels, pseudo_labels_.reshape(-1)], dim=0))
            
        point_assign_results = []
        for i in range(num_imgs):
            assign_result = self.point_assigner.assign(
                point_pred[i], cls_score[i], all_gt_points[i],
                all_gt_labels[i], img_metas[i])
            point_sampling_result = self.point_sampler.sample(
                assign_result, point_pred[i], all_gt_points[i]
            )
            point_assign_results.append(point_sampling_result)
        
        point_targets = self.point_head._get_targets_(point_assign_results, all_gt_points,
                                                   all_gt_labels, self.train_cfg, True)
        loss_point = self.point_head._loss_(point_results['cls_score'],
                                          point_results['point_pred'],
                                          *point_targets, imgs_whwh=imgs_whwh)
        return loss_point
        
    def _point_forward_train(self, x, gt_points, gt_labels, img_metas, attns,
                             imgs_whwh=None, gt_bboxes=None, patch_size=None):
        # x -> point tokens -> (batch_size, point_num, c)
        num_imgs = len(img_metas)
        # num_proposals = x.size(1)
        # inference in mlp_head
        point_results = self._point_forward(x, img_metas)
        cls_score = point_results['cls_score'].detach() # num_stage, bs, num_query, 20
        point_pred = point_results['point_pred'].detach() # num_stage, bs, num_query, 2
        # get assign and sample results
        imgs_whwh = imgs_whwh.repeat(1, cls_score.size(2), 1)
        num_stage = cls_score.size(0)
        
        pos_pred_points = []
        pos_inds = []
        pos_assigned_gt_inds = []
        point_targets = []
        losses = dict()
        
        for i_stage in range(num_stage):
            point_assign_results = []
            for i in range(num_imgs):
                assign_result = self.point_assigner.assign(
                    point_pred[i_stage][i], cls_score[i_stage][i], gt_points[i],
                    gt_labels[i], img_metas[i])
                point_sampling_result = self.point_sampler.sample(
                    assign_result, point_pred[i_stage][i], gt_points[i]
                )
                point_assign_results.append(point_sampling_result)
            # get point targets
            _point_targets_ = self.point_head.get_targets(point_assign_results, gt_points,
                                                      gt_labels, self.train_cfg, True)
            # point loss

            loss_point = self.point_head.loss(point_results['cls_score'][i_stage],
                                              point_results['point_pred'][i_stage],
                                              *_point_targets_, imgs_whwh=imgs_whwh)
            
            pos_inds.append([sample_results.pos_inds 
                                       for sample_results in point_assign_results])
            pos_assigned_gt_inds.append([sample_results.pos_assigned_gt_inds 
                                       for sample_results in point_assign_results])
            point_targets.append(_point_targets_)
            
            for k, v in loss_point.items():
                losses[k + '_{:d}'.format(i_stage)] = v
        # point_results.update(point_loss=losses)
        
        # matched for generating box
        point_results.update(pos_inds=pos_inds) # pos_inds
        # align the gt_bboxes
        point_results.update(pos_assigned_gt_inds=pos_assigned_gt_inds) 
        point_results.update(gt_bboxes=gt_bboxes) 
        point_results.update(point_targets=point_targets)
        
        # attn 
        patch_h, patch_w = patch_size
        multiple_cams, gt_labels, gt_points, gt_bboxes = self._generate_multi_scale_cams_(point_results['attns'], 
                                                                                          point_results)
        split_lengths = [len(bboxes) for bboxes in gt_bboxes]
        # 每个attn maps 变成 ori 尺度大小，再进行生成框操作
        # multiple_cams = multiple_cams.reshape(-1, multiple_cams.size(1), patch_h, patch_w)
        multiple_cams = F.interpolate(multiple_cams, 
                                     (patch_h * 16, patch_w * 16), 
                                     mode='bilinear') # num_gt, num_scale, H, W
        multiple_cams = list(torch.split(multiple_cams, split_lengths, dim=0))
        # proposals
        pseudo_proposals, refined_multiple_masks = self.pre_get_bboxes(torch.cat(multiple_cams),
                                            gt_points,
                                            patch_size=patch_size,
                                            test_cfg=self.test_cfg)
        output_point_results = dict(
            pos_inds=pos_inds,
            loss_point=losses,
            multiple_cams=multiple_cams,
            gt_labels=gt_labels,
            gt_points=gt_points,
            gt_bboxes=gt_bboxes,
            pseudo_proposals=pseudo_proposals,
            refined_multiple_masks=refined_multiple_masks,
            cls_score=point_results['cls_score'],
            point_pred=point_results['point_pred'],
            vit_feat=point_results['vit_feat']
        )
        if not self.with_mil:
            pseudo_gt_bboxes = [pseudo.squeeze(1) for pseudo in pseudo_proposals]
            iou_metric = bbox_overlaps(torch.cat(pseudo_gt_bboxes),
                                       torch.cat(gt_bboxes),
                                       is_aligned=True).mean()
            output_point_results['iou_metric'] = iou_metric
            output_point_results['pseudo_gt_bboxes'] = pseudo_gt_bboxes
            output_point_results['pseudo_gt_labels'] = gt_labels
            
        return output_point_results
    
    def _point_forward(self, x, img_metas):
        cls_score, point_pred, attns, vit_feat = self.point_head(x, img_metas)
        point_results = dict(
            cls_score=cls_score[0], point_pred=point_pred[0], 
            point_tokens=x, attns=attns[0], vit_feat=vit_feat[0])
        return point_results
    
    
    def forward_train_mil(self,
                          x,
                          vit_feat,
                          point_tokens,
                          pred_offset_map,
                          pred_seg_map,
                          multiple_cams,
                          multiple_masks,
                          pseudo_proposals,
                          img_metas,
                          gt_bboxes,
                          gt_labels,
                          gt_points,
                          imgs_whwh=None,
                          gt_bboxes_ignore=None):
        losses = dict()
        patch_h, patch_w = x[2].size(-2), x[2].size(-1)
        
        if self.with_mil:
            mil_results = self._mil_forward_train(vit_feat, multiple_cams,
                                                  multiple_masks,
                                                  pred_offset_map,
                                                  pred_seg_map,
                                                  pseudo_proposals,
                                                  gt_points, gt_labels,
                                                  gt_bboxes,
                                                  patch_size=(patch_h, patch_w),
                                                  test_cfg=self.test_cfg)
            loss_mil = self.mil_head.loss(mil_results['bag_score'], 
                                          mil_results['pseudo_gt_labels'])
            mil_results.update(loss_mil=loss_mil)
            return mil_results
        
        
    def _mil_forward_train(self, vit_feat, multiple_cams, 
                           multiple_masks,
                           pred_offset_map, pred_seg_map,
                           pseudo_proposals,
                           gt_points, gt_labels, 
                           gt_bboxes,
                           patch_size=None,
                           test_cfg=None):
        mil_results = self._mil_forward(vit_feat, multiple_cams,
                                        multiple_masks,
                                        pred_offset_map, pred_seg_map,
                                        pseudo_proposals,
                                        gt_points, gt_labels, 
                                        gt_bboxes, patch_size,
                                        test_cfg=test_cfg)
        
        loss_mil = self.mil_head.loss(mil_results['bag_score'], 
                                      mil_results['pseudo_gt_labels'])
        mil_results.update(loss_mil=loss_mil)
        return mil_results
    
    def _mil_forward(self, x, multiple_cams,  
                     multiple_masks,
                     pred_offset_map, pred_seg_map,
                     pseudo_proposals,
                     gt_points, gt_labels,
                     gt_bboxes, patch_size=None,
                     test_cfg=None):
        num_imgs = len(gt_points)
        # 1. 生成isntance map
        # 用 分割预测生成instance map
        if pred_offset_map is not None and pred_seg_map is not None:
            with torch.no_grad():
                refined_multiple_masks = []
                new_proposals = []
                for offset_map, seg_map, points, labels, proposals in \
                        zip(pred_offset_map, pred_seg_map, gt_points, gt_labels, pseudo_proposals):
                    instance_masks, proposals = self.single_get_instance_maps(offset_map, # scale, 2, h, w
                                                                       seg_map, # scale, 21, h, w
                                                                       points, # num_gt, 2
                                                                       labels, # num_gt
                                                                       proposals, # num_gt, num_scale, 4
                                                                      )
                    refined_multiple_masks.append(instance_masks)
                    new_proposals.append(proposals)

            x = x.permute(0, 2, 1).reshape(num_imgs, -1, *patch_size).contiguous()
            rois = bbox2roi([proposals.reshape(-1, 4) for proposals in new_proposals]) # 利用instance seg获得proposal
#             rois = bbox2roi([proposals.reshape(-1, 4) for proposals in pseudo_proposals])
            instance_feats = self.instance_extractor(
                [x][:self.instance_extractor.num_inputs], rois)
            bag_score = self.mil_head(instance_feats, num_scale=self.scale)
            # 利用结果生成监督信息, 首先测试自refine的策略，不行再用atten
            with torch.no_grad():
                # matched_cams, semantic_scores = self.post_get_bboxes(pseudo_proposals, 
#                     semantic_scores = self.post_get_bboxes(new_proposals,  # 这个可以用预测特征输入的mask
                pseudo_gt_labels, pseudo_gt_bboxes, matched_cams, matched_masks, \
                    semantic_scores = self.post_get_bboxes(new_proposals,  # 这个可以用预测特征输入的mask
                                                           bag_score,
                                                           gt_labels,
                                                           torch.cat(multiple_cams), 
                                                           torch.cat(multiple_masks), # 但是这个需要用attn的mask
                                                           test_cfg=test_cfg)

                iou_metric = bbox_overlaps(torch.cat(pseudo_gt_bboxes),
                                           torch.cat(gt_bboxes),
                                           is_aligned=True).mean()
                offset_targets = []
                weights = []
                gt_assigned = []
                centers = []
                seg_weights = []
                seg_targets = []
                for masks, scores, points, labels, proposals in zip(matched_masks, semantic_scores, 
                                                                    gt_points, gt_labels, pseudo_proposals):
                                                                    # gt_points, gt_labels, new_proposals):
                    offset_target_maps, weight_maps, gt_assigned_maps, semantic_centers, seg_maps, seg_weight_maps = \
                        self.single_get_targets(masks, 
                                                scores,
                                                points,
                                                labels,
                                                proposals,
                                                patch_size=patch_size, 
                                                test_cfg=test_cfg)
                    offset_targets.append(offset_target_maps)
                    weights.append(weight_maps)
                    gt_assigned.append(gt_assigned_maps)
                    centers.append(semantic_centers)
                    seg_targets.append(seg_maps)
                    seg_weights.append(seg_weight_maps)

            mil_results = dict(bag_score=bag_score, 
                               matched_cams=matched_cams,
                               refined_multiple_masks=refined_multiple_masks,
                               new_proposals=new_proposals,
                               pseudo_gt_labels=pseudo_gt_labels,
                               pseudo_gt_bboxes=pseudo_gt_bboxes,
                               iou_metric=iou_metric,
                               matched_masks=matched_masks,
                               semantic_scores=semantic_scores,
                               offset_targets=offset_targets,
                               weights=weights,
                               gt_assigned=gt_assigned,
                               centers=centers,
                               seg_targets=seg_targets,
                               seg_weights=seg_weights)
            return mil_results
        
        # 用 attn map生成框, 不用生成,后面直接用pseudo proposals就可以看出来
        else:
            # x = x.permute(0, 2, 1).reshape(num_imgs, -1, *patch_size).contiguous()
            rois = bbox2roi([proposals.reshape(-1, 4) for proposals in pseudo_proposals]) 
            instance_feats = self.instance_extractor(
                [x][:self.instance_extractor.num_inputs], rois)
            bag_score = self.mil_head(instance_feats, num_scale=self.scale)
            # 利用结果生成监督信息, 首先测试自refine的策略，不行再用atten
            with torch.no_grad():
                # matched_cams, semantic_scores = self.post_get_bboxes(pseudo_proposals, 
                pseudo_gt_labels, pseudo_gt_bboxes, matched_cams, matched_masks, \
                    semantic_scores = self.post_get_bboxes(pseudo_proposals,  # 这个可以用预测特征输入的mask
                                                           bag_score,
                                                           gt_labels,
                                                           torch.cat(multiple_cams), 
                                                           torch.cat(multiple_masks), # 但是这个需要用attn的mask
                                                           test_cfg=test_cfg)

                iou_metric = bbox_overlaps(torch.cat(pseudo_gt_bboxes),
                                           torch.cat(gt_bboxes),
                                           is_aligned=True).mean()

            mil_results = dict(bag_score=bag_score, 
                               pseudo_gt_labels=pseudo_gt_labels,
                               pseudo_gt_bboxes=pseudo_gt_bboxes,
                               iou_metric=iou_metric,
                               matched_masks=matched_masks,
                               matched_cams=matched_cams,
                               semantic_scores=semantic_scores)
            return mil_results
        
    def single_get_instance_maps(self, 
                                 pred_offset_map, # scale, 2, h, w
                                 pred_seg_map, # scale, 21, h, w
                                 gt_points, # num_gt, 2
                                 gt_labels, # num_gt
                                 pseudo_proposals, # num_gt, num_scale, 4
                                ):
        num_classes = pred_seg_map.size(1) - 1
        num_gt = gt_points.size(0)
        num_scale = pred_offset_map.size(0)
        h, w = pred_offset_map.size(-2), pred_offset_map.size(-1)
        centers = gt_points.unsqueeze(1).repeat(1, num_scale, 1)
        
        # 1. 获得instance的分割边界区分图像
        y_coord = torch.arange(h, dtype=pred_offset_map.dtype, device=pred_offset_map.device).repeat(1, w, 1).transpose(1, 2)
        x_coord = torch.arange(w, dtype=pred_offset_map.dtype, device=pred_offset_map.device).repeat(1, h, 1)
        coord = torch.cat((x_coord, y_coord), dim=0)
        
        index_maps = [] # num_scale 次循环
        for i, (centers_, offset_map) in enumerate(zip(centers.transpose(1, 0), pred_offset_map)):
            ctr_loc = coord + offset_map
            ctr_loc = ctr_loc.reshape((2, h * w)).transpose(1, 0) # N, 2
            centers_ = centers_.unsqueeze(1) # K, 1, 2
            ctr_loc = ctr_loc.unsqueeze(0) # 1, N, 2
            dist = torch.norm(centers_ - ctr_loc, dim=-1)
            instance_id = torch.argmin(dist, dim=0).reshape((h, w)) + 1 # K, N
            index_maps.append(instance_id)
        index_maps = torch.stack(index_maps) # num_scale, h, w
        
        # 2. 生成semantic segmentation mask
        # 2.1 先生成最终的seg_map 分数
        seg_map = pred_seg_map.softmax(1)
        # 生成所含gt label 的目标的one hot 过滤到不存在图像中的 seg分数
        image_one_hot = torch.zeros(1, num_classes).to(seg_map.device)
        image_one_hot[:, gt_labels] = 1
        # 只去除正例, 0维是背景
        seg_map[:, 1:, ...] *= image_one_hot[:, :, None, None]

        # # 分数滤波, 可视化上好像没用 
        # temp = seg_map[:, 1:, ...].reshape(-1)
        # temp[temp <= 0.2] = 0
        # seg_map_fg = temp.reshape(self.lss_head.scale, 20, img_h, img_w)
        # seg_map[:, 1:, ...] = seg_map_fg
        seg_map_ = seg_map.argmax(1) # 直接选择seg的mask作为输出 # num_scale, h, w
        num_cls_image = len(gt_labels.unique()) # 获得目标的label
        seg_mask_label = []
        for cls in gt_labels.unique(): # label中0是第1类，而seg里面0是背景
            for i, seg in enumerate(seg_map_): # num_scale, h, w # 每个pixel值为 类别label+1
                label_mask = torch.zeros_like(seg).long()
                label_mask[seg == cls + 1] = cls + 1
                seg_mask_label.append(label_mask)
        seg_mask_label = torch.stack(seg_mask_label).reshape(num_cls_image, num_scale, h, w)

        # 3. 结合index_maps 和 semantic segmentation 生成instance segmentation
        seg_mask_label_ = seg_mask_label.transpose(1, 0) # num_scale, num_cls_image, h, w
        index2label = {} # 将gt_index对应到gt label
        for i in range(len(gt_labels)):
            index2label[i + 1] = gt_labels[i]
            
        instance_segmentations = []
        new_proposals = []
        
        for index_map, seg, proposals in zip(index_maps, seg_mask_label_, pseudo_proposals.transpose(1, 0)):
            instance_ids = torch.arange(num_gt).to(index_map.device) + 1
            # instance_ids = index_map.unique() # 选择所有的instance 的 id
            for index, pro in zip(instance_ids, proposals):
                label = index2label[int(index)] # instance 对应的类别
                label2unique = torch.nonzero(seg == label + 1)[:, 0].unique() # 在seg上找到该instance对应的类别的维度
                segmentation = torch.zeros_like(seg[0])
                if len(label2unique) == 0: # 如果没有这一类，我们就没有该类的分割结果,框就用原来的框
                    instance_segmentations.append(segmentation)
                    new_proposals.append(pro.clone())
                else: # 如果有这类
                    segmentation[index_map == index] = 1 
                    segmentation = segmentation.reshape(h, w) * seg[label2unique[0]] # 将该区域乘以seg对应label的区域，获得instance seg
                    instance_segmentations.append(segmentation)
                    
                    coords = segmentation.nonzero()
                    if len(coords) == 0: # 如果没有生成proposal，那么用attn map得到的
                        new_pro = pro.clone()
                    else: # 如果生成了，用自己的
                        xmin = coords[:, 1].min()
                        xmax = coords[:, 1].max()
                        ymin = coords[:, 0].min()
                        ymax = coords[:, 0].max()
                        # min_max 操作，最好这个最有用，完全去点中心点的支撑
                        new_pro = pro.new_tensor([xmin, ymin, xmax, ymax])
                        # expand
                        # xc, yc = gt_p
                        # if abs(xc - xmin) > abs(xc - xmax):
                        #     gt_xmin = xmin
                        #     gt_xmax = xc * 2 -  gt_xmin
                        #     gt_xmax = gt_xmax if gt_xmax < img_w else float(img_w)
                        # else:
                        #     gt_xmax = xmax
                        #     gt_xmin = xc * 2 -  gt_xmax
                        #     gt_xmin = gt_xmin if gt_xmin > 0 else 0.0
                        # if abs(yc - ymin) > abs(yc - ymax):
                        #     gt_ymin = ymin
                        #     gt_ymax = yc * 2 -  gt_ymin
                        #     gt_ymax = gt_ymax if gt_ymax < img_h else float(img_h)
                        # else:
                        #     gt_ymax = ymax
                        #     gt_ymin = yc * 2 -  gt_ymax
                        #     gt_ymin = gt_ymin if gt_ymin > 0 else 0.0
                        # new_p = p.new_tensor([gt_xmin, gt_ymin, gt_xmax, gt_ymax])
                    new_proposals.append(new_pro)
        instance_segmentations = torch.stack(instance_segmentations).reshape(num_scale, num_gt, h, w).transpose(1, 0)
        new_proposals = torch.stack(new_proposals).reshape(num_scale, num_gt, 4).transpose(1, 0)
        
        return instance_segmentations, new_proposals

   # if instance extraction is roi
    def pre_get_bboxes(self, multiple_cams, gt_points, 
                       patch_size=None, test_cfg=None):
        num_imgs = len(gt_points)
        split_lengths = [len(p) for p in gt_points]
        gt_points = torch.cat(gt_points)
        patch_h, patch_w = patch_size
        num_scale = multiple_cams.size(1)
        
        pseudo_proposals = []
        refined_multiple_masks = []
        for cam_per_gt, point in zip(multiple_cams, gt_points):
            for cam in cam_per_gt:
                box, mask = get_bbox_from_cam_fast(cam,
                                        point, 
                                        cam_thr=test_cfg['cam_thr'], 
                                        area_ratio=test_cfg['area_ratio'], 
                                        img_size=(patch_h * 16, patch_w * 16), 
                                        box_method=test_cfg['box_method'],
                                        erode=test_cfg['erode'])
                pseudo_proposals.append(box)
                refined_multiple_masks.append(mask)
        pseudo_proposals = torch.cat(pseudo_proposals).reshape(-1, num_scale, 4)
        pseudo_proposals = list(torch.split(pseudo_proposals, split_lengths, dim=0))
        refined_multiple_masks = torch.stack(refined_multiple_masks).reshape(-1, num_scale, 
                                                                             patch_h * 16, patch_w * 16)
        refined_multiple_masks = list(torch.split(refined_multiple_masks, split_lengths, dim=0))
        return pseudo_proposals, refined_multiple_masks
    
    # if instance extraction is roi
    def post_get_bboxes(self, pseudo_proposals, bag_score, 
                        gt_labels, multiple_cams, multiple_masks, test_cfg=None):
        
        merge_method = test_cfg['merge_method']
        topk_merge = test_cfg['topk']
        topk_matched = test_cfg['topk_']
        num_imgs = len(gt_labels)
        split_lengths = [len(proposals) for proposals in pseudo_proposals]
        
        gt_labels = torch.cat(gt_labels)
        pseudo_proposals = torch.cat(pseudo_proposals)
        num_scale = bag_score.size(-2)
        # 
        index = gt_labels.reshape(-1, 1, 1).repeat(1, num_scale, 1)
        bag_score = torch.gather(bag_score, dim=-1, index=index)[..., 0]
#         _, pseudo_index = bag_score.topk(1)
#         pseudo_index = pseudo_index.reshape(-1, 1, 1).repeat(1, 1, 4)
#         pseudo_gt_bboxes = torch.gather(pseudo_proposals,
#                                         dim=1,
#                                         index=pseudo_index).reshape(-1, 4)

        num_gt = bag_score.size(0)
        scale_scores, pseudo_index = bag_score.topk(topk_merge)
        semantic_scores, selects = bag_score.topk(topk_matched)
#         semantic_scores = bag_score.clone().detach()
        selects = torch.arange(topk_matched).to(semantic_scores.device).reshape(1, -1).repeat(num_gt, 1)
#         semantic_scores = semantic_scores
        semantic_scores = scale_scores.clone().detach()
        matched_cams = torch.gather(multiple_cams, dim=1, 
                                    index=selects.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 
                                                                     multiple_cams.size(-2), 
                                                                     multiple_cams.size(-1)))
        matched_masks = torch.gather(multiple_masks, dim=1, 
                                    index=selects.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 
                                                                     multiple_masks.size(-2), 
                                                                     multiple_masks.size(-1)))
        
        pseudo_index = pseudo_index.reshape(-1, topk_merge, 1).repeat(1, 1, 4)
        pseudo_gt_bboxes = torch.gather(pseudo_proposals,
                                        dim=1,
                                        index=pseudo_index) #.reshape(-1, 4)
        # norm the score
        scale_scores = scale_scores / scale_scores.sum(-1).unsqueeze(-1)
        # average the proposals
        if merge_method == 'top_merge':
            pseudo_gt_bboxes = pseudo_gt_bboxes * scale_scores.unsqueeze(-1)
            pseudo_gt_bboxes = pseudo_gt_bboxes.sum(-2)
        elif merge_method == 'sift_on_box':
            bag_score = bag_score / bag_score.sum(-1).unsqueeze(-1)
            matched_layer_index = torch.arange(num_scale).type_as(
                scale_scores).to(scale_scores.device) * bag_score
            matched_layer_index = matched_layer_index.sum(-1)
            upper_index = matched_layer_index.long() + 1
            lower_index = matched_layer_index.long()
            
            residual_index = torch.cat([
                lower_index.unsqueeze(-1),
                upper_index.unsqueeze(-1)], dim=-1) #.reshape(-1, 2, 1).repeat(1, 1, 4)
            
            # 判断是否超过尺度
            small_scale_inds = (residual_index > num_scale - 1).long().sum(-1).nonzero()
            large_scale_inds = (residual_index < 0).long().sum(-1).nonzero()
            residual_index[small_scale_inds] = residual_index.new_tensor(
                (num_scale - 1, num_scale - 1))
            residual_index[large_scale_inds] = residual_index.new_tensor(
                (0., 0.))
            residual_index = residual_index.reshape(-1, 2, 1).repeat(1, 1, 4)
            residual_proposals = torch.gather(pseudo_proposals,
                                        dim=1,
                                        index=residual_index) #.reshape(-1, 4)
            residual_weights = torch.cat([
                1 - (matched_layer_index - lower_index).unsqueeze(-1),
                (matched_layer_index - lower_index).unsqueeze(-1)], dim=-1)
            
            residual_weights[small_scale_inds] = residual_weights.new_tensor((0.5, 0.5))
            residual_weights[large_scale_inds] = residual_weights.new_tensor((0.5, 0.5))
            pseudo_gt_bboxes = residual_proposals * residual_weights.unsqueeze(-1)
            pseudo_gt_bboxes = pseudo_gt_bboxes.sum(-2)
                
        elif merge_method == 'sift_on_attn':
            assert False, 'no implement'
        pseudo_gt_bboxes = list(torch.split(pseudo_gt_bboxes, split_lengths, dim=0))
        pseudo_gt_labels = list(torch.split(gt_labels, split_lengths, dim=0))
        matched_cams = list(torch.split(matched_cams, split_lengths, dim=0))
        matched_masks = list(torch.split(matched_masks, split_lengths, dim=0))
        semantic_scores = list(torch.split(semantic_scores, split_lengths, dim=0))
        # return pseudo_gt_labels, pseudo_gt_bboxes, matched_cams, matched_cams_, semantic_scores
        return pseudo_gt_labels, pseudo_gt_bboxes, matched_cams, matched_masks, semantic_scores
    
    def single_get_targets(self, 
                           multiple_masks, # use the refined mask, # num_gt, 7, H, W
                           semantic_scores, # num_gt, 7
                           points, # num_gt, 2
                           labels, # num_gt
                           proposals,
                           patch_size=None,
                           test_cfg=None):
        
        patch_h, patch_w = patch_size
        num_gt, num_scale, img_h, img_w = multiple_masks.size()
        
        semantic_centers = points.unsqueeze(1).repeat(1, self.scale, 1)
        # 3. obtain inter mask to assign labels
        # inter_instance_masks = refined_instance_masks.long().sum(0)
        inter_instance_masks = multiple_masks.long().sum(0)
        inter_instance_masks[inter_instance_masks <= 1] = 0
        inter_instance_masks[inter_instance_masks > 1] = 1
        
        # TODO: 应当先分配面积比较大的目标，再分配面积占比小的目标。 否则回出现小响应分配不到
        # TODO：semantic center应当与 unique对应的区域来得到中心，因为这样噪声小
        # 4. assign the certain offset 先给确定是instance 的offset分配index
        gt_assigned_maps = torch.zeros_like(inter_instance_masks, dtype=torch.long) # num_scale, H, W
        # gt_inds = torch.arange(1, len(center_points) + 1).to(offset_target_maps.device)
        gt_inds = torch.arange(1, len(semantic_centers) + 1).to(gt_assigned_maps.device)
        # for i_gt, mask in zip(gt_inds, refined_instance_masks):
        for i_gt, mask in zip(gt_inds, multiple_masks):
            unique_mask = mask ^ inter_instance_masks.bool()
            coords = torch.nonzero(unique_mask)
            gt_assigned_maps[coords[:, 0], coords[:, 1], coords[:, 2]] = i_gt # num_scale, H, W
            
        # 5. assign overlap area 然后对有overlap的instance分配index，根据中心找对应的分配 #  
        for i_scale, inter_mask in enumerate(inter_instance_masks):
            coords = torch.nonzero(inter_mask) # 可能出现没有共同覆盖的目标
            if len(coords) == 0:
                continue
            locations = torch.cat([coords[:, 1:2], coords[:, 0:1]], dim=-1).float()
            point_cost = torch.cdist(locations, semantic_centers[:, i_scale, :], p=2)
            indexes = torch.argmin(point_cost, dim=-1)
            gt_assigned_maps[i_scale][coords[:, 0], coords[:, 1]] = gt_inds[indexes]
        
        # 生成seg map 
        seg_maps = torch.zeros_like(gt_assigned_maps) # 都设为背景
        for i_scale, gt_assigned_map in enumerate(gt_assigned_maps):
            for i_gt, label in enumerate(labels):
                seg_maps[i_scale][gt_assigned_map == i_gt + 1] = label + 1                
        
        # 生成soft seg weight map 背景的权重是1/7 
        seg_weight_maps = torch.ones_like(gt_assigned_maps).to(torch.float32) * (1 / 7)
        for i_scale, (semantic_score, gt_assigned_map) in enumerate(zip(semantic_scores.permute(1, 0), gt_assigned_maps)):
            for i_gt, weight in enumerate(semantic_score):
                seg_weight_maps[i_scale][gt_assigned_map == i_gt + 1] = weight
                
                
        pseudo_proposals = proposals.transpose(1, 0).long()
        remained_weights = []
        box_weights = torch.zeros_like(seg_weight_maps).bool()
        for i_scale, proposals in enumerate(pseudo_proposals):
            for proposal in proposals:
                box_weights[i_scale][proposal[1]:proposal[3], proposal[0]:proposal[2]] = 1
        true_foreground = box_weights.sum(0).bool()

        for seg_w, seg_t in zip(seg_weight_maps, seg_maps):
            remained_weights.append(~(true_foreground ^ seg_t.bool()))
        remained_weights = torch.stack(remained_weights).reshape(num_scale, img_h, img_w)
        seg_weight_maps = remained_weights.long() * seg_weight_maps
        
        # 生成soft weight map
        weight_maps = torch.zeros_like(gt_assigned_maps).to(torch.float32)
        for i_scale, (semantic_score, gt_assigned_map) in enumerate(zip(semantic_scores.permute(1, 0), gt_assigned_maps)):
            for i_gt, weight in enumerate(semantic_score):
                weight_maps[i_scale][gt_assigned_map == i_gt + 1] = weight
                
        # 6. obtain offest maps # TODO: 得到理想的map, 高斯函数应该是旋转的一个scale相关的 能量，是个椭圆，
        # 应该先大后小来填充，#而且如果存在微信上这种情况需要注意, 一定确保在instance上是一致的
        offset_target_maps_xs = torch.zeros_like(gt_assigned_maps, dtype=torch.float32)
        offset_target_maps_ys = torch.zeros_like(gt_assigned_maps, dtype=torch.float32)
        for i_scale, (gt_assigned_map, center_points) in enumerate(zip(gt_assigned_maps, semantic_centers.permute(1, 0, 2))):
            for i_gt, center in zip(gt_inds, center_points):
                instance = gt_assigned_map == i_gt
                coords = torch.nonzero(instance)
                locations = torch.cat([coords[:, 1:2], coords[:, 0:1]], dim=-1).float()
                offset_target_maps_xs[i_scale, coords[:, 0], 
                                coords[:, 1]] = (center - locations)[:, 0]
                offset_target_maps_ys[i_scale, coords[:, 0], 
                                    coords[:, 1]] = (center - locations)[:, 1]
        offset_target_maps = torch.stack([offset_target_maps_xs, offset_target_maps_ys], dim=0).permute(1, 0, 2, 3)
        
        return offset_target_maps, weight_maps, gt_assigned_maps, semantic_centers, seg_maps, seg_weight_maps
    
    def simple_test(self,
                    x,
                    vit_feat,
                    point_tokens,
                    attns,
                    img_metas,
                    gt_bboxes,
                    gt_labels,
                    gt_points,
                    imgs_whwh=None,
                    gt_bboxes_ignore=None,
                    proposals=None, 
                    rescale=False):
        
        patch_h, patch_w = x[2].size(-2), x[2].size(-1)
        if self.with_point:
            points_results = self._point_forward_train(point_tokens, gt_points, 
                                                      gt_labels, img_metas, 
                                                      attns,
                                                      imgs_whwh=imgs_whwh, 
                                                      gt_bboxes=gt_bboxes,
                                                      patch_size=(patch_h, patch_w)
                                                     )
        if self.with_mil:
            mil_results = self._mil_forward_train(vit_feat, 
                                                  points_results['multiple_cams'],
                                                  points_results['refined_multiple_masks'],
                                                  None,
                                                  None,
                                                  points_results['pseudo_proposals'],
                                                  points_results['gt_points'], 
                                                  points_results['gt_labels'],
                                                  points_results['gt_bboxes'],
                                                  patch_size=(patch_h, patch_w),
                                                  test_cfg=self.test_cfg)
            if not rescale:
                scale_factor = img_metas[0]['scale_factor'] 
                pseudo_gt_labels = mil_results['pseudo_gt_labels'] 
                pseudo_gt_bboxes = mil_results['pseudo_gt_bboxes'] 
                gt_bboxes = points_results['gt_bboxes']
                scale_factor = pseudo_gt_bboxes[0].new_tensor(scale_factor)
                pseudo_gt_bboxes[0] /= scale_factor
                scale_factor = gt_bboxes[0].new_tensor(scale_factor)
                gt_bboxes[0] /= scale_factor
            
            return pseudo_gt_labels, pseudo_gt_bboxes, gt_bboxes
        else:
            assert False, 'with_point and with_mil must be True'

def get_bbox_from_cam_fast(cam, point, cam_thr=0.2, area_ratio=0.5, 
                      img_size=None, box_method='expand', erode=False):
    img_h, img_w = img_size
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
    cam[cam >= cam_thr] = 1 # binary the map
    cam[cam < cam_thr] = 0
        
    labeled_components = connected_components_labeling(cam.to(torch.uint8))
    labels = labeled_components.unique()
    
    areas = []
    label_masks = []
    for label in labels: # label=0 为背景 filter the area with little area
        if label == 0:
            continue
        label_mask = (labeled_components == label)
        area = label_mask.sum()
        label_masks.append(label_mask)
        areas.append(area)
    label_masks = torch.stack(label_masks)
    areas = torch.stack(areas)
    max_area = areas.max()
    remained_label_masks = label_masks[areas >= area_ratio * max_area].sum(0).bool()
    # remained_label_mask: value threshold + area threshold
    
    coordinates = torch.nonzero(remained_label_masks).to(torch.float32)
    if len(coordinates) == 0:
        estimated_bbox = cam.new_tensor([[0, 0, 1, 1]])
    else:
        proposal_xmin = coordinates[:, 1].min()
        proposal_ymin = coordinates[:, 0].min()
        proposal_xmax = coordinates[:, 1].max()
        proposal_ymax = coordinates[:, 0].max()
        if box_method == 'min_max':
            estimated_bbox = cam.new_tensor([[proposal_xmin, proposal_ymin, 
                                        proposal_xmax, proposal_ymax]])
        elif box_method == 'expand':
            xc, yc = point
            if abs(xc - proposal_xmin) > abs(xc - proposal_xmax):
                gt_xmin = proposal_xmin
                gt_xmax = xc * 2 -  gt_xmin
                gt_xmax = gt_xmax if gt_xmax < img_w else float(img_w)
            else:
                gt_xmax = proposal_xmax
                gt_xmin = xc * 2 -  gt_xmax
                gt_xmin = gt_xmin if gt_xmin > 0 else 0.0
            if abs(yc - proposal_ymin) > abs(yc - proposal_ymax):
                gt_ymin = proposal_ymin
                gt_ymax = yc * 2 -  gt_ymin
                gt_ymax = gt_ymax if gt_ymax < img_h else float(img_h)
            else:
                gt_ymax = proposal_ymax
                gt_ymin = yc * 2 -  gt_ymax
                gt_ymin = gt_ymin if gt_ymin > 0 else 0.0
            estimated_bbox = cam.new_tensor([[gt_xmin, gt_ymin, gt_xmax, gt_ymax]])
    return estimated_bbox, remained_label_masks
    
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