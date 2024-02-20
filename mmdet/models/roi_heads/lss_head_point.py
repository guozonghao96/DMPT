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

@HEADS.register_module()
class LatentScaleSelectionHead(nn.Module):
    def __init__(self,
                 instance_extractor=None,
                 point_head=None,
                 roi_cls_head=None,
                 threshold_head=None,
                 train_cfg=None,
                 test_cfg=None,
                ):
        super(LatentScaleSelectionHead, self).__init__()
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
        
        if point_head is not None:
            self.init_point_head(point_head)
        if roi_cls_head is not None:
            self.init_roi_cls_head(instance_extractor, roi_cls_head)
        if threshold_head is not None:
            self.init_threshold_head(threshold_head)
            
        self.init_assigner_sampler()
        
    @property
    def with_point(self):
        return hasattr(self, 'point_head') and self.point_head is not None
    
    @property
    def with_threshold(self):
        return hasattr(self, 'threshold_head') and hasattr(self, 'roi_cls_head') \
            and self.threshold_head is not None and self.roi_cls_head is not None
        
    def init_point_head(self, point_head):
        self.point_head = build_head(point_head)
    
    def init_roi_cls_head(self, instance_extractor, roi_cls_head):
        self.instance_extractor = build_roi_extractor(instance_extractor)
        self.roi_cls_head = build_head(roi_cls_head)
        
    def init_threshold_head(self, threshold_head):
        self.threshold_head = build_head(threshold_head)
    
    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.point_assigner = None
        self.point_sampler = None
        if self.train_cfg:
            self.point_assigner = build_assigner(self.train_cfg.assigner)
            self.point_sampler = build_sampler(
                self.train_cfg.sampler, context=self)

    def init_weights(self):
        if self.with_threshold:
            self.instance_extractor.init_weights()
            self.threshold_head.init_weights()
            self.roi_cls_head.init_weights()
            
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
        attns_maps = attns_maps[-self.scale:]
        # 1. filter the noise 
        if self.filter:
            bs = attns_maps.size(1)
            num_patches = attns_maps.size(-1)
            for i in range(len(attns_maps)):
                flat = attns_maps[i].reshape(bs, -1)
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
            joint_attentions = joint_attentions[:, -self.scale, ...].unsqueeze(1)
        elif self.scale_method == 'auto':
            joint_attentions = joint_attentions[:, -self.scale:, ...]
        
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
    
    
    
#     def generate_multi_scale_cams(self, attns_maps, point_results, 
#                                   align_label=True):
#         '''
#         input:
#             attns_maps -> (blocks, batch_size, N, N)
#             #只取需要的层数
#             -> attns_maps -> (scale, batch_size, N, N)
#             #得到多尺度的attns
#             -> joint_attentions -> (scale, batch_size, N, N)
#             -> joint_attentions -> (batch_size, scale, N, N)
#             #取出match的object的attention maps,并resize成maps
#             -> point_attentions -> (batch_size, scale, num_gt, patch_h * patch_w)
#         output:
#             point_attentions -> (batch_size, scale, num_gt, patch_h * patch_w)
#         '''
#         # 1. filter the noise 
#         if self.filter:
#             bs = attns_maps.size(1)
#             num_patches = attns_maps.size(-1)
#             for i in range(len(attns_maps)):
#                 flat = attns_maps[i].reshape(bs, -1)
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
        
#         # 4. get point matched attention maps
#         pos_inds = point_results['pos_inds']        
#         num_imgs = len(pos_inds)
#         num_points = point_results['cls_score'].size(1)
        
#         multiple_cams = []
#         for i in range(num_imgs):
#             pos_inds_per_img = pos_inds[i]
#             point_attn_maps_per_img = joint_attentions[i, :, -num_points:, 1:-num_points]
#             matched_point_attn_maps_per_img = point_attn_maps_per_img[:, pos_inds_per_img] # (scale, num_gt, num_patches)
#             matched_point_attn_maps_per_img = matched_point_attn_maps_per_img.permute(1, 0, 2) # (num_gt, scale, num_patches)
#             multiple_cams.append(matched_point_attn_maps_per_img)
#         multiple_cams = torch.cat(multiple_cams)
#         # 4. align oders of "gt_labels" and "gt_points" to that of "pos_inds"
#         gt_labels = []
#         gt_points = []
#         labels = point_results['point_targets'][0].reshape(-1, num_points)
#         points = point_results['point_targets'][2].reshape(-1, num_points, 2)
        
#         for i in range(num_imgs):
#             pos_inds_per_img = pos_inds[i]
#             labels_per_img = labels[i]
#             points_per_img = points[i]
            
#             gt_labels.append(labels_per_img[pos_inds_per_img])
#             gt_points.append(points_per_img[pos_inds_per_img])        
#         # 5. align oders of "gt_bboxes" to that of "pos_inds"
#         gt_bboxes_ = []
#         pos_assigned_gt_inds = point_results['pos_assigned_gt_inds']
#         gt_bboxes = point_results['gt_bboxes']
#         for i in range(num_imgs):
#             gt_inds = pos_assigned_gt_inds[i]
#             bboxes = gt_bboxes[i]
#             gt_bboxes_.append(bboxes[gt_inds])
            
#         return multiple_cams, gt_labels, gt_points, gt_bboxes_
    
    def forward_train(self,
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
        losses = dict()
        patch_h, patch_w = x[2].size(-2), x[2].size(-1)
        if self.with_point:
            point_results = self._point_forward_train(point_tokens, gt_points, 
                                                      gt_labels, img_metas, 
                                                      imgs_whwh=imgs_whwh, 
                                                      gt_bboxes=gt_bboxes)
            losses.update(point_results['loss_point'])
            
        if self.with_threshold:
            threshold_results = self._threshold_forward_train(vit_feat, attns, gt_points,
                                                              gt_labels, point_results, 
                                                              patch_size=(patch_h, patch_w),
                                                              test_cfg=self.test_cfg)
            losses.update(threshold_results['loss_roi_cls'])
            return threshold_results['gt_labels'], threshold_results['pseudo_proposals'], losses, threshold_results['iou_metric']
        # baseline use averaged activated map or last map
        else:
            return None, None, losses
            
    def _point_forward_train(self, x, gt_points, gt_labels, img_metas, imgs_whwh=None, gt_bboxes=None):
        # inference in mlp_head
        point_results = self._point_forward(x)
        loss_point = self.point_head.loss(point_results['cls_score'],
                                          torch.cat(gt_labels))
        point_results.update(loss_point=loss_point)
        return point_results

    def _point_forward(self, x):
        cls_score = self.point_head(x)
        point_results = dict(
            cls_score=cls_score, point_tokens=x)
        return point_results
    
    def _threshold_forward_train(self, vit_feat, attns, gt_points, 
                                 gt_labels, point_results, patch_size=None,
                                 test_cfg=None):
        threshold_results = self._threshold_forward(vit_feat, attns, point_results, patch_size,
                                                    test_cfg=test_cfg)
        loss_roi_cls = self.roi_cls_head.loss(threshold_results['scores'], 
                                              threshold_results['gt_labels'],
                                              threshold_results['valid_weights']
                                             )
        threshold_results.update(loss_roi_cls=loss_roi_cls)
        return threshold_results
    
    def _threshold_forward(self, x, attns, point_results, patch_size=None,
                           test_cfg=None):
        num_imgs = x.size(0)
        patch_h, patch_w = patch_size
        # (num_gt1 + num_gt2, scale, patch_h * patch_w)
        multiple_cams, gt_labels, gt_points, gt_bboxes = self.generate_multi_scale_cams(attns, point_results)
        num_gt_per_batch = [len(labels) for labels in gt_labels]
        num_gt = sum(num_gt_per_batch)
        # (num_gt1 + num_gt2, 1)
        thresholds = self.threshold_head(x, num_gt_per_batch) # num_gt, 1 | num_gt, 1
        multiple_cams_ori = F.interpolate(multiple_cams.reshape(num_gt, -1, *patch_size), 
                                     (patch_h * 16, patch_w * 16), 
                                     mode='bicubic') # num_gt, num_scale, H, W
        pseudo_proposals, valid_weights = self.get_proposals(multiple_cams_ori, 
                                              thresholds.detach())
        pseudo_proposals = list(torch.split(pseudo_proposals, num_gt_per_batch, dim=0))
        
        rois = bbox2roi([proposals.reshape(-1, 4) for proposals in pseudo_proposals])
        x = x.permute(0, 2, 1).reshape(num_imgs, -1, *patch_size).contiguous()
        instance_feats = self.instance_extractor(
            [x][:self.instance_extractor.num_inputs], rois) # num_gt, c, s, s
        
        roi_size = instance_feats.size(-1)
        threshold_feat = thresholds.reshape(num_gt, -1, 1, 1).repeat(1, 1, roi_size, roi_size)
        instance_feats = torch.cat([instance_feats, threshold_feat], dim=1)
        scores = self.roi_cls_head(instance_feats)
        
        iou_metric = bbox_overlaps(torch.cat(pseudo_proposals),
                                   torch.cat(gt_bboxes), is_aligned=True)
        roi_cls_results = dict(scores=scores,
                               gt_labels=gt_labels,
                               multiple_cams_ori=multiple_cams_ori,
                               pseudo_proposals=pseudo_proposals,
                               gt_bboxes=gt_bboxes,
                               thresholds=thresholds,
                               valid_weights=valid_weights,
                               iou_metric=iou_metric.mean())
        return roi_cls_results
    
    # if instance extraction is "attn" and mil_head is performed
    def get_proposals(self, multiple_cams_ori, thresholds):
        cams_ori = multiple_cams_ori[:, 0, ...]
        num_gt, img_h, img_w = cams_ori.size()
        # 1. norm the attention maps
        cams_mins = cams_ori.flatten(1).min(1)[0].reshape(num_gt, 1, 1)
        cams_maxs = cams_ori.flatten(1).max(1)[0].reshape(num_gt, 1, 1)
        cams_norm = (cams_ori - cams_mins) / (cams_maxs - cams_mins)
        
        proposals = []
        valid_weights = []
        # 2. get the attn masks, and obtain proposals
        for cams_per_gt, threshold in zip(cams_norm, thresholds):
            attn_mask = cams_per_gt > threshold
            coordinates = torch.nonzero(attn_mask).to(torch.float32)
            if len(coordinates) != 0:
                xmin = coordinates[:, 1].min()
                xmax = coordinates[:, 1].max()
                ymin = coordinates[:, 0].min()
                ymax = coordinates[:, 0].max()
                proposal = coordinates.new_tensor((xmin, ymin, xmax, ymax))
                weight = coordinates.new_tensor((1.))
            else:
                proposal = coordinates.new_tensor((0, 0, 1, 1))
                weight = coordinates.new_tensor((0.))
                
            proposals.append(proposal)
            valid_weights.append(weight)
                
        proposals = torch.stack(proposals)
        valid_weights = torch.stack(valid_weights)
        return proposals, valid_weights
    
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
        if self.with_point and self.with_mil:
            point_results = self._point_forward_train(point_tokens, gt_points, 
                                                      gt_labels, img_metas, 
                                                      imgs_whwh=imgs_whwh,
                                                     gt_bboxes=gt_bboxes)
            mil_results = self._mil_forward_train(vit_feat, attns, gt_points,
                                                  gt_labels, point_results, 
                                                  patch_size=(patch_h, patch_w),
                                                  test_cfg=self.test_cfg)
            if not rescale:
                scale_factor = img_metas[0]['scale_factor'] 
                pseudo_gt_labels = mil_results['pseudo_gt_labels'] 
                pseudo_gt_bboxes = mil_results['pseudo_gt_bboxes'] 
                gt_bboxes = mil_results['gt_bboxes']
                scale_factor = pseudo_gt_bboxes[0].new_tensor(scale_factor)
                pseudo_gt_bboxes[0] /= scale_factor
                scale_factor = gt_bboxes[0].new_tensor(scale_factor)
                gt_bboxes[0] /= scale_factor
            
            return pseudo_gt_labels, pseudo_gt_bboxes, gt_bboxes
        else:
            assert False, 'with_point and with_mil must be True'
    
