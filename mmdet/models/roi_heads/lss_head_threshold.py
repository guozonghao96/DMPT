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

@HEADS.register_module()
class LatentScaleSelectionHead(nn.Module):
    def __init__(self,
                 instance_extractor=None,
                 point_head=None,
                 mil_head=None,
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
        if self.with_mil:
            mil_results = self._mil_forward_train(vit_feat, attns, gt_points,
                                                  gt_labels, point_results, 
                                                  patch_size=(patch_h, patch_w),
                                                  test_cfg=self.test_cfg)
            losses.update(mil_results['loss_mil'])
            return mil_results['pseudo_gt_labels'], mil_results['pseudo_gt_bboxes'], losses
        # baseline use averaged activated map or last map
        else:
            return None, None, losses
            
    def _point_forward_train(self, x, gt_points, gt_labels, img_metas, imgs_whwh=None, gt_bboxes=None):
        # x -> point tokens -> (batch_size, point_num, c)
        num_imgs = len(img_metas)
        num_proposals = x.size(1)
        imgs_whwh = imgs_whwh.repeat(1, num_proposals, 1)
        # inference in mlp_head
        point_results = self._point_forward(x)
        cls_score = point_results['cls_score'].detach()
        point_pred = point_results['point_pred'].detach()
        # get assign and sample results
        point_assign_results = []
        for i in range(num_imgs):
            assign_result = self.point_assigner.assign(
                point_pred[i], cls_score[i], gt_points[i],
                gt_labels[i], img_metas[i])
            point_sampling_result = self.point_sampler.sample(
                assign_result, point_pred[i], gt_points[i]
            )
            point_assign_results.append(point_sampling_result)  
        # matched for generating box
        point_results.update(pos_inds=[sample_results.pos_inds 
                                       for sample_results in point_assign_results]) # pos_inds
        # align the gt_bboxes
        point_results.update(pos_assigned_gt_inds=[sample_results.pos_assigned_gt_inds 
                                       for sample_results in point_assign_results]) # pos_inds
        point_results.update(gt_bboxes=gt_bboxes) # pos_inds
        # get point targets
        point_targets = self.point_head.get_targets(point_assign_results, gt_points,
                                                  gt_labels, self.train_cfg, True)
        point_results.update(point_targets=point_targets)
        # point loss
        loss_point = self.point_head.loss(point_results['cls_score'],
                                          point_results['point_pred'],
                                          *point_targets, imgs_whwh=imgs_whwh)
        point_results.update(loss_point=loss_point)
        return point_results

    def _point_forward(self, x):
        cls_score, point_pred = self.point_head(x)
        point_results = dict(
            cls_score=cls_score, point_pred=point_pred, point_tokens=x)
        return point_results
    
    def _mil_forward_train(self, vit_feat, attns, gt_points, 
                           gt_labels, point_results, patch_size=None,
                           test_cfg=None):
        mil_results = self._mil_forward(vit_feat, attns, point_results, patch_size,
                                        test_cfg=test_cfg)
        
        loss_mil = self.roi_cls_head.loss(mil_results['scores'], 
                                      mil_results['gt_labels'])
        mil_results.update(loss_mil=loss_mil)
        return mil_results
    
    def _mil_forward(self, x, attns, point_results, patch_size=None,
                     test_cfg=None):
        num_imgs = x.size(0)
        patch_h, patch_w = patch_size
        # (num_gt1 + num_gt2, scale, patch_h * patch_w)
        multiple_cams, gt_labels, gt_points, gt_bboxes = self.generate_multi_scale_cams(attns, point_results)
        
        num_gt_per_batch = [len(gt_labels) for labels in gt_labels]
        # (num_gt1 + num_gt2, 1)
        thresholds, threshold_feat = self.threshold_head(x, num_gt_per_batch) # num_gt, 1 | num_gt, 1
        print('threshold', thresholds.size())
        print('threshold_feat', threshold_feat.size())
        multiple_cams_ori = F.interpolate(multiple_cams, 
                                     (patch_h * 16, patch_w * 16), 
                                     mode='bicubic') # num_gt, num_scale, H, W
        print('attns', multiple_cams_ori.size())
        pseudo_proposals = self.get_proposals(multiple_cams_ori, 
                                              thresholds.detach())
        print('pseudo_propopsals', pseudo_proposals.size())
        pseudo_proposals = list(torch.split(pseudo_proposals, num_gt_per_batch, dim=0))

        rois = bbox2roi([proposals.reshape(-1, 4) for proposals in pseudo_proposals])
        x = x.permute(0, 2, 1).reshape(num_imgs, -1, *patch_size).contiguous()
        instance_feats = self.instance_extractor(
            [x][:self.instance_extractor.num_inputs], rois) # num_gt, c, s, s
        roi_size = instance_feats.size(-1)
        threshold_feat.reshape(-1, 1, 1, 1).repeat(1, 1, roi_size, roi_size)
        instance_feats = torch.cat([instance_feats, threshold_feat], dim=1)
        scores = self.roi_cls_head(instance_feats)
        print('scores', scores.size())
        mil_results = dict(scores=scores, 
                           gt_labels=gt_labels,
                           multiple_cams_ori=multiple_cams_ori,
                           pseudo_proposals=pseudo_proposals,
                           gt_bboxes=gt_bboxes,
                           thresholds=thresholds)
        
        return mil_results
    # if instance extraction is "attn" and mil_head is performed
    def get_proposals(self, multiple_cams_ori, thresholds):
        cams_ori = multiple_cams_ori[:, 0, ...]
        num_gt, img_h, img_w = cams_ori.size()
        # 1. norm the attention maps
        cams_mins = cams_ori.flatten(1).min(1)[0].reshape(num_gt, 1, 1)
        cams_maxs = cams_ori.flatten(1).max(1)[0].reshape(num_gt, 1, 1)
        cams_norm = (cams_ori - cams_mins) / (cams_maxs - cams_mins)
        
        proposals = []
        # 2. get the attn masks, and obtain proposals
        for cams_per_gt, threshold in zip(cams_norm, thresholds):
            attn_masks = cams_per_gt > threshold
            coordinates = attn_masks.nonzero(mask).to(torch.float32)
            xmin = coordinates[:, 1].min()
            xmax = coordinates[:, 1].max()
            ymin = coordinates[:, 0].min()
            ymax = coordinates[:, 0].max()
            proposal = ch.new_tensor((xmin, ymin, xmax, ymax))
            proposals.append(proposal)
        proposals = torch.stack(proposals)
        return proposals
    
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
    
