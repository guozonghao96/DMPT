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
            return mil_results['pseudo_gt_labels'], mil_results['pseudo_gt_bboxes'], losses, mil_results['iou_metric'], mil_results['offset_targets'], mil_results['weights'], mil_results['gt_points'], mil_results['pseudo_proposals'], mil_results['snake_targets_mask'], mil_results['max_area_center_points_'], mil_results['matched_masks'], mil_results['points'], mil_results['gt_points'], mil_results['multiple_cams'], mil_results['matched_cams']
        # baseline use averaged activated map or last map
        else:
            multiple_cams, gt_labels, gt_points, gt_bboxes = self.generate_multi_scale_cams(attns, point_results)
            pseudo_gt_labels, pseudo_gt_bboxes, iou_metric = self.get_single_bboxes(multiple_cams, gt_points, gt_labels, gt_bboxes,
                                                          patch_size=(patch_h, patch_w), test_cfg=self.test_cfg)
            # return None, None, losses
            return pseudo_gt_labels, pseudo_gt_bboxes, losses, iou_metric
            
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
        loss_mil = self.mil_head.loss(mil_results['bag_score'], 
                                      mil_results['pseudo_gt_labels'])
        mil_results.update(loss_mil=loss_mil)
        return mil_results
    
    def _mil_forward(self, x, attns, point_results, patch_size=None,
                     test_cfg=None):
        # (num_gt1 + num_gt2, scale, patch_h * patch_w)
        patch_h, patch_w = patch_size
        multiple_cams, gt_labels, gt_points, gt_bboxes = self.generate_multi_scale_cams(attns, point_results)
        split_length = [len(bboxes) for bboxes in gt_bboxes]
        # 每个attn maps 变成 ori 尺度大小，再进行生成框操作
        multiple_cams = multiple_cams.reshape(-1, multiple_cams.size(1), patch_h, patch_w)
        multiple_cams = F.interpolate(multiple_cams, 
                                     (patch_h * 16, patch_w * 16), 
                                     mode='bilinear') # num_gt, num_scale, H, W
        if self.pooling_type == 'roi':
            num_imgs = len(gt_points)
            pseudo_proposals, refined_multiple_masks = self.pre_get_bboxes(multiple_cams,
                                                gt_points,
                                                patch_size=patch_size,
                                                test_cfg=test_cfg)
            
            # refined_multiple_masks 是 二值滤波 并 面积滤波的mask
            # 
            # x --> (bs, n, c)
            x = x.permute(0, 2, 1).reshape(num_imgs, -1, *patch_size).contiguous()
            rois = bbox2roi([proposals.reshape(-1, 4) for proposals in pseudo_proposals])
            
            instance_feats = self.instance_extractor(
                [x][:self.instance_extractor.num_inputs], rois)
            
            # 
            # pseudo_proposals_ = torch.cat(pseudo_proposals) # num_gt, 7, 4
            # gt_bboxes_ = torch.cat(gt_bboxes).unsqueeze(1).repeat(1, self.scale, 1) # num_gt, 1, 4
            # loc_scores = bbox_overlaps(pseudo_proposals_.reshape(-1, 4),
            #                            gt_bboxes_.reshape(-1, 4),
            #                            is_aligned=True).reshape(-1, self.scale).softmax(-1)
            
            bag_score = self.mil_head(instance_feats, num_scale=self.scale)
            pseudo_gt_labels, pseudo_gt_bboxes, matched_masks, matched_cams = self.post_get_bboxes(pseudo_proposals, 
                                                                    bag_score,
                                                                    gt_labels,
                                                                    refined_multiple_masks,
                                                                    multiple_cams,
                                                                    test_cfg=test_cfg)
            iou_metric = bbox_overlaps(torch.cat(pseudo_gt_bboxes),
                                       torch.cat(gt_bboxes),
                                       is_aligned=True).mean()
            # prepare targets
            offset_targets = []
            instances = []
            refined_instances = []
            points = []
            weights_ = []
            snake_targets = []
            max_area_center_points_ = []
            for masks in matched_masks:
                offset_target_map, instance_masks, refined_instance_masks, center_points, \
                weights, instance_labels, snake_target_masks, max_area_center_points = \
                    self.single_get_targets(masks, 
                                            patch_size=patch_size, 
                                            test_cfg=test_cfg)
                offset_targets.append(offset_target_map)
                instances.append(instance_masks)
                refined_instances.append(refined_instance_masks)
                points.append(center_points)
                weights_.append(weights)
                snake_targets.append(snake_target_masks)
                max_area_center_points_.append(max_area_center_points)
            offset_targets = torch.stack(offset_targets) # batch, 2, H, W
            instances = torch.cat(instances) # num_gt, H, W
            refined_instances = torch.cat(refined_instances) # num_gt, H, W
            points = torch.cat(points) # num_gt, 2 (attention map响应的 centers)
            weights_ = torch.stack(weights_)  #batch,.1, H, W,  (预测offset 的正例权重)
            snake_targets = torch.cat(snake_targets) # num_gt, H, W (连通域最大的attention mask)
            max_area_center_points_ = torch.cat(max_area_center_points_) # num_gt, 2 (连通域最大的响应的 centers)
            # generate offset map
            mil_results = dict(bag_score=bag_score, 
                               multiple_cams=multiple_cams,
                               pseudo_proposals=pseudo_proposals,
                               pseudo_gt_labels=pseudo_gt_labels,
                               pseudo_gt_bboxes=pseudo_gt_bboxes,
                               gt_bboxes=gt_bboxes,
                               gt_points=gt_points,
                               gt_labels=gt_labels,
                               iou_metric=iou_metric,
                               matched_masks=matched_masks,
                               matched_cams=matched_cams,
                               offset_targets=offset_targets,
                               instances=instances,
                               refined_instances=refined_instances,
                               points=points,
                               weights=weights_,
                               instance_labels=instance_labels,
                               snake_targets_mask=snake_targets,
                               max_area_center_points_=max_area_center_points_,
                               refined_multiple_masks=refined_multiple_masks
                              )
            return mil_results
        
#         elif self.pooling_type == 'attn':
#             assert False, 'its performance is poor, so we never support this method'
#             pos_inds = point_results['pos_inds']
#             split_lengths = [len(inds) for inds in pos_inds]
#             multiple_cams = list(torch.split(multiple_cams, split_lengths, dim=0))
            
#             instance_feats = self.instance_extractor(x, multiple_cams, patch_size=patch_size)
#             if isinstance(instance_feats, tuple):
#                 bag_score = self.mil_head(instance_feats[0], instance_feats[1])
#             else:
#                 bag_score = self.mil_head(instance_feats)
            
#             multiple_cams = torch.cat(multiple_cams)
#             pseudo_gt_labels, pseudo_gt_bboxes = self.get_bboxes(multiple_cams, 
#                                                                 bag_score, 
#                                                                 gt_points, 
#                                                                 gt_labels,
#                                                                 patch_size=patch_size,
#                                                                 test_cfg=test_cfg)
#             mil_results = dict(bag_score=bag_score,
#                                multiple_cams=multiple_cams,
#                                pseudo_gt_labels=pseudo_gt_labels,
#                                pseudo_gt_bboxes=pseudo_gt_bboxes)
#             return mil_results

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
        return pseudo_proposals, refined_multiple_masks
    
    # if instance extraction is roi
    def post_get_bboxes(self, pseudo_proposals, bag_score, 
                        gt_labels, multiple_cams, multiple_cams_, test_cfg=None):
        
        merge_method = test_cfg['merge_method']
        topk_merge = test_cfg['topk']
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

        scale_scores, pseudo_index = bag_score.topk(topk_merge)
        matched_cams = torch.gather(multiple_cams, dim=1, 
                                    index=pseudo_index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 
                                                                     multiple_cams.size(-2), 
                                                                     multiple_cams.size(-1)))
        matched_cams_ = torch.gather(multiple_cams_, dim=1, 
                                    index=pseudo_index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 
                                                                     multiple_cams.size(-2), 
                                                                     multiple_cams.size(-1)))
        
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
        matched_cams_ = list(torch.split(matched_cams_, split_lengths, dim=0))
        return pseudo_gt_labels, pseudo_gt_bboxes, matched_cams, matched_cams_
    
    def single_get_targets(self, 
                           multiple_cams, # use the refined mask
                           patch_size=None,
                           test_cfg=None):
        patch_h, patch_w = patch_size
        num_gt = multiple_cams.size(0)
        
        # 1. 直接对mask进行闭运算,去掉中空区域
        instance_masks = erode(dilate(multiple_cams)).squeeze(1)
        # 2. 找到最大联通区域，当作伪标签的context
        snake_target_masks = []
        refined_instance_masks = []
        center_points = []
        max_area_center_points = []
        for mask in instance_masks:
            labeled_components = connected_components_labeling(mask.to(torch.uint8))
            labels = labeled_components.unique()
            max_area = 0
            max_label_map = 0
            for label in labels: # label=0 为背景
                if label == 0:
                    continue
                label_mask = (labeled_components == label)
                area = label_mask.sum()
                if area > max_area:
                    max_area = area
                    max_label_map = label_mask
            # coords = torch.nonzero(max_label_map).float()
            
            coords = torch.nonzero(mask).float()
            center_point = torch.as_tensor([coords[:, 1].mean(),
                                            coords[:, 0].mean()],
                                          device=instance_masks.device)
            coords = torch.nonzero(max_label_map).float()
            max_area_center_point = torch.as_tensor([coords[:, 1].mean(),
                                                    coords[:, 0].mean()],
                                          device=instance_masks.device)
            snake_target_masks.append(max_label_map)
            refined_instance_masks.append(mask)
            center_points.append(center_point)
            max_area_center_points.append(max_area_center_point)
            
        snake_target_masks = torch.stack(snake_target_masks)
        refined_instance_masks = torch.stack(refined_instance_masks)
        center_points = torch.stack(center_points)
        max_area_center_points = torch.stack(max_area_center_points)
        
        # 3. obtain inter mask to assign labels
        inter_instance_masks = refined_instance_masks.long().sum(0)
        inter_instance_masks[inter_instance_masks <= 1] = 0
        inter_instance_masks[inter_instance_masks > 1] = 1
        
        # 4. assign the certain offset 先给确定是instance 的offset分配index
        offset_target_maps = torch.zeros_like(inter_instance_masks, dtype=torch.long)
        gt_inds = torch.arange(1, len(center_points) + 1).to(offset_target_maps.device)
        for i_gt, mask in zip(gt_inds, refined_instance_masks):
            unique_mask = mask ^ inter_instance_masks.bool()
            coords = torch.nonzero(unique_mask)
            offset_target_maps[coords[:, 0], coords[:, 1]] = i_gt
            
        # 5. assign overlap area 然后对有overlap的instance分配index，根据中心找对应的分配
        coords = torch.nonzero(inter_instance_masks)
        locations = torch.cat([coords[:, 1:2], coords[:, 0:1]], dim=-1).float()
        point_cost = torch.cdist(locations, center_points, p=2)
        indexes = torch.argmin(point_cost, dim=-1)
        offset_target_maps[coords[:, 0], coords[:, 1]] = gt_inds[indexes]
        
        # 6. obtain offest maps # TODO: 得到理想的map, 高斯函数应该是旋转的一个scale相关的 能量，是个椭圆，
        # 应该先大后小来填充，#而且如果存在微信上这种情况需要注意, 一定确保在instance上是一致的
        offset_target_maps_xs = torch.zeros_like(offset_target_maps, dtype=torch.float32)
        offset_target_maps_ys = torch.zeros_like(offset_target_maps, dtype=torch.float32)
        for i_gt, center in zip(gt_inds, center_points):
            instance = offset_target_maps == i_gt
            coords = torch.nonzero(instance)
            locations = torch.cat([coords[:, 1:2], coords[:, 0:1]], dim=-1).float()
            
            offset_target_maps_xs[coords[:, 0], 
                                coords[:, 1]] = (center - locations)[:, 0]
            offset_target_maps_ys[coords[:, 0], 
                                coords[:, 1]] = (center - locations)[:, 1]
        offset_target_maps_ = torch.stack([offset_target_maps_xs, offset_target_maps_ys], dim=0)
        return offset_target_maps_, instance_masks, refined_instance_masks, center_points, \
               refined_instance_masks.sum(0).bool(), offset_target_maps, snake_target_masks, \
               max_area_center_points
    
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

#     # if instance extraction is "attn" and mil_head is performed
#     def get_bboxes(self, multiple_cams, bag_score, 
#                         gt_points, gt_labels, patch_size=None, 
#                         test_cfg=None):
        
#         split_lengths = [len(p) for p in gt_points]
#         patch_h, patch_w = patch_size
#         # 选择bag实际标签的score分数
#         pseudo_gt_labels = gt_labels
        
#         gt_points = torch.cat(gt_points)
#         gt_labels = torch.cat(gt_labels)   
#         num_scale = bag_score.size(-2)
#         # 
#         index = gt_labels.reshape(-1, 1, 1).repeat(1, num_scale, 1)
#         bag_score = torch.gather(bag_score, dim=-1, index=index)[..., 0]
#         _, pseudo_index = bag_score.topk(1)
#         # 获得响应尺度响应最高的层标号
#         # pseudo_index (num_gt, topk)
#         # multiple_cams (num_gt, scale, h*w)
#         pseudo_index = pseudo_index.reshape(
#             pseudo_index.size(0), 1, 1).repeat(1, 1, multiple_cams.size(-1))
#         matched_cams = torch.gather(multiple_cams, dim=1, index=pseudo_index)[:, 0, :]
#         matched_cams = matched_cams.reshape(-1, patch_h, patch_w)
#         # 变成原图大小并进行后处理
#         matched_cams = F.interpolate(matched_cams.unsqueeze(1), 
#                                      (patch_h * 16, patch_w * 16), 
#                                      mode='bilinear').squeeze(1)
#         pseudo_gt_bboxes = []
#         refined_
# #         for cam, point in zip(matched_cams.detach().cpu().numpy(), gt_points.detach().cpu().numpy()):
#         for cam, point in zip(matched_cams, gt_points.detach().cpu().numpy()):
# #             box = get_bbox_from_cam(cam, 
#             box, remained_label_masks = get_bbox_from_cam_fast(cam,
#                                     point, 
#                                     cam_thr=test_cfg['cam_thr'], 
#                                     area_ratio=test_cfg['area_ratio'], 
#                                     img_size=(patch_h * 16, patch_w * 16), 
#                                     box_method=test_cfg['box_method'])
#             pseudo_gt_bboxes.append(torch.as_tensor(box, 
#                                                     dtype=gt_points.dtype, 
#                                                     device=gt_points.device))
#         pseudo_gt_bboxes = torch.cat(pseudo_gt_bboxes)
#         pseudo_gt_bboxes = list(torch.split(pseudo_gt_bboxes, split_lengths, dim=0))
#         return pseudo_gt_labels, pseudo_gt_bboxes
    
#     # if instance extraction is "attn" and mil_head is excluded
#     def get_single_bboxes(self, multiple_cams, gt_points, gt_labels, 
#                           gt_bboxes, patch_size=None, test_cfg=None):
#         num_gt = multiple_cams.size(0)
#         split_lengths = [len(p) for p in gt_points]
#         patch_h, patch_w = patch_size
#         # 选择bag实际标签的score分数
#         pseudo_gt_labels = gt_labels
#         gt_points = torch.cat(gt_points)
#         matched_cams = multiple_cams.reshape(-1, patch_h, patch_w)
#         # 变成原图大小并进行后处理
#         matched_cams = F.interpolate(matched_cams.unsqueeze(1), 
#                                      (patch_h * 16, patch_w * 16), 
#                                      mode='bilinear').squeeze(1)
#         # norm
#         multiple_cams_min = matched_cams.flatten(1).min(1)[0].reshape(num_gt, 1, 1)
#         multiple_cams_max = matched_cams.flatten(1).max(1)[0].reshape(num_gt, 1, 1)
#         multiple_cams_norm = (matched_cams - multiple_cams_min) / (multiple_cams_max - multiple_cams_min)
        
#         # auto set some threshold
#         threshold_ = [0.1, 0.6]
#         times = 12
#         threshold = np.linspace(threshold_[0], threshold_[1], times)
#         proposals = []
#         for i_gt, cams in enumerate(multiple_cams_norm):
#             centers_per_gt = []
#             proposals_per_gt = []
#             for t in threshold:
#                 mask = cams > t
#                 coordinates = torch.nonzero(mask).to(torch.float32)
#                 xmin = coordinates[:, 1].min()
#                 xmax = coordinates[:, 1].max()
#                 ymin = coordinates[:, 0].min()
#                 ymax = coordinates[:, 0].max()
                
#                 xc, yc = (xmin + xmax) / 2, (ymin + ymax) / 2
#                 proposal = coordinates.new_tensor((xmin, ymin, xmax, ymax))
#                 proposal_center = coordinates.new_tensor((xc, yc))
#             centers_per_gt.append(proposal_center)
#             centers_per_gt = torch.stack(centers_per_gt) # 10, 2
            
#             proposals_per_gt.append(proposal)
#             proposals_per_gt = torch.stack(proposals_per_gt)
            
#             # find the points with the smallest distance
#             gt_point = gt_points[i_gt].reshape(1, 2)
#             distances = ((gt_point - centers_per_gt) ** 2).sum(-1).sqrt()
#             min_index = torch.argmin(distances)
#             proposals.append(proposals_per_gt[min_index])
#         proposals = torch.stack(proposals)
#         iou_metric = bbox_overlaps(proposals, torch.cat(gt_bboxes)).mean()
#         pseudo_gt_bboxes = list(torch.split(proposals, split_lengths, dim=0))
#         return pseudo_gt_labels, pseudo_gt_bboxes, iou_metric
    
    
    
#     # if instance extraction is "attn" and mil_head is excluded
#     def get_single_bboxes(self, multiple_cams, gt_points, gt_labels, 
#                           patch_size=None, test_cfg=None):
        
#         split_lengths = [len(p) for p in gt_points]
#         patch_h, patch_w = patch_size
#         # 选择bag实际标签的score分数
#         pseudo_gt_labels = gt_labels
#         gt_points = torch.cat(gt_points)
#         matched_cams = multiple_cams.reshape(-1, patch_h, patch_w)
#         # 变成原图大小并进行后处理
#         matched_cams = F.interpolate(matched_cams.unsqueeze(1), 
#                                      (patch_h * 16, patch_w * 16), 
#                                      mode='bilinear').squeeze(1)
#         pseudo_gt_bboxes = []
#         for cam, point in zip(matched_cams.detach().cpu().numpy(), gt_points.detach().cpu().numpy()):
#             box = get_bbox_from_cam(cam, 
#                                     point, 
#                                     cam_thr=test_cfg['cam_thr'], 
#                                     area_ratio=test_cfg['area_ratio'], 
#                                     img_size=(patch_h * 16, patch_w * 16), 
#                                     box_method=test_cfg['box_method'])
#             pseudo_gt_bboxes.append(torch.as_tensor(box, 
#                                                     dtype=gt_points.dtype, 
#                                                     device=gt_points.device))
#         pseudo_gt_bboxes = torch.cat(pseudo_gt_bboxes)
#         pseudo_gt_bboxes = list(torch.split(pseudo_gt_bboxes, split_lengths, dim=0))
#         return pseudo_gt_labels, pseudo_gt_bboxes
    
def get_bbox_from_cam(cam, point, cam_thr=0.2, area_ratio=0.5, 
                      img_size=None, box_method='expand'):
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
    img_h, img_w = img_size
    cam = (cam * 255.).astype(np.uint8)
    map_thr = cam_thr * np.max(cam)
    _, thr_gray_heatmap = cv2.threshold(cam,
                                        int(map_thr), 255,
                                        cv2.THRESH_TOZERO)
    contours, _ = cv2.findContours(thr_gray_heatmap,
                                       cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 0:
        estimated_bbox = []
        areas = list(map(cv2.contourArea, contours))
        area_idx = sorted(range(len(areas)), key=areas.__getitem__, reverse=True)
        for idx in area_idx:
            if areas[idx] >= areas[area_idx[0]] * area_ratio:
                c = contours[idx]
                x, y, w, h = cv2.boundingRect(c)
                estimated_bbox.append([x, y, x + w, y + h])
    else:
        estimated_bbox = [[0, 0, 1, 1]]
    estimated_bbox = np.array(estimated_bbox)
    
    proposal_xmin = np.min(estimated_bbox[:, 0])
    proposal_ymin = np.min(estimated_bbox[:, 1])
    proposal_xmax = np.max(estimated_bbox[:, 2])
    proposal_ymax = np.max(estimated_bbox[:, 3])
    
    if box_method == 'min_max':
        estimated_bbox = np.array([[proposal_xmin, proposal_ymin, 
                                    proposal_xmax, proposal_ymax]])
        return estimated_bbox
    
    elif box_method == 'expand':
        xc, yc = point
        if np.abs(xc - proposal_xmin) > np.abs(xc - proposal_xmax):
            gt_xmin = proposal_xmin
            gt_xmax = xc * 2 -  gt_xmin
            gt_xmax = gt_xmax if gt_xmax < img_w else float(img_w)
        else:
            gt_xmax = proposal_xmax
            gt_xmin = xc * 2 -  gt_xmax
            gt_xmin = gt_xmin if gt_xmin > 0 else 0.0

        if np.abs(yc - proposal_ymin) > np.abs(yc - proposal_ymax):
            gt_ymin = proposal_ymin
            gt_ymax = yc * 2 -  gt_ymin
            gt_ymax = gt_ymax if gt_ymax < img_h else float(img_h)
        else:
            gt_ymax = proposal_ymax
            gt_ymin = yc * 2 -  gt_ymax
            gt_ymin = gt_ymin if gt_ymin > 0 else 0.0

        estimated_bbox = np.array([[gt_xmin, gt_ymin, gt_xmax, gt_ymax]])
        return estimated_bbox    
    
# def get_bbox_from_cam_fast(cam, point, cam_thr=0.2, area_ratio=0.5, 
#                       img_size=None, box_method='expand', erode=False):
#     img_h, img_w = img_size
#     cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
#     cam[cam >= cam_thr] = 1 # binary the map
#     cam[cam < cam_thr] = 0
    
#     if erode:
#         cam_erode = tensor_erode(cam)
#         cam = cam.bool() ^ cam_erode.bool()
        
#     cam_out = connected_components_labeling(cam.to(torch.uint8))
#     labels = cam_out.unique()
    
#     # 先获得一个无序的contours     
#     contours = []
#     for i, label in enumerate(labels):
#         if i == 0:
#             continue
#         else:
#             contour = torch.nonzero(cam_out == label)
#             contours.append(contour.cpu().numpy())
            
#     # 我们重新画一个numpy数组来做 findcounter，可能会快，没有了cuda2cpu的漫长
#     contour_map = np.zeros((img_h, img_w), dtype=np.uint8)
#     if len(contours) != 0: # 说明有高响应部分
#         for contour in contours:
#             contour_map[contour[:, 0], contour[:, 1]] = 255
        
#         contours, _ = cv2.findContours(contour_map,
#                                         cv2.RETR_TREE,
#                                         cv2.CHAIN_APPROX_SIMPLE)
#         estimated_bbox = []
#         areas = list(map(cv2.contourArea, contours))
#         area_idx = sorted(range(len(areas)), key=areas.__getitem__, reverse=True)
#         for idx in area_idx:
#             if areas[idx] >= areas[area_idx[0]] * area_ratio:
#                 c = contours[idx]
#                 x, y, w, h = cv2.boundingRect(c)
#                 estimated_bbox.append([x, y, x + w, y + h])
    
#         estimated_bbox = np.array(estimated_bbox)
#         proposal_xmin = np.min(estimated_bbox[:, 0])
#         proposal_ymin = np.min(estimated_bbox[:, 1])
#         proposal_xmax = np.max(estimated_bbox[:, 2])
#         proposal_ymax = np.max(estimated_bbox[:, 3])
        
#         if box_method == 'min_max':
#             estimated_bbox = np.array([[proposal_xmin, proposal_ymin, 
#                                         proposal_xmax, proposal_ymax]])
#             return estimated_bbox

#         elif box_method == 'expand':
#             xc, yc = point
#             if np.abs(xc - proposal_xmin) > np.abs(xc - proposal_xmax):
#                 gt_xmin = proposal_xmin
#                 gt_xmax = xc * 2 -  gt_xmin
#                 gt_xmax = gt_xmax if gt_xmax < img_w else float(img_w)
#             else:
#                 gt_xmax = proposal_xmax
#                 gt_xmin = xc * 2 -  gt_xmax
#                 gt_xmin = gt_xmin if gt_xmin > 0 else 0.0

#             if np.abs(yc - proposal_ymin) > np.abs(yc - proposal_ymax):
#                 gt_ymin = proposal_ymin
#                 gt_ymax = yc * 2 -  gt_ymin
#                 gt_ymax = gt_ymax if gt_ymax < img_h else float(img_h)
#             else:
#                 gt_ymax = proposal_ymax
#                 gt_ymin = yc * 2 -  gt_ymax
#                 gt_ymin = gt_ymin if gt_ymin > 0 else 0.0

#             estimated_bbox = np.array([[gt_xmin, gt_ymin, gt_xmax, gt_ymax]])
#             return estimated_bbox    
#     else:
#         estimated_bbox = np.array([[0, 0, 1, 1]])
#         return estimated_bbox
    
    
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