from ..builder import DETECTORS
from .two_stage import TwoStageDetector
from ..builder import build_backbone, build_head, build_neck
import torch.nn as nn
import torch
from mmdet.core import bbox2result, bbox2roi, bbox_xyxy_to_cxcywh, bboxbbox2result, bboxpoint2results
from mmdet.core.bbox.iou_calculators import bbox_overlaps
import numpy as np

@DETECTORS.register_module()
class DPM(TwoStageDetector):
    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 lss_head,
                 keypoint_head,
                 train_cfg,
                 test_cfg,
                 roi_skip_fpn=False,
                 test_wo_detector=False,
                 test_on_fcos=False,
                 neck=None,
                 init_cfg=None,
                 *args, **kwargs):
        super(DPM, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            *args, **kwargs)
        
        if lss_head is not None:
            lss_train_cfg = train_cfg.lss if train_cfg is not None else None
            lss_head.update(train_cfg=lss_train_cfg)
            lss_head.update(test_cfg=test_cfg.lss)
            self.lss_head = build_head(lss_head)
        if keypoint_head is not None:
#             keypoint_head.update(train_cfg=train_cfg.keypoint_head)
            self.keypoint_head = build_head(keypoint_head)
            
        self.roi_skip_fpn = roi_skip_fpn
        self.test_wo_detector = test_wo_detector
        self.test_on_fcos = test_on_fcos
        
    @property
    def with_lss_head(self):
        return hasattr(self, 'lss_head') and self.lss_head is not None
    @property
    def with_decoder_head(self):
        return hasattr(self, 'decoder_head') and self.decoder_head is not None
            
    def get_roi_feat(self, x, vit_feat):
        B, _, H, W = x[2].shape
        x = [
            vit_feat.transpose(1, 2).reshape(B, -1, H, W).contiguous()
        ]
        return x
    
    def extract_feat(self, img, gt_points=None, gt_labels=None):
        # 说明只有一个用于imted的fpn
        x = self.backbone(img)
        x = list(x)
        x[0] = self.neck(x[0])
        return x
            
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_points=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        
        batch_input_shape = tuple(img[0].size()[-2:])
        for i, _ in enumerate(img_metas):
            img_metas[i]['batch_input_shape'] = batch_input_shape
        # aug strategy causes that there may be no gts.
        # 
        gt_masks_ = []
        for gt_mask in gt_masks:
            gt_mask = gt_mask.masks
            n, ori_h, ori_w = gt_mask.shape
            padding_gt_mask = np.zeros((n, *batch_input_shape), dtype=np.bool)
            padding_gt_mask[:, :ori_h, :ori_w] = gt_mask
            padding_gt_mask = torch.as_tensor(padding_gt_mask, device=gt_bboxes[0].device).bool()
            gt_masks_.append(padding_gt_mask)
        gt_masks = gt_masks_
        # 
        empty = False
        for g in gt_bboxes: # 只要有一个batch中有一个img 无gt的，就直接都不要了
            if len(g) == 0:
                empty = True
        if empty:
            gt_bboxes = [torch.as_tensor([[20, 20, 40, 40]], 
                                         dtype=gt_bboxes[0].dtype,
                                         device=gt_bboxes[0].device) 
                         for _ in range(len(img_metas))]
            gt_labels = [torch.as_tensor([0], 
                                         dtype=gt_labels[0].dtype,
                                         device=gt_labels[0].device) 
                         for _ in range(len(img_metas))]
            gt_points = [torch.as_tensor([[30, 30]], 
                                         dtype=gt_bboxes[0].dtype,
                                         device=gt_bboxes[0].device) 
                         for _ in range(len(img_metas))]
            gt_masks = [torch.zeros((1, *batch_input_shape), device=gt_bboxes[0].device).bool()
                         for _ in range(len(img_metas))]
            
        pseudo_gt_bboxes, pseudo_gt_labels = None, None
        x = self.extract_feat(img)
        if len(x) == 6:
            x, vit_feat, point_tokens, attns, scale_features, vit_feat_be_norm = x
            # center points as gt_points
            gt_points = [torch.cat([
                bboxes[:, 0::2].mean(-1).unsqueeze(-1), 
                bboxes[:, 1::2].mean(-1).unsqueeze(-1)
            ], dim=-1) for bboxes in gt_bboxes]
            # point settings
            imgs_whwh = []
            for meta in img_metas:
                h, w, _ = meta['img_shape']
                imgs_whwh.append(x[0].new_tensor([[w, h]]))
            imgs_whwh = torch.cat(imgs_whwh, dim=0)
            imgs_whwh = imgs_whwh[:, None, :]
            
            losses = dict()            
            # point training / pseudo gt generation
            points_results = self.lss_head.forward_train_point(x,
                                                            vit_feat,
                                                            point_tokens,
                                                            attns,
                                                            img_metas,
                                                            gt_bboxes,
                                                            gt_labels,
                                                            gt_points,
                                                            imgs_whwh=imgs_whwh,
                                                            gt_masks=gt_masks)
            losses.update(points_results['loss_point'])
            # 第一阶段wsddn
            mil_results = self.lss_head.forward_train_mil(x,
                                                        vit_feat,
                                                        point_tokens,
                                                        None,
                                                        None,
                                                        points_results['multiple_cams'],
                                                        points_results['refined_multiple_masks'],
                                                        points_results['pseudo_proposals'],
                                                        img_metas,
                                                        points_results['gt_bboxes'],
                                                        points_results['gt_labels'],
                                                        points_results['gt_points'],
                                                        imgs_whwh=imgs_whwh) 
            losses.update(mil_results['loss_mil'])
            iou_metric = mil_results['iou_metric']
            iou_metric = dict(iou_metric=iou_metric)
            losses.update(iou_metric)
            
            pseudo_gt_bboxes = mil_results['pseudo_gt_bboxes']
            pseudo_gt_labels = mil_results['pseudo_gt_labels']
            
            attnshift_results, losses_hinge = self.keypoint_head.forward_train(x,
                                                               vit_feat,
                                                               mil_results['matched_cams'],
                                                               img_metas,
                                                               pseudo_gt_bboxes,
                                                               pseudo_gt_labels,
                                                               points_results['gt_points'],
                                                               mil_results['semantic_scores'],
                                                               vit_feat_be_norm,
                                                               imgs_whwh=imgs_whwh,
                                                               gt_masks=points_results['gt_masks'],
                                                            )
            losses.update(losses_hinge)
            
            # losses_point = self.lss_head._calculate_loss(img_metas,
            #                                              points_results['gt_points'], 
            #                                              points_results['gt_labels'], 
            #                                              # dedetr_results['fg_points'], 
            #                                              attnshift_results['all_semantic_points'],
            #                                              attnshift_results['all_visible_weights'],
            #                                              points_results,
            #                                              imgs_whwh=imgs_whwh)
            # losses.update(losses_point)
            
            # rpn setting 
            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes if pseudo_gt_bboxes is None else pseudo_gt_bboxes,
                # points_results['pseudo_proposals'],
                # gt_labels=points_results['gt_labels'],
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            
            losses.update(rpn_losses)
            # rcnn setting
            if self.roi_skip_fpn: # imted
                roi_losses = self.roi_head.forward_train(self.get_roi_feat(x, vit_feat), img_metas, proposal_list,
                                                         gt_bboxes if pseudo_gt_bboxes is None else pseudo_gt_bboxes, 
                                                         gt_labels if pseudo_gt_labels is None else pseudo_gt_labels,
#                                                          gt_bboxes if pseudo_gt_bboxes is None else dedetr_results['gt_bboxes'], 
#                                                          gt_labels if pseudo_gt_labels is None else dedetr_results['gt_labels'],
                                                         attnshift_results['pseudo_points'], 
                                                         attnshift_results['pseudo_bin_labels'],
#                                                          attnshift_results['all_semantic_points'],
#                                                          attnshift_results['all_visible_weights'],
                                                         # attnshift_results['all_dpm_points'],
                                                         # attnshift_results['all_dpm_visible'],
                                                         attnshift_results['all_mask_sup_points'],
                                                         attnshift_results['all_mask_sup_visibles'],
                                                         gt_bboxes_ignore, 
#                                                          gt_masks=attnshift_results['gt_masks'],
                                                         img=img, **kwargs)        
        
            else: # faster rcnn
                assert False, 'no implement'
#                 roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
#                                                          pseudo_gt_bboxes, pseudo_gt_labels,
#                                                          gt_bboxes_ignore, gt_masks,
#                                                          **kwargs)
            losses.update(roi_losses)
            # debug
            # if losses['dpm_bce_loss'] == 0 or losses['dpm_aux_bce_loss'] == 0:
            #     print(losses['dpm_bce_loss'], losses['dpm_aux_bce_loss'])
                
            if empty:
                for k, v in losses.items():
                    if k == 'loss_rpn_cls' or k == 'loss_rpn_bbox':
                        losses[k] = [l * 0 for l in losses[k]]
                    else:
                        losses[k] = v * 0
            return losses
        else:
            assert False, 'no implemention'
            
    def simple_test(self, img, img_metas, 
                    gt_bboxes=None, 
                    gt_labels=None,
                    gt_points=None, 
                    gt_masks=None,
                    proposals=None, 
                    rescale=False):
        
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)
#         if len(x) == 5:
        if len(x) == 6:
            x, vit_feat, point_tokens, attns, scale_features, vit_feat_be_norm = x
            # if False:
            #     pass
            if self.test_wo_detector:
                # 去掉多尺度测试
                gt_bboxes = gt_bboxes[0]
                gt_labels = gt_labels[0]

                # print(len(img_metas), img_metas, len(img), len(img[0]))
                batch_input_shape = tuple(img[0].size()[-2:])
                for i, _ in enumerate(img_metas):
                    img_metas[i]['batch_input_shape'] = batch_input_shape
            
                gt_masks_ = []
                for gt_mask in gt_masks[0]:
                    gt_mask = gt_mask.masks
                    n, ori_h, ori_w = gt_mask.shape
                    padding_gt_mask = np.zeros((n, *batch_input_shape), dtype=np.bool)
                    padding_gt_mask[:, :ori_h, :ori_w] = gt_mask
                    padding_gt_mask = torch.as_tensor(padding_gt_mask, device=gt_bboxes[0].device).bool()
                    gt_masks_.append(padding_gt_mask)
                gt_masks = gt_masks_
                
                        
                # center points as gt_points
                gt_points = [torch.cat([
                    bboxes[:, 0::2].mean(-1).unsqueeze(-1), 
                    bboxes[:, 1::2].mean(-1).unsqueeze(-1)
                ], dim=-1) for bboxes in gt_bboxes]
                
                # point settings
                imgs_whwh = []
                for meta in img_metas:
                    h, w, _ = meta['img_shape']
                    imgs_whwh.append(x[0].new_tensor([[w, h]]))
                imgs_whwh = torch.cat(imgs_whwh, dim=0)
                imgs_whwh = imgs_whwh[:, None, :]
                
                # point training / pseudo gt generation
                points_results = self.lss_head.forward_train_point(x,
                                                                vit_feat,
                                                                point_tokens,
                                                                attns,
                                                                img_metas,
                                                                gt_bboxes,
                                                                gt_labels,
                                                                gt_points,
                                                                imgs_whwh=imgs_whwh,
                                                                gt_masks=gt_masks)
                # 第一阶段wsddn
                mil_results = self.lss_head.forward_train_mil(x,
                                                            vit_feat,
                                                            point_tokens,
                                                            None,
                                                            None,
                                                            points_results['multiple_cams'],
                                                            points_results['refined_multiple_masks'],
                                                            points_results['pseudo_proposals'],
                                                            img_metas,
                                                            points_results['gt_bboxes'],
                                                            points_results['gt_labels'],
                                                            points_results['gt_points'],
                                                            imgs_whwh=imgs_whwh) 
                pseudo_gt_bboxes = mil_results['pseudo_gt_bboxes']
                pseudo_gt_labels = mil_results['pseudo_gt_labels']
                
                attnshift_results, losses_hinge = self.keypoint_head.forward_train(x,
                                                                   vit_feat,
                                                                   mil_results['matched_cams'],
                                                                   img_metas,
                                                                   pseudo_gt_bboxes,
                                                                   pseudo_gt_labels,
                                                                   points_results['gt_points'],
                                                                   mil_results['semantic_scores'],
                                                                   vit_feat_be_norm,
                                                                   imgs_whwh=imgs_whwh,
                                                                   gt_masks=points_results['gt_masks'],
                                                                )
                dpm_points = attnshift_results['all_dpm_points']
                gt_points_vis = attnshift_results['all_dpm_visible']
                
                fg_masks = attnshift_results['fg_masks']

                seg_results = []
                # for i_img in range(len(fg_masks)):
                for i_img in range(len(points_results['gt_masks'])):
                    cls_segms = [[] for _ in 
                                 range(self.lss_head.point_head.num_classes)]  # BG is not included in num_classes
                    for attn_mask, label in zip(points_results['gt_masks'][i_img], pseudo_gt_labels[i_img]):
                        cls_segms[label].append(attn_mask.detach().cpu().numpy())
                    seg_results.append(cls_segms)

            
                return seg_results
                # bbox_results = [
                #     bboxpoint2results(pseudo_gt_bboxes[i], 
                #                     pseudo_gt_labels[i], 
                #                     gt_bboxes[i],
                #                     dpm_points[i],
                #                     gt_points_vis[i],
                #                     self.lss_head.point_head.num_classes)
                #     for i in range(len(pseudo_gt_bboxes))
                # ]
                # return bbox_results
            else:
                proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
                if self.roi_skip_fpn: # imted
                    return self.roi_head.simple_test(
                        self.get_roi_feat(x, vit_feat), proposal_list, img_metas, rescale=rescale)
                else:
                    return self.roi_head.simple_test(
                        x, proposal_list, img_metas, rescale=rescale)

    def scale_selection_(self, points_results, mil_results, matched_results, img_size):
        vis_match_cls_probs = matched_results['vis_match_cls_probs'] #.reshape(num_gt, self.lss_head.scale, -1)
        vis_match_loc_probs = matched_results['vis_match_loc_probs'] #.reshape(num_gt, self.lss_head.scale, -1)
        pseudo_proposals = points_results['pseudo_proposals']
        pseudo_gt_labels = points_results['gt_labels']
        gt_bboxes = points_results['gt_bboxes']

        pseudo_gt_bboxes = []
        for cls_probs, loc_probs, proposals in zip(vis_match_cls_probs, vis_match_loc_probs, pseudo_proposals):
            num_gt = proposals.size(0)
            # free anchor
            matched_prob = cls_probs * loc_probs
            weight = 1 / torch.clamp(1 - matched_prob, 1e-12, None)
            weight /= weight.sum(dim=1).unsqueeze(dim=-1)
            bag_prob = (weight * matched_prob)

#             matched_prob = (cls_probs * loc_probs).reshape(num_gt, self.lss_head.scale, -1)
#             weight1 = 1 / torch.clamp(1 - matched_prob, 1e-12, None)
#             weight1 /= weight1.sum(dim=-1).unsqueeze(dim=-1)
#             weight2 = 1 / torch.clamp(1 - matched_prob, 1e-12, None)
#             weight2 /= weight2.sum(dim=1).unsqueeze(dim=1)
#             weight = (weight1 * weight2).flatten(1)
#             bag_prob = (weight * matched_prob.flatten(1))
            
            bag_prob_per_gt = bag_prob.reshape(num_gt, self.lss_head.scale, -1).sum(-1)
            max_gt_ind = torch.argmax(bag_prob_per_gt, -1)
            select_pseudo_gt_bboxes = proposals[torch.arange(num_gt).to(max_gt_ind.device), max_gt_ind]
            pseudo_gt_bboxes.append(select_pseudo_gt_bboxes)

        iou_metric = bbox_overlaps(torch.cat(pseudo_gt_bboxes),
                               torch.cat(gt_bboxes),
                               is_aligned=True).mean()
        return pseudo_gt_bboxes, pseudo_gt_labels, iou_metric, gt_bboxes
        
    
    def scale_selection(self, points_results, mil_results, matched_results, img_size):
        strides = self.scale_head.strides
        w, h = img_size
        num_img = len(points_results['pseudo_proposals'])
        flatten_cls_scores = matched_results['flatten_cls_scores']
        flatten_bbox_preds = matched_results['flatten_bbox_preds']
        flatten_labels = matched_results['flatten_labels']
        flatten_inds = matched_results['flatten_inds']
        flatten_centerness_targets = matched_results['flatten_centerness_targets']
        pseudo_proposals = points_results['pseudo_proposals'] #[0].reshape(-1, 4)


        gt_bboxes = points_results['gt_bboxes']
        num_gt = [proposals.size(0) for proposals in pseudo_proposals]
        num_scale = pseudo_proposals[0].size(1)

        all_level_points = matched_results['all_level_points']
        a = [points.size(0) for points in all_level_points]
        split_lengths = [points.size(0) * num_img for points in all_level_points]
        lvl_cls_scores = torch.split(flatten_cls_scores, split_lengths)
        lvl_bbox_preds = torch.split(flatten_bbox_preds, split_lengths)
        lvl_gt_inds = torch.split(flatten_inds, split_lengths)
        lvl_gt_labels = torch.split(flatten_labels, split_lengths)
        lvl_centerness_targets = torch.split(flatten_centerness_targets, split_lengths)
        bg_class_ind = lvl_cls_scores[0].size(-1)

        pseudo_gt_bboxes = []
        pseudo_gt_labels = []
        pseudo_gt_scores = []
        remain_gt_bboxes = []
        iou_metric = torch.as_tensor(0.).to(all_level_points[0].device)
        for i_batch in range(num_img):
            pseudo_gt_bboxes_ = []
            pseudo_gt_labels_ = []
            pseudo_gt_scores_ = []
            for ind in range(num_gt[i_batch]):
                lvl_scores_per_gt = []
                lvl_cls_scores_per_gt = []
                lvl_pred_bboxes_per_gt = []
                lvl_label_per_gt = []
                for i_lvl, num_points_per_lvl in enumerate(split_lengths):
                    lvl_split_into_batch = [num_points_per_lvl // num_img] * num_img
                    batch_lvl_cls_scores = torch.split(lvl_cls_scores[i_lvl], lvl_split_into_batch)[i_batch]
                    batch_lvl_centerness_targets = torch.split(lvl_centerness_targets[i_lvl], lvl_split_into_batch)[i_batch]
                    batch_lvl_bbox_preds = torch.split(lvl_bbox_preds[i_lvl], lvl_split_into_batch)[i_batch]
                    batch_lvl_labels = torch.split(lvl_gt_labels[i_lvl], lvl_split_into_batch)[i_batch]
                    batch_lvl_gt_inds = torch.split(lvl_gt_inds[i_lvl], lvl_split_into_batch)[i_batch]
                    batch_lvl_points = all_level_points[i_lvl]
                    if self.training:
                        batch_lvl_bbox_preds = batch_lvl_bbox_preds * strides[i_lvl]
                    batch_lvl_pred_bboxes = self.scale_head.bbox_coder.decode(batch_lvl_points, batch_lvl_bbox_preds)
                    
                    pos_inds = ((batch_lvl_labels >= 0)
                                & (batch_lvl_labels < bg_class_ind)).nonzero().reshape(-1)

                    pos_labels = batch_lvl_labels[pos_inds]
                    pos_centerness_targets = batch_lvl_centerness_targets[pos_inds].sigmoid() #因为centerness是bce获得,因此用sigmoid获得分数
                    pos_cls_scores = batch_lvl_cls_scores[pos_inds].sigmoid() #因为分数是focal获得,因此用sigmoid获得分数
                    # 利用图像中存在的label,只选择存在的类别的分数
                    pos_cls_scores = torch.gather(pos_cls_scores, dim=-1, index=pos_labels.reshape(-1, 1)).reshape(-1)
                    pos_pred_bboxes_ = batch_lvl_pred_bboxes[pos_inds] #因为pred bbox会超过图像,因此我们用图像约束其范围
                    pos_pred_bboxes = torch.cat([
                        torch.clamp(pos_pred_bboxes_[:, 0], min=0, max=w).unsqueeze(-1),
                        torch.clamp(pos_pred_bboxes_[:, 1], min=0, max=h).unsqueeze(-1),
                        torch.clamp(pos_pred_bboxes_[:, 2], min=0, max=w).unsqueeze(-1),
                        torch.clamp(pos_pred_bboxes_[:, 3], min=0, max=h).unsqueeze(-1),
                    ], dim=-1)
                    pos_gt_inds = batch_lvl_gt_inds[pos_inds]

                    # 一定需要分数过滤才能选好层 
                    threshold = self.scale_head.score_threshold
                    if self.scale_head.centerness_weight_on_threshold:
                        keep_flag = ((pos_cls_scores * pos_centerness_targets) >= threshold)
                    else:
                        keep_flag = (pos_cls_scores >= threshold)
                        
                    pos_labels = pos_labels[keep_flag]
                    pos_cls_scores = pos_cls_scores[keep_flag]
                    pos_pred_bboxes = pos_pred_bboxes[keep_flag]
                    pos_gt_inds = pos_gt_inds[keep_flag]
                    pos_centerness_targets = pos_centerness_targets[keep_flag]
                    # 一定需要分数过滤才能选好层 
                    matched_gt_flag = (pos_gt_inds == ind)
                    if matched_gt_flag.sum() == 0:
                        lvl_scores = pos_cls_scores.new_tensor(0.)
                        matched_scores = pos_cls_scores.new_tensor(0.)
                        matched_bboxes = pos_pred_bboxes.new_tensor([0., 0., 1., 1.])
                        matched_labels = pos_labels.new_tensor(bg_class_ind)
                        lvl_scores_per_gt.append(lvl_scores)
                        lvl_cls_scores_per_gt.append(matched_scores)
                        lvl_pred_bboxes_per_gt.append(matched_bboxes)
                        lvl_label_per_gt.append(matched_labels)
                    else:
                        if self.scale_head.centerness_weight_on_lvl_scores:
                            lvl_scores = (pos_cls_scores[matched_gt_flag] * pos_centerness_targets[matched_gt_flag]).mean()
                        else:
                            lvl_scores = pos_cls_scores[matched_gt_flag].mean()
                            
                        if self.scale_head.centerness_weight_on_selection:
                            matched_scores, max_ind = (pos_cls_scores[matched_gt_flag] * pos_centerness_targets[matched_gt_flag]).max(0)
                        else:
                            matched_scores, max_ind = pos_cls_scores[matched_gt_flag].max(0)
                        
                        matched_bboxes = pos_pred_bboxes[matched_gt_flag][max_ind]
                        matched_labels = pos_labels[matched_gt_flag][max_ind]
                        lvl_scores_per_gt.append(lvl_scores)
                        lvl_cls_scores_per_gt.append(matched_scores)
                        lvl_pred_bboxes_per_gt.append(matched_bboxes)
                        lvl_label_per_gt.append(matched_labels)

                lvl_scores_per_gt = torch.stack(lvl_scores_per_gt)
                lvl_pred_bboxes_per_gt = torch.stack(lvl_pred_bboxes_per_gt)
                lvl_label_per_gt = torch.stack(lvl_label_per_gt)
                lvl_cls_scores_per_gt = torch.stack(lvl_cls_scores_per_gt)
                
                matched_lvl_id = torch.argmax(lvl_scores_per_gt)
                pseudo_gt_bboxes_.append(lvl_pred_bboxes_per_gt[matched_lvl_id])
                pseudo_gt_labels_.append(lvl_label_per_gt[matched_lvl_id])
                pseudo_gt_scores_.append(lvl_cls_scores_per_gt[matched_lvl_id])
            pseudo_gt_bboxes_ = torch.stack(pseudo_gt_bboxes_)
            pseudo_gt_labels_ = torch.stack(pseudo_gt_labels_)
            pseudo_gt_scores_ = torch.stack(pseudo_gt_scores_)
            # 可能存在没有任何点 的分数 大于 score threshold 因此,这时候千万不要有任何框,直接去除
            keep = (pseudo_gt_scores_ >= threshold)
            pseudo_gt_bboxes_ = pseudo_gt_bboxes_[keep]
            pseudo_gt_labels_ = pseudo_gt_labels_[keep]
            pseudo_gt_scores_ = pseudo_gt_scores_[keep]
            gts = gt_bboxes[i_batch][keep]
            if len(pseudo_gt_bboxes_) == 0:
                iou_metric += 0
            else:
                iou_metric += bbox_overlaps(pseudo_gt_bboxes_,
                                       gts,
                                       is_aligned=True).mean()
            pseudo_gt_bboxes.append(pseudo_gt_bboxes_)
            pseudo_gt_labels.append(pseudo_gt_labels_)
            pseudo_gt_scores.append(pseudo_gt_scores_)
            remain_gt_bboxes.append(gts)
        iou_metric /= num_img
        return pseudo_gt_bboxes, pseudo_gt_labels, iou_metric, remain_gt_bboxes