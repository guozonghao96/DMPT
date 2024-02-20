# Copyright (c) OpenMMLab. All rights reserved.
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob, constant_init
from mmcv.runner import force_fp32

from mmdet.core import multi_apply
from mmdet.models.utils.transformer import inverse_sigmoid
from ..builder import HEADS
from .point_detr_head import PointDETRHead
from mmdet.models.losses import accuracy
from ..builder import build_roi_extractor
from mmdet.core import (bbox2result, bbox2roi, build_assigner, build_sampler, multi_apply,
                        reduce_mean, bbox_xyxy_to_cxcywh, bbox_cxcywh_to_xyxy)
from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding

from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None
from mmdet.models.utils import PMMs    

def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.

    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)

class KeyPointL1Cost(object):
    def __init__(self, weight=1.):
        self.weight = weight
    
@HEADS.register_module()
class PointDeformableDETRHead(PointDETRHead):

    def __init__(self,
                 # instance attention map的参数
                 iam_num_points_init=10,
                 iam_thr_pos=0.35, 
                 iam_thr_neg=0.8,
                 iam_refine_times=2, 
                 iam_obj_tau=0.9,
                 pca_dim=64,
                 meanshift_refine_times=5,
                 num_points_for_meanshift=20,
                 # instance attention map的参数
                 discriminate_loss_weight=1.0,
                 num_query=300,
                 point_feat_extractor=None,
                 with_gt_points=False,
                 num_classes=20,
                 in_channels=256,
                 with_box_refine=False,
                 sync_cls_avg_factor=True,
                 as_two_stage=False,
                 num_keypoints=6,
                 transform_method='minmax',
                 transformer=None,
                 transformer_refine=None,
                 train_cfg=dict(
                    assigner=dict(
                        type='HungarianPointAssigner',
                        cls_cost=dict(type='FocalLossCost', weight=1.0),
                        reg_cost=dict(type='PointL1Cost', weight=10.0)),
                    sampler=dict(type='PointPseudoSampler'),
                 ),
                 **kwargs):
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        self.with_gt_points = with_gt_points
        self.num_keypoints = num_keypoints
        self.transform_method = transform_method
        
        # instance attention map的参数
        self.num_query = num_query
        self.iam_thr_pos = iam_thr_pos
        self.iam_thr_neg = iam_thr_neg
        self.iam_refine_times = iam_refine_times
        self.iam_obj_tau = iam_obj_tau
        self.iam_num_points_init = iam_num_points_init
        self.point_feat_extractor = point_feat_extractor
        self.pca_dim = pca_dim
        self.meanshift_refine_times = meanshift_refine_times
        self.num_points_for_meanshift = num_points_for_meanshift
        self.discriminate_loss_weight = discriminate_loss_weight
        
        
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage

        super(PointDeformableDETRHead, self).__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            transformer=transformer,
            transformer_refine=transformer_refine,
            **kwargs)
        
        self.assigner = build_assigner(train_cfg.assigner)
        self.sampler = build_sampler(train_cfg.sampler, context=self)
        # 
        self.cluster_generator = PMMs(384, k=self.num_keypoints - 1, stage_num=10, kappa=5)
        
    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""

        fc_cls = Linear(self.embed_dims, self.cls_out_channels)
        reg_branch = []
        
        reg_branch.append(Linear(self.embed_dims, self.embed_dims))
        reg_branch.append(nn.ReLU())
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, 2))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers
        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
            # 
        else:
            assert False, 'no implementation'

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m.bias, bias_init)
        for m in self.reg_branches:
            constant_init(m[-1], 0, bias=0)
            
    def part_keypoint_assign(self, point_pred, gt_points, 
                             gt_visible_weights, num_gts=None, imgs_wh=None):
        
        point_pred = torch.split(point_pred, num_gts, dim=0)
        
        all_matched_row_inds = []
        all_matched_col_inds = []
        
        for preds, target_points, visible_weights, wh in \
            zip(point_pred, gt_points, gt_visible_weights, imgs_wh):
            num_gt = preds.size(0)
            target_points = target_points / wh
            preds_ = preds.detach().cpu()
            target_points_ = target_points.detach().cpu()
            visible_weights_ = visible_weights.detach().cpu()
            
            for i in range(num_gt):
                visible = visible_weights_[i].bool()
                # cost = torch.cdist(preds_[i], target_points_[i][visible], p=1)
                cost = torch.cdist(preds_[i][1:], target_points_[i][1:][visible[1:]], p=1) # 只取不是gt的点来匹配
                matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
                
                # num_visible = cost.size(-1)
                num_visible = cost.size(-1) + 1 # 因为cost计算减掉了gt point个数，所以实际上个数需要再加1
                
                num_invisible = self.num_keypoints - num_visible
                
                # matched_row_inds = np.concatenate([matched_row_inds, 
                #                                    -np.ones(num_invisible).astype(matched_row_inds.dtype)])
                # matched_col_inds = np.concatenate([matched_col_inds, 
                #                                    -np.ones(num_invisible).astype(matched_col_inds.dtype)])
                
                # 把gt point和pred point on ref point对齐
                # （gt point是0index，ref point 是 0 index）0<->0，其他点的匹配的inds需要+1
                matched_row_inds = np.concatenate([np.zeros(1).astype(matched_row_inds.dtype),
                                                   matched_row_inds + 1, 
                                                   -np.ones(num_invisible).astype(matched_row_inds.dtype)])
                matched_col_inds = np.concatenate([np.zeros(1).astype(matched_col_inds.dtype),
                                                   matched_col_inds + 1, 
                                                   -np.ones(num_invisible).astype(matched_col_inds.dtype)])
                
                all_matched_row_inds.append(matched_row_inds)
                all_matched_col_inds.append(matched_col_inds)
                
        all_matched_row_inds = np.stack(all_matched_row_inds, axis=0).reshape(sum(num_gts), self.num_keypoints)
        all_matched_col_inds = np.stack(all_matched_col_inds, axis=0).reshape(sum(num_gts), self.num_keypoints)

        all_matched_row_inds = torch.from_numpy(all_matched_row_inds).to(
             point_pred[0].device)
        all_matched_col_inds = torch.from_numpy(all_matched_col_inds).to(
            point_pred[0].device)
        return all_matched_row_inds, all_matched_col_inds
   
    def discriminate_loss(self, part_feats):
        norm_part_feats = torch.norm(part_feats, p=2, dim=-1)
        norm_matrix = torch.bmm(norm_part_feats.unsqueeze(-1), norm_part_feats.unsqueeze(-2))
        dot_matrix = torch.bmm(part_feats, part_feats.transpose(-2, -1))
        eye = ~torch.eye(self.num_keypoints).bool().to(part_feats.device)
        dis_loss = ((eye * (dot_matrix / norm_matrix)) ** 2).sum()
        return self.discriminate_loss_weight * dis_loss
        
        # self.discriminate_loss_weight

    def forward_train(self,
                      x,  # fpn feature (5, ) (b, c, h, w)
                      vit_feat, # 直接用最后一层特征就行
                      matched_cams,
                      img_metas,
                      gt_bboxes, # mil_pseudo_bboxes
                      gt_labels, # gt_labels
                      gt_points, # gt_points
                      dy_weights,
                      vit_feat_be_norm,
                      imgs_whwh=None,
                     ):
        
        # 生成监督label
        featmap_sizes = [featmap.size()[-2:] for featmap in x]
        num_imgs = len(img_metas)
        vit_feat = vit_feat.permute(0, 2, 1).reshape(num_imgs, -1, *featmap_sizes[2]).contiguous()
        vit_feat_be_norm = vit_feat_be_norm.permute(0, 2, 1).reshape(num_imgs, -1, *featmap_sizes[2]).contiguous()
        instance_cos_fg_maps = []
        instance_cos_bg_maps = []
        fg_points = []
        fg_point_feats = []
        bg_point_feats = []
        
        pseudo_points = []
        pseudo_bin_labels = []
        
        all_visible_weights = []
        all_semantic_points = []
        all_part_attn_maps = []
        
        fg_masks = []
        for i_img in range(num_imgs):
            # 第一个mean shift操作，后的instance attention map
            # refine次数，num_gt, img_h, img_w
            # self-attention map的前景点和背景点 -> num_gt, num_point, 2
            neg_points, neg_point_feats, map_cos_fg, map_cos_bg, \
                points_bg_attn, points_fg_attn = self.instance_attention_generation(matched_cams[i_img], 
                                                                               gt_bboxes[i_img],
                                                                               vit_feat[i_img],
                                                                               vit_feat_be_norm[i_img],
                                                                               pos_thr=self.iam_thr_pos, 
                                                                               neg_thr=self.iam_thr_neg, 
                                                                               num_gt=20, # 就是取20个特征做mean shift
                                                                               refine_times=self.iam_refine_times, 
                                                                               obj_tau=self.iam_obj_tau,
                                                                               gt_points=gt_points[i_img])
            # 在最后一次shift的instance attention上均匀采样得到的前景点，num_gt, num_point, 2 
            # (并且是在feat map size下，即小了16倍)
            # 并且vit feat上直接取（可以用roi algin/grid sample进行插值财经）num_gt, num_point, 256
            sampled_points, pos_point_feats, \
                visible_weights, semantic_points, fg_mask = self.get_semantic_centers(map_cos_fg[-1].clone(), 
                                                                    map_cos_bg[-1].clone(), 
                                                                    gt_bboxes[i_img], 
                                                                    vit_feat[i_img], 
                                                                    pos_thr=0.35,
                                                                    n_points_sampled=self.iam_num_points_init,
                                                                    gt_points=gt_points[i_img] if self.with_gt_points else None,
                                                                    gt_labels=gt_labels[i_img])
            pseudo_points_ = torch.cat([sampled_points, neg_points], dim=1).float()
            pseudo_bin_labels_ = torch.cat([torch.ones_like(sampled_points)[..., 0],
                                        torch.zeros_like(neg_points)[..., 0]], dim=1).bool()
            
            pseudo_points.append(pseudo_points_)
            pseudo_bin_labels.append(pseudo_bin_labels_)
            instance_cos_fg_maps.append(map_cos_fg)
            instance_cos_bg_maps.append(map_cos_bg)
            fg_points.append(sampled_points) # 原图大小的点
            fg_point_feats.append(pos_point_feats)
            bg_point_feats.append(neg_point_feats)
            
            all_visible_weights.append(visible_weights)
            all_semantic_points.append(semantic_points)
            fg_masks.append(fg_mask)
            
#             # 用gmm获得prototype和point
#             with torch.no_grad():
#                 part_attn_maps, semantic_points, \
#                     visible_weights = self.cluster_generator(vit_feat_be_norm[i_img], 
#                                                              fg_mask, gt_bboxes[i_img])
                
#             semantic_points = torch.cat([gt_points[i_img].unsqueeze(1), semantic_points], dim=1)
#             _gt_visible_weights = torch.ones(visible_weights.size(0), 1).long().to(visible_weights.device)
#             visible_weights = torch.cat([_gt_visible_weights, visible_weights], dim=1)
            
#             all_visible_weights.append(visible_weights)
#             all_semantic_points.append(semantic_points)
#             all_part_attn_maps.append(part_attn_maps)
            
            
        num_gts = [ponits.size(0) for ponits in fg_points]
        # 进行point dedetr
        mlvl_feats = [vit_feat]
        batch_size = mlvl_feats[0].size(0)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        img_masks = mlvl_feats[0].new_ones(
            (batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            img_masks[img_id, :img_h, :img_w] = 0
            
        mlvl_masks = []
        mlvl_positional_encodings = []
        for feat in mlvl_feats:
            mlvl_masks.append(
                F.interpolate(img_masks[None],
                              size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            mlvl_positional_encodings.append(
                self.positional_encoding(mlvl_masks[-1]))
            
        ref_points = torch.cat(fg_points).detach()
        # ref_points = torch.cat([gt_points[i].unsqueeze(1).repeat(1, self.num_keypoints, 1)
        #                         for i in range(len(num_gts))], dim=0)
        
        hs, init_reference, inter_references, \
            enc_outputs_class, enc_outputs_coord, \
                intermediate_shifted_points_refine, \
                intermediate_attn_weights = self.transformer(
                    ref_points,
                    mlvl_feats,
                    mlvl_masks,
                    None, # query_embeds
                    mlvl_positional_encodings,
                    reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                    cls_branches=self.cls_branches if self.as_two_stage else None,  # noqa:E501
                    imgs_wh=imgs_whwh,
                    num_gts=num_gts
            )
        hs = hs.permute(0, 2, 1, 3) # num_stage, num_gt, num_keypoints, 256
        outputs_classes = []
        outputs_coords = []
            
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        
        # outputs_classes = torch.split(outputs_classes, num_gts, dim=1)
        # outputs_coords = torch.split(outputs_coords, num_gts, dim=1)
        
        # 输入loss进行loss计算
        gt_point_labels = []
        gt_fg_points = []
        gt_visible_weights = []
        for fg_points_, labels_, weights_ in zip(all_semantic_points, gt_labels, all_visible_weights):
            num_gt, num_points = fg_points_.size()[:2]
            assert num_points == self.num_keypoints
            fg_points_ = fg_points_.reshape(num_gt, self.num_keypoints, 2)
            labels_ = labels_.unsqueeze(-1).repeat(1, self.num_keypoints)
            gt_fg_points.append(fg_points_)
            gt_point_labels.append(labels_)
            gt_visible_weights.append(weights_)
            
        # 计算loss
        # gt_point_labels   -->   [num_gt, num_keypoints]
        # gt_fg_points      -->   [num_gt, num_keypoints, 2]
        # gt_visible_weights   -->   [num_gt, num_keypoints]
        # outputs_classes   -->   [num_stage, num_gt, num_keypoints, num_classes]
        # outputs_coords    -->   [num_stage, num_gt, num_keypoints, 2]
        
        # 每个stage for 一个循环
        losses = dict()
        for i, (cls_scores, pred_points, part_feats) in \
                enumerate(zip(outputs_classes, outputs_coords, hs)):
            target_labels = torch.cat(gt_point_labels, dim=0)
            # kpt cls
            loss_cls = self.loss_cls(cls_scores.reshape(-1, self.num_classes),
                                     target_labels.reshape(-1),
                                     reduction_override='none').sum() / max(1, sum(num_gts) * self.num_keypoints)
            losses[f'd{i}.loss_point_cls'] = loss_cls
            # kpt acc
            acc = accuracy(cls_scores.reshape(-1, self.num_classes),
                           target_labels.reshape(-1))
            losses[f'd{i}.pos_acc'] = acc
            
            # kpt discriminate
            loss_dis = self.discriminate_loss(part_feats)
            losses[f'd{i}.loss_dis'] = loss_dis
            
            # kpt loss
            imgs_wh_ = [imgs_whwh[j].unsqueeze(0).expand(num_gts[j], -1, -1)
                       for j in range(len(num_gts))]
            
            matched_row_inds, matched_col_inds = self.part_keypoint_assign(pred_points, gt_fg_points, 
                                      gt_visible_weights, num_gts=num_gts, imgs_wh=imgs_wh_)
            
            target_points = torch.cat(gt_fg_points) / torch.cat(imgs_wh_)
            
            vis_preds = []
            vis_targets = []
            invis_preds = []
            init_mask = torch.ones(self.num_keypoints).to(pred_points.device).bool()
            for i_pos, (row_inds, col_inds, preds_, targets_) in \
                enumerate(zip(matched_row_inds, matched_col_inds, pred_points, target_points)):
                vis_preds.append(preds_[row_inds[row_inds != -1]])
                vis_targets.append(targets_[col_inds[col_inds != -1]])
                
                invis_mask = init_mask.clone()
                invis_mask[row_inds[row_inds != -1]] = False
                invis_preds.append(preds_[invis_mask])
                
            vis_preds = torch.cat(vis_preds, dim=0)
            vis_targets = torch.cat(vis_targets, dim=0)
            invis_preds = torch.cat(invis_preds, dim=0)
            
            loss_kpt = self.loss_point(vis_preds, vis_targets,
                                      reduction_override='none').sum() / max(1, vis_targets.size(0))
            losses[f'd{i}.loss_point'] = loss_kpt + invis_preds.sum() * 0
            
        dedetr_results = dict(
            pseudo_points=pseudo_points,
            pseudo_bin_labels=pseudo_bin_labels,
            fg_points=fg_points,
            outputs_coords=outputs_coords,
            instance_cos_fg_maps=instance_cos_fg_maps,
            intermediate_shifted_points_refine=intermediate_shifted_points_refine,
            intermediate_attn_weights=intermediate_attn_weights,
            all_semantic_points=all_semantic_points,
            all_visible_weights=all_visible_weights,
            all_part_attn_maps=all_part_attn_maps,
            fg_masks=fg_masks
            # vis_pred_keypoints=vis_pred_keypoints,
            # vis_target_keypoints=vis_target_keypoints,
            # vis_target_bboxes=vis_target_bboxes,
            # vis_pos_inds=vis_pos_inds,
            # intermediate_shifted_points_init=intermediate_shifted_points_init,
            # vis_row_inds=vis_row_inds,
            # vis_col_inds=vis_col_inds,
            # transfered_bboxes_xyxy=transfered_bboxes_xyxy
        )
        
        return losses, dedetr_results
    
        
    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list'))
    def loss(self,
             all_cls_scores,
             temp_bbox_preds,
             all_bbox_preds, 
             all_points_preds, 
             enc_cls_scores,
             # enc_bbox_preds,
             enc_kpt_preds,
             gt_bboxes_list,
             gt_labels_list,
             gt_points_list,
             gt_visible_weights_list,
             img_metas,
             gt_bboxes_ignore=None):
        
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        num_dec_layers = len(all_cls_scores)
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_visible_weights_list = [gt_visible_weights_list for _ in range(num_dec_layers)]
        all_gt_points_list = [gt_points_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]
        
         
        losses_cls, losses_bbox, losses_iou, losses_point, \
            cls_acc, refine_targets = multi_apply(
                self.loss_single, all_cls_scores, temp_bbox_preds, all_bbox_preds, all_points_preds,
                all_gt_bboxes_list, all_gt_labels_list, all_gt_points_list,
                all_gt_visible_weights_list, img_metas_list,
                all_gt_bboxes_ignore_list)

        loss_dict = dict()
        
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(img_metas))
            ]
            enc_loss_cls, enc_losses_kpt, cls_acc_rpn = \
                self.loss_single_rpn(
                    enc_cls_scores, enc_kpt_preds, binary_labels_list,
                    gt_points_list, gt_visible_weights_list, img_metas)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_pos_acc'] = cls_acc_rpn
            loss_dict['enc_loss_kpt'] = enc_losses_kpt
            
        num_dec_layer = 0
        # for loss_cls_i, loss_point_i in zip(losses_cls[:-1], losses_point[:-1]):
        for loss_cls_i, loss_bbox_i, loss_iou_i, loss_point_i, cls_acc_i in \
            zip(losses_cls, losses_bbox, losses_iou, losses_point, cls_acc):
            
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.pos_acc'] = cls_acc_i
            loss_dict[f'd{num_dec_layer}.loss_point'] = loss_point_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
            num_dec_layer += 1
            
        # loss_dict['pos_inds'] = torch.stack(pos_inds, dim=0)
        return loss_dict, refine_targets
    
    def loss_single_rpn(self,
                        cls_scores,
                        kpt_preds,
                        gt_labels_list,
                        gt_keypoints_list,
                        gt_visible_weights_list,
                        img_metas):
        """Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            kpt_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (x_{i}, y_{i}) and
                shape [bs, num_query, K*2].
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_keypoints_list (list[Tensor]): Ground truth keypoints for each
                image with shape (num_gts, K*3) in [p^{1}_x, p^{1}_y, p^{1}_v,
                ..., p^{K}_x, p^{K}_y, p^{K}_v] format.
            gt_areas_list (list[Tensor]): Ground truth mask areas for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.

        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        kpt_preds_list = [kpt_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets_kpt(cls_scores_list, kpt_preds_list,
                                           gt_labels_list, gt_keypoints_list, 
                                           gt_visible_weights_list, img_metas)
        (labels_list, label_weights_list, kpt_targets_list, kpt_weights_list,
         num_total_pos, num_total_neg, all_matched_row_inds_list, 
         all_matched_col_inds_list) = cls_reg_targets
        
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        kpt_targets = torch.cat(kpt_targets_list, 0)
        kpt_weights = torch.cat(kpt_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)
        
        # 计算一下acc, rpn只有前景
        bg_class_ind = 1
        pos_inds = (labels >= 0) & (labels < bg_class_ind)
        cls_acc = accuracy(cls_scores[pos_inds], labels[pos_inds])
        
        # keypoint regression loss
        
        # 计算reg loss的时候，要考虑到匹配的visible points
        
        # kpt_targets = torch.cat(kpt_targets_list, 0)
        # kpt_weights = torch.cat(kpt_weights_list, 0)
        matched_row_inds = torch.cat(all_matched_row_inds_list, dim=0)
        matched_col_inds = torch.cat(all_matched_col_inds_list, dim=0)
        kpt_preds = kpt_preds.reshape(-1, self.num_keypoints * 2)
        # 这个target没有重新按照visible进行sort， 因为有些没有匹配到point
        pos_kpt_preds = kpt_preds[pos_inds].reshape(-1, self.num_keypoints, 2)
        # 这个target没有重新按照匹配进行sort
        pos_kpt_targets = kpt_targets[pos_inds].reshape(-1, self.num_keypoints, 2)
        # 这个weight没有重新按照匹配进行sort
        pos_kpt_weights = kpt_weights[pos_inds].reshape(-1, self.num_keypoints, 2)
        
        vis_preds = []
        vis_targets = []
        invis_preds = []
        init_mask = torch.ones(self.num_keypoints).to(kpt_preds.device).bool()
        
        for i_pos, (row_inds, col_inds, preds_, targets_, weights_) in \
            enumerate(zip(matched_row_inds, matched_col_inds, pos_kpt_preds, 
                          pos_kpt_targets, pos_kpt_weights)):
            vis_preds.append(preds_[row_inds[row_inds != -1]])
            vis_targets.append(targets_[col_inds[col_inds != -1]])
            
            invis_mask = init_mask.clone()
            invis_mask[row_inds[row_inds != -1]] = False
            invis_preds.append(preds_[invis_mask])
            
        vis_preds = torch.cat(vis_preds, dim=0)
        vis_targets = torch.cat(vis_targets, dim=0)
        invis_preds = torch.cat(invis_preds, dim=0)
        
        # regression L1 loss
        
        num_valid_kpt = torch.clamp(
            reduce_mean(torch.ones(len(vis_targets)).to(vis_targets.device).sum()), min=1).item()
        loss_point = self.loss_point(
            vis_preds, vis_targets, avg_factor=num_valid_kpt)
        loss_point = loss_point + invis_preds.sum() * 0
        
        # kpt_preds = kpt_preds.reshape(-1, kpt_preds.shape[-1])
        # num_valid_kpt = torch.clamp(
        #     reduce_mean(kpt_weights.sum()), min=1).item()
        # # assert num_valid_kpt == (kpt_targets>0).sum().item()
        # loss_kpt = self.loss_kpt_rpn(
        #     kpt_preds, kpt_targets, kpt_weights, avg_factor=num_valid_kpt)

        return loss_cls, loss_point, cls_acc
    
    def loss_single(self,
                    cls_scores,
                    temp_bbox_preds,
                    bbox_preds,
                    point_preds,
                    gt_bboxes_list, 
                    gt_labels_list,
                    gt_points_list,
                    gt_visible_weights_list,
                    img_metas,
                    gt_bboxes_ignore_list=None):
        
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        # bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        temp_bbox_preds_list = [temp_bbox_preds[i] for i in range(num_imgs)]
        point_preds_list = [point_preds[i] for i in range(num_imgs)]
        # cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
        cls_reg_targets = self.get_targets(cls_scores_list, temp_bbox_preds_list,
                                           point_preds_list,
                                           gt_bboxes_list, gt_labels_list,
                                           gt_points_list,
                                           gt_visible_weights_list,
                                           img_metas, gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         point_targets_list, point_weights_list, num_total_pos, num_total_neg, 
         all_matched_row_inds_list, all_matched_col_inds_list) = cls_reg_targets
        
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)
        
        # 计算一下acc
        bg_class_ind = self.num_classes 
        pos_inds = (labels >= 0) & (labels < bg_class_ind)
        cls_acc = accuracy(cls_scores[pos_inds], labels[pos_inds])
        
        # bbox loss
        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()
        
        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(img_metas, bbox_preds):
            img_h, img_w, _ = img_meta['img_shape']
            # factor = bbox_pred.new_tensor([img_w, img_h, img_w,
            #                                img_h]).unsqueeze(0).repeat(
            #                                    bbox_pred.size(0), 1)
            
            # 因为是多头预测，所以需要repeat一下factor
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 8, 1).reshape(-1, 4)
            factors.append(factor)
        factors = torch.cat(factors, 0)
        
        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        
        # 因为是多头预测，所以需要repeat一下target和weights
        bbox_targets = bbox_targets.reshape(num_imgs, -1, 1, 4).repeat(1, 1, 8, 1).reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(num_imgs, -1, 1, 4).repeat(1, 1, 8, 1).reshape(-1, 4)
        
        bbox_preds = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        # loss_iou = self.loss_iou(
        #     bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos) / 8 ######## 这里除以8是head数目

        # regression L1 loss
        # loss_bbox = self.loss_bbox(
        #     bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos) / 8 ######## 这里除以8是head数目
        
        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        
        # 计算reg loss的时候，要考虑到匹配的visible points
        point_targets = torch.cat(point_targets_list, 0)
        point_weights = torch.cat(point_weights_list, 0)
        matched_row_inds = torch.cat(all_matched_row_inds_list, dim=0)
        matched_col_inds = torch.cat(all_matched_col_inds_list, dim=0)
        point_preds = point_preds.reshape(-1, self.num_keypoints * 2)
        # 这个target没有重新按照visible进行sort， 因为有些没有匹配到point
        pos_point_preds = point_preds[pos_inds].reshape(-1, self.num_keypoints, 2)
        # 这个target没有重新按照匹配进行sort
        pos_point_targets = point_targets[pos_inds].reshape(-1, self.num_keypoints, 2)
        # 这个weight没有重新按照匹配进行sort
        pos_point_weights = point_weights[pos_inds].reshape(-1, self.num_keypoints, 2)
        
        vis_preds = []
        vis_targets = []
        invis_preds = []
        init_mask = torch.ones(self.num_keypoints).to(point_preds.device).bool()
        
        for i_pos, (row_inds, col_inds, preds_, targets_, weights_) in \
            enumerate(zip(matched_row_inds, matched_col_inds, pos_point_preds, 
                          pos_point_targets, pos_point_weights)):
            vis_preds.append(preds_[row_inds[row_inds != -1]])
            vis_targets.append(targets_[col_inds[col_inds != -1]])
            
            invis_mask = init_mask.clone()
            invis_mask[row_inds[row_inds != -1]] = False
            invis_preds.append(preds_[invis_mask])
            
        vis_preds = torch.cat(vis_preds, dim=0)
        vis_targets = torch.cat(vis_targets, dim=0)
        invis_preds = torch.cat(invis_preds, dim=0)
        
        # regression L1 loss
        num_valid_kpt = torch.clamp(
            reduce_mean(torch.ones(len(vis_targets)).to(vis_targets.device).sum()), min=1).item()
        loss_point = self.loss_point(
            vis_preds, vis_targets, avg_factor=num_valid_kpt)
        loss_point = loss_point + invis_preds.sum() * 0
        # # regression L1 loss
        # loss_bbox = self.loss_bbox(
        #     bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        
        return loss_cls, loss_bbox, loss_iou, loss_point, cls_acc, \
                (pos_inds, point_preds_list, 
                 point_targets_list, labels_list, bbox_targets_list,
                 all_matched_row_inds_list, all_matched_col_inds_list
                ) #, loss_iou

    def get_targets_kpt(self,
                    cls_scores_list,
                    # bbox_preds_list, 
                    points_preds_list,
                    # gt_bboxes_list, 
                    gt_labels_list,
                    gt_points_list,
                    gt_visible_weights_list,
                    img_metas,
                    gt_bboxes_ignore_list=None):
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        # (labels_list, label_weights_list, bbox_targets_list,
        #  bbox_weights_list, point_targets_list, point_weights_list,
        #  pos_inds_list, neg_inds_list, all_matched_row_inds_list, 
        #  all_matched_col_inds_list) = multi_apply(
        #      self._get_target_single, cls_scores_list, bbox_preds_list,
        #      points_preds_list, gt_bboxes_list, gt_labels_list, 
        #      gt_points_list, gt_visible_weights_list,
        #      img_metas, gt_bboxes_ignore_list)
        
        (labels_list, label_weights_list, point_targets_list, 
         point_weights_list, pos_inds_list, neg_inds_list, 
         all_matched_row_inds_list, all_matched_col_inds_list) = multi_apply(
             self._get_target_kpt_single, cls_scores_list, 
             points_preds_list, gt_labels_list, 
             gt_points_list, gt_visible_weights_list,
             img_metas, gt_bboxes_ignore_list)
        
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, point_targets_list, 
                point_weights_list, num_total_pos, num_total_neg,
                all_matched_row_inds_list, all_matched_col_inds_list
               )
        
    def _get_target_kpt_single(self,
                           cls_score,
                           # bbox_pred,
                           point_pred,
                           # gt_bboxes,
                           gt_labels,
                           gt_points,
                           gt_visible_weights,
                           img_meta,
                           gt_bboxes_ignore=None):

        # num_bboxes = bbox_pred.size(0)
        num_bboxes = point_pred.size(0)
        # assigner and sampler
        # assign_result = self.assigner.assign(bbox_pred, cls_score, point_pred,
        #                                      gt_bboxes, gt_labels, gt_points,
        #                                      gt_visible_weights, 
        #                                      img_meta, gt_bboxes_ignore)
        assign_result = self.assigner.assign(None, cls_score, point_pred,
                                             None, gt_labels, gt_points,
                                             gt_visible_weights, 
                                             img_meta, gt_bboxes_ignore)
        
        # sampling_result = self.sampler.sample(assign_result, bbox_pred,
        #                                       gt_bboxes)
        sampling_result = self.sampler.sample(assign_result, point_pred,
                                              gt_points)
        
        # 加入一个inner 匈牙利匹配，获得每个预测结果应该最小化匹配目标点的index
        all_matched_row_inds, all_matched_col_inds = sampling_result.inner_matches
        
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        # labels = gt_bboxes.new_full((num_bboxes, ),
        #                             self.num_classes,
        #                             dtype=torch.long)
        # labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        # label_weights = gt_bboxes.new_ones(num_bboxes)
        labels = gt_points.new_full((num_bboxes, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_points.new_ones(num_bboxes)

        # bbox targets
        
        # bbox_targets = torch.zeros_like(bbox_pred)
        # bbox_weights = torch.zeros_like(bbox_pred)
        # bbox_weights[pos_inds] = 1.0
        
        # point targets
        point_targets = torch.zeros_like(point_pred)
        point_weights = torch.zeros_like(point_pred)
        point_weights[pos_inds] = 1.0
        
        img_h, img_w, _ = img_meta['img_shape']

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        
        # factor = bbox_pred.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)
        # pos_gt_bboxes_normalized = gt_bboxes[sampling_result.pos_assigned_gt_inds] / factor
        # pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        # bbox_targets[pos_inds] = pos_gt_bboxes_targets
        
        # point 匹配
        factor = point_pred.new_tensor([img_w, img_h]).reshape(1, 1, 2).expand(-1, self.num_keypoints, -1).reshape(1, -1)
        pos_gt_points_normalized = gt_points[sampling_result.pos_assigned_gt_inds] / factor        
        point_targets[pos_inds] = pos_gt_points_normalized
        
        # return (labels, label_weights, bbox_targets, bbox_weights, 
        #         point_targets, point_weights, pos_inds,
        #         neg_inds, all_matched_row_inds, all_matched_col_inds)
        return (labels, label_weights, 
                point_targets, point_weights, pos_inds,
                neg_inds, all_matched_row_inds, all_matched_col_inds)
    
    
    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list, 
                    points_preds_list,
                    gt_bboxes_list, 
                    gt_labels_list,
                    gt_points_list,
                    gt_visible_weights_list,
                    img_metas,
                    gt_bboxes_ignore_list=None):
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, point_targets_list, point_weights_list,
         pos_inds_list, neg_inds_list, all_matched_row_inds_list, 
         all_matched_col_inds_list) = multi_apply(
             self._get_target_single, cls_scores_list, bbox_preds_list,
             points_preds_list, gt_bboxes_list, gt_labels_list, 
             gt_points_list, gt_visible_weights_list,
             img_metas, gt_bboxes_ignore_list)
        
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, point_targets_list, point_weights_list,
                num_total_pos, num_total_neg,
                all_matched_row_inds_list, all_matched_col_inds_list
               )

        
    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           point_pred,
                           gt_bboxes,
                           gt_labels,
                           gt_points,
                           gt_visible_weights,
                           img_meta,
                           gt_bboxes_ignore=None):

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        assign_result = self.assigner.assign(bbox_pred, cls_score, point_pred,
                                             gt_bboxes, gt_labels, gt_points,
                                             gt_visible_weights, 
                                             img_meta, gt_bboxes_ignore)
        
        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        
        # 加入一个inner 匈牙利匹配，获得每个预测结果应该最小化匹配目标点的index
        all_matched_row_inds, all_matched_col_inds = sampling_result.inner_matches
        
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0
        
        # point targets
        point_targets = torch.zeros_like(point_pred)
        point_weights = torch.zeros_like(point_pred)
        point_weights[pos_inds] = 1.0
        
        img_h, img_w, _ = img_meta['img_shape']

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        
        factor = bbox_pred.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)
        pos_gt_bboxes_normalized = gt_bboxes[sampling_result.pos_assigned_gt_inds] / factor
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        bbox_targets[pos_inds] = pos_gt_bboxes_targets
        
        factor = point_pred.new_tensor([img_w, img_h]).reshape(1, 1, 2).expand(-1, self.num_keypoints, -1).reshape(1, -1)
        pos_gt_points_normalized = gt_points[sampling_result.pos_assigned_gt_inds] / factor        
        point_targets[pos_inds] = pos_gt_points_normalized
        
        return (labels, label_weights, bbox_targets, bbox_weights, 
                point_targets, point_weights, pos_inds,
                neg_inds, all_matched_row_inds, all_matched_col_inds)
    
    def uniform_sample_grid(self, maps, vit_feat, 
                            rois=None, thr=0.35, 
                            n_points=20, gt_points=None):
        select_coords = []
        for i_obj, map_ in enumerate(maps):
            pos_map = map_ >= thr
            num_pos = pos_map.sum()
            pos_idx = pos_map.nonzero() # 这里用到的nonzero获得坐标是 y,x 而不是x,y
            if num_pos >= n_points: # 所需要采样的点，比点数多时，对其军用采样
                grid = torch.arange(0, num_pos, step=num_pos//n_points)[:n_points]
                coords = pos_idx[grid]
            elif num_pos > 0: # 点数少于采样点时候，递归取点做填充
                coords = pos_idx
                coords = fill_in_idx(coords, n_points)

            else: # num_pos == 0 # 没有点时候分两种情况
                if rois is not None: # 有roi输入时只中心点
                    coords = ((rois[i_obj][:2] + rois[i_obj][2:]) // (2 * 16)).long().view(1, 2).flip(1)
                    coords = coords.repeat(n_points, 1)
                else:
                    pos_map = map_ >= 0 # 或者直接把阈值调到0来获取正例点
                    num_pos = pos_map.sum()
                    pos_idx = pos_map.nonzero()
                    grid = torch.arange(0, num_pos, step=num_pos//n_points)[:n_points]
                    coords = pos_idx[grid]

            select_coords.append(coords)
        select_coords = torch.stack(select_coords).float()
        
        if gt_points is not None: # 引入标注点来进行dedetr特征query,注意这里面gt point是原图，并且是xy格式(需要flip之后除16)
#             select_coords = torch.cat([select_coords, (gt_points.unsqueeze(1).flip(-1) / 16).long()], dim=1)
            select_coords = torch.cat([gt_points.unsqueeze(1).flip(-1), select_coords], dim=1)
            
        prototypes = idx_by_coords(vit_feat[None].permute(0,2,3,1).expand(select_coords.shape[0],-1,-1,-1), 
                                   (select_coords[..., 0] // 16).long(), 
                                   (select_coords[..., 1] // 16).long()).clone() # 这利用可以用roi align 插值来做提高精度
        # prototypes = point_align(vit_feat[None], select_coords, self.point_feat_extractor) # 这利用可以用roi align 插值来做提高精度
        return select_coords, prototypes
            
    def get_semantic_centers(self, 
                            map_cos_fg, 
                            map_cos_bg,
                            rois, 
                            vit_feat, 
                            pos_thr=0.35,
                            n_points_sampled=20,
                            merge_thr=0.85,
                            gt_points=None,
                            gt_labels=None,
                            ):
        vit_feat = vit_feat.clone().detach()
        map_cos_fg_corr = corrosion_batch(torch.where(map_cos_fg > pos_thr, 
                                                      torch.ones_like(map_cos_fg), 
                                                      torch.zeros_like(map_cos_fg))[None], corr_size=11)[0]
        
        fg_inter = map_cos_fg_corr # 后面直接是原图大小，为了保持baseline性能而已
        map_fg = torch.where(fg_inter > pos_thr, torch.ones_like(fg_inter), torch.zeros_like(fg_inter))
        sampled_points, point_feats = self.uniform_sample_grid(map_fg, vit_feat, rois, 
                                                               thr=pos_thr, n_points=n_points_sampled,
                                                               gt_points=gt_points)
        fg_mask = map_fg.clone()
        sampled_points = sampled_points.flip(-1) # 变xy为yx
        num_gt = sampled_points.size(0)
        
        # 后面用mean shift获得稳定的目标特征点
        fg_inter = F.interpolate(map_cos_fg_corr.unsqueeze(0), vit_feat.shape[-2:], mode='bilinear')[0]
        bg_inter = F.interpolate(map_cos_bg.unsqueeze(0).max(dim=1, keepdim=True)[0], vit_feat.shape[-2:], mode='bilinear')[0]
        map_fg = torch.where(fg_inter > pos_thr, torch.ones_like(fg_inter), torch.zeros_like(fg_inter))
        pca = PCA(n_components=self.pca_dim)
        pca.fit(vit_feat.flatten(1).permute(1, 0).cpu().numpy())
        vit_feat_pca = torch.from_numpy(pca.fit_transform(vit_feat.flatten(1).permute(1, 0).cpu().numpy())).to(vit_feat.device)
        vit_feat_pca = vit_feat_pca.permute(1, 0).unflatten(1, vit_feat.shape[-2:])
        map_fg = torch.where(fg_inter > pos_thr, torch.ones_like(fg_inter), torch.zeros_like(fg_inter))
        prototypes_fg, sim_fg = self.mean_shift_grid_prototype(map_fg, vit_feat_pca, rois, 
                                                               tau=0.1, temp=0.1, 
                                                               n_shift=self.meanshift_refine_times, 
                                                               n_points=self.num_points_for_meanshift)
        
        sim_fg, idx_pos = filter_maps(sim_fg.unflatten(0, (sim_fg.shape[0]//self.num_points_for_meanshift,
                                                           self.num_points_for_meanshift)), fg_inter, bg_inter)
        split_size = idx_pos.sum(dim=-1).tolist()
        prototypes_fg = merge_maps(prototypes_fg[idx_pos.flatten()].split(split_size, dim=0), thr=merge_thr)
        # prototypes_fg = torch.split(prototypes_fg, map_fg.shape[0], dim=0)
        sim_fg = [cal_similarity(prot, vit_feat_pca.permute(1,2,0)) for prot in prototypes_fg]
        _, coord_semantic_center_split = get_center_coord(sim_fg, rois, 
                                                          gt_labels, num_max_obj=self.num_keypoints - 1)
        
        # 生成 visable weights 和 coords
        visible_weights = torch.zeros(num_gt, self.num_keypoints - 1).to(sampled_points.device).long()
        # semantic_points = torch.zeros(num_gt, self.num_keypoints - 1, 2).to(sampled_points.device)
        semantic_points = -1000 * torch.ones(num_gt, self.num_keypoints - 1, 2).to(sampled_points.device)
        
        if len(coord_semantic_center_split) == 0: # 一个semantic center都没有
            pass
        else:
            assert num_gt == len(coord_semantic_center_split)
            for i_gt, points in enumerate(coord_semantic_center_split):
                num_vis = points.size(0)
                visible_weights[i_gt][:num_vis] = 1
                semantic_points[i_gt][:num_vis] = points
                
        gt_visible_weights = torch.ones(num_gt, 1).long().to(sampled_points.device)
        visible_weights = torch.cat([gt_visible_weights, visible_weights], dim=1)
        semantic_points = torch.cat([gt_points.unsqueeze(1), semantic_points], dim=1)
        
        return sampled_points, point_feats, visible_weights, semantic_points, fg_mask
    
    def mean_shift_grid_prototype(self, maps, vit_feat, rois=None, 
                                  thr=0.35, n_shift=5, output_size=(4,4), 
                                  tau=0.1, temp=0.1, n_points=20):
        # TODO: debug
        # get prototypes of earch instance
        # maps = F.interpolate(maps[None], scale_factor=1/16, mode='bilinear')[0]
        # vit_feat_pca = torch.pca_lowrank(vit_feat.flatten(1).permute(1, 0), q=64)

        prototypes = []
        select_coords = []
        for i_obj, map_ in enumerate(maps):
            pos_map = map_ >= thr
            num_pos = pos_map.sum()
            pos_idx = pos_map.nonzero()
            if num_pos >= n_points:
                grid = torch.arange(0, num_pos, step=num_pos//n_points)[:n_points]
                coords = pos_idx[grid]
            elif num_pos > 0:
                coords = pos_idx
                coords = fill_in_idx(coords, n_points)

            else: # num_pos == 0
                if rois is not None:
                    coords = ((rois[i_obj][:2] + rois[i_obj][2:]) // (2 * 16)).long().view(1, 2).flip(1)
                    coords = coords.repeat(n_points, 1)
                else:
                    pos_map = map_ >= 0
                    num_pos = pos_map.sum()
                    pos_idx = pos_map.nonzero()
                    grid = torch.arange(0, num_pos, step=num_pos//n_points)[:n_points]
                    coords = pos_idx[grid]

            select_coords.append(coords)
        select_coords = torch.stack(select_coords)
        prototypes = idx_by_coords(vit_feat[None].permute(0,2,3,1).expand(select_coords.shape[0],-1,-1,-1), 
                                   select_coords[..., 0], select_coords[..., 1]).clone()
        prot_objs = []
        sims_objs = []
        if rois is not None:
            maps_bbox = box2mask(rois//16, vit_feat.shape[-2:], default_val=0)
            prototypes, sim = cosine_shift_batch(prototypes.clone(), 
                                                 (vit_feat[None] * maps_bbox[:, None]).flatten(-2).transpose(1, 2).clone(), 
                                                 vit_feat.flatten(-2).transpose(0,1).clone(), 
                                                 tau=tau, temp=temp, n_shift=n_shift)
        else:
            prototypes, sim = cosine_shift_self(prototypes[0], 
                                                (vit_feat).flatten(-2).transpose(0,1).clone(),
                                                vit_feat.flatten(-2).transpose(0,1).clone(),
                                                tau=tau, n_shift=n_shift)
        return prototypes, sim.unflatten(-1, vit_feat.shape[-2:]).clamp(0)
            
    def instance_attention_generation(self, 
                                  attn, # num_gt, 1, img_h, img_w
                                  rois, # num_gt, 4
                                  vit_feat, # C, H, W
                                  vit_feat_be_norm, # C, H, W
                                  pos_thr=0.6, #
                                  neg_thr=0.1, 
                                  num_gt=20, 
                                  corr_size=21, # 腐蚀的kernel 大小
                                  refine_times=2, 
                                  obj_tau=0.85, 
                                  gt_points=None):
        num_points = attn.shape[0]
        attn_map = attn.detach().clone()[:, 0]
        map_cos_fg, map_cos_bg, points_bg, points_fg = get_cosine_similarity_refined_map(attn_map, 
                                                                                         vit_feat_be_norm, 
                                                                                         rois, 
                                                                                         thr_pos=pos_thr, 
                                                                                         thr_neg=neg_thr, 
                                                                                         num_points=num_gt, 
                                                                                         thr_fg=0.7, 
                                                                                         refine_times=refine_times, 
                                                                                         obj_tau=obj_tau,
                                                                                         gt_points=gt_points,
                                                                                         point_feat_extractor=self.point_feat_extractor
                                                                                        )
        neg_coords = []
        num_objs = map_cos_fg[0].shape[0]
        for i_p, map_fg, map_bg in zip(range(num_objs), map_cos_fg[-1], map_cos_bg[-1]):
            xmin, ymin, xmax, ymax = rois[i_p].int().tolist()
            H, W = map_fg.shape[-2:]
            map_crop_fg = map_fg[ymin:ymax, xmin:xmax]
            map_crop_bg = map_bg[ymin:ymax, xmin:xmax]
            # 只用来选反例
            coor = get_mask_points_single_box_cos_map_fg_bg(map_crop_fg, map_crop_bg,
                                                            pos_thr=pos_thr, neg_thr=neg_thr, 
                                                            num_gt=self.iam_num_points_init, i=i_p, 
                                                            corr_size=corr_size)
            coor[:, 0] += ymin
            coor[:, 1] += xmin
            coor = coor.flip(1)
            neg_coords.append(coor)
            # label_chosen.append(label)
        neg_coords = torch.stack(neg_coords).float() # 图像尺度大小xy格式        
        neg_prototypes = idx_by_coords(vit_feat[None].permute(0,2,3,1).expand(neg_coords.shape[0],-1,-1,-1), 
                                   (neg_coords[..., 1] // 16).long(), # 特征图尺度大小的点的y
                                   (neg_coords[..., 0] // 16).long()).clone() # 特征图尺度大小的点的x
        
        return neg_coords, neg_prototypes, map_cos_fg, map_cos_bg, points_bg, points_fg
    
# 用于做instance attention map的函数
def corrosion_batch(cam, corr_size=11):
    return -F.max_pool2d(-cam, corr_size, 1, corr_size//2)

def norm_attns(attns):
    N, H, W = attns.shape
    max_val, _ = attns.view(N,-1,1).max(dim=1, keepdim=True)
    min_val, _ = attns.view(N,-1,1).min(dim=1, keepdim=True)
    return (attns - min_val) / (max_val - min_val)

def decouple_instance(map_bg, map_fg):
    map_bg = normalize_map(map_bg)
    map_fg = normalize_map(map_fg)
    map_fack_bg = 1 - (map_fg*0.5 + map_bg*0.5)
    return map_bg + map_fack_bg

def box2mask(bboxes, img_size, default_val=0.5):
    N = bboxes.shape[0]
    mask = torch.zeros(N, img_size[0], img_size[1], device=bboxes.device, dtype=bboxes.dtype) + default_val
    for n in range(N):
        box = bboxes[n]
        mask[n, int(box[1]):int(box[3]+1), int(box[0]):int(box[2]+1)] = 1.0
    return mask

def normalize_map(map_):
    max_val = map_.flatten(-2).max(-1,keepdim=True)[0].unsqueeze(-1)
    map_ = (map_ / (max_val + 1e-8))
    return map_

def fill_in_idx(idx_chosen, num_gt):
    assert idx_chosen.shape[0] != 0, '不能一个点都不选!'
    if idx_chosen.shape[0] >= num_gt / 2:
        idx_chosen = torch.cat((idx_chosen, idx_chosen[:num_gt-idx_chosen.shape[0]]), dim=0)
    else:
        repeat_times = num_gt // idx_chosen.shape[0]
        idx_chosen = idx_chosen.repeat(repeat_times, 1)
        idx_chosen = fill_in_idx(idx_chosen, num_gt)
    return idx_chosen

def idx_by_coords(map_, idx0, idx1, dim=0):
    # map.shape: N, H, W, C
    # idx0.shape: N, k
    # idx1.shape: N, k
    N = idx0.shape[0]
    k = idx0.shape[-1]
    idx_N = torch.arange(N, dtype=torch.long, device=idx0.device).unsqueeze(1).expand_as(idx0)
    idx_N = idx_N.flatten(0)
    return map_[idx_N, idx0.flatten(0), idx1.flatten(0)].unflatten(dim=dim, sizes=[N, k])

def point_align(map_, coords, point_feat_extractor): # coords 是 yx的格式 # 1, C, H, W    num_gt, num_point, 2  
    # output num_gt, num_point, C
    num_gt, num_points = coords.size()[:2]
    coords = coords.flip(-1).reshape(-1, 2)
    point_boxes = torch.cat([coords, coords + 16], dim=-1).float()
    rois = bbox2roi([point_boxes.reshape(-1, 4)])
    point_feats = point_feat_extractor(
                [map_][:point_feat_extractor.num_inputs], rois)
    point_feats = point_feats.reshape(num_gt, num_points, -1)
    return point_feats

def update_density_batch(prototypes, feats, mask_weight):
    similarity = F.cosine_similarity(prototypes[:, :, None], feats[:, None], dim=-1)
    density =(similarity * mask_weight).sum(-1)
    density = 1 - torch.where(mask_weight.sum(-1)>=1, density / mask_weight.sum(-1), torch.zeros_like(density))
    return density.clamp(1e-10).unsqueeze(-1)

def sample_point_grid(maps, num_points=10, thr=0.2, is_pos=False, gt_points=None):
    ret_coords = []
    for i_obj, map_ in enumerate(maps):
        factor = 1.0
        if is_pos:
            coords = (map_ >= thr*factor).nonzero(as_tuple=False).view(-1, 2)
        else:
            coords = (map_ < thr*factor).nonzero(as_tuple=False).view(-1, 2)
        # coords = coords[:0]
        num_pos_pix = coords.shape[0] 
        
        if num_pos_pix < num_points:
            if is_pos:
                coords_chosen = torch.cat((coords, gt_points[i_obj].repeat(num_points-num_pos_pix, 1)), dim=0)
                ret_coords.append(coords_chosen)
                continue
            else:
                while num_pos_pix < num_points:
                    print(f'factor adjusted from {thr * factor} to {thr * factor * 2}')
                    factor *= 2
                    coords = (map_ < thr*factor).nonzero(as_tuple=False)
                    num_pos_pix = coords.shape[0]

        step = num_pos_pix // num_points
        idx_chosen = torch.arange(0, num_pos_pix, step=step)
        idx_chosen = torch.randint(num_pos_pix, idx_chosen.shape) % num_pos_pix
        coords_chosen = coords[idx_chosen][:num_points]
        ret_coords.append(coords_chosen)
    return torch.stack(ret_coords).flip(-1)

def cal_similarity(prototypes, feat, dim=-1):
    if isinstance(prototypes, list):
        return torch.zeros(0, 0)
    sim = F.cosine_similarity(prototypes[:, None, None,:], feat[None], dim=-1)
    return sim

def get_refined_similarity(point_coords, feats, bboxes, ratio=1, refine_times=1, tau=0.85, is_select=False, point_feat_extractor=None):
    cos_map = get_point_cos_similarity_map(point_coords, feats, ratio=ratio, point_feat_extractor=point_feat_extractor)
    # fg_map = cos_map.argmax(dim=0, keepdim=True)  < bboxes.shape[0]
    # cos_map *= fg_map
    cos_map1 = cos_map.clone()
    cos_rf = []
    bbox_mask = box2mask(bboxes//16, cos_map.shape[-2:], default_val=0)
    # cos_map[:bboxes.shape[0]] = cos_map[:bboxes.shape[0]] * bbox_mask
    if is_select:
        # cos_map_select = torch.where(idx_max_aff==range_obj[:,None,None], cos_map, torch.zeros_like(cos_map))
        cos_map[:bboxes.shape[0]] = cos_map[:bboxes.shape[0]] * bbox_mask
        # max_val = cos_map.flatten(1).max(1, keepdim=True)[0].unsqueeze(-1)
        # cos_map = cos_map / (max_val + 1e-8)
        idx_max_aff = cos_map.argmax(0, keepdim=True).expand_as(cos_map)
        range_obj = torch.arange(cos_map.shape[0], device=cos_map.device)
        cos_rf.append(torch.where(idx_max_aff==range_obj[:,None,None], cos_map.clone(), torch.zeros_like(cos_map)))
    else:
        cos_rf.append(cos_map.clone())

    for i in range(refine_times):
        # fg_map = cos_map1.argmax(dim=0, keepdim=True)  < bboxes.shape[0]
        # cos_map1 *= fg_map
        # cos_map1[:bboxes.shape[0]] = bbox_mask * cos_map1[:bboxes.shape[0]]
        max_val = cos_map1.flatten(1).max(1, keepdim=True)[0].unsqueeze(-1)
        thr = max_val * tau
        cos_map1[cos_map1 < thr] *= 0
        feats_mask = feats * cos_map1.unsqueeze(1)
        feats_mask = feats_mask.sum([2,3], keepdim=True) / (cos_map1.unsqueeze(1).sum([2,3], keepdim=True).clamp(1e-8))
        cos_map1 = F.cosine_similarity(feats, feats_mask, dim=1)
        if is_select:
            # cos_map_select = torch.where(idx_max_aff==range_obj[:,None,None], cos_map1, torch.zeros_like(cos_map1))
            # cos_map1[:bboxes.shape[0]] = bbox_mask * cos_map1[:bboxes.shape[0]]
            cos_map1[:bboxes.shape[0]] = cos_map1[:bboxes.shape[0]] * bbox_mask
            idx_max_aff = cos_map1.argmax(0, keepdim=True).expand_as(cos_map1)
            range_obj = torch.arange(cos_map1.shape[0], device=cos_map1.device)
            cos_rf.append(torch.where(idx_max_aff==range_obj[:,None,None], cos_map1.clone(), torch.zeros_like(cos_map1)))
        else:
            cos_rf.append(cos_map1.clone())

    return torch.stack(cos_rf)

def get_point_cos_similarity_map(point_coords, feats, ratio=1, point_feat_extractor=None):
    feat_expand = feats.permute(0,2,3,1).expand(point_coords.shape[0], -1, -1, -1)
    point_feats = idx_by_coords(feat_expand, 
                                (point_coords[...,1] // 16 * ratio).long().clamp(0, feat_expand.shape[1]),
                                (point_coords[...,0] // 16 * ratio).long().clamp(0, feat_expand.shape[2]))
    # point_feats = point_align(feats, point_coords, point_feat_extractor)
    point_feats_mean = point_feats.mean(dim=1, keepdim=True)
    sim = F.cosine_similarity(feat_expand.flatten(1,2), point_feats_mean, dim=2)
    # sim = torch.cdist(feat_expand.flatten(1,2), point_feats_mean, p=2).squeeze(-1)
    return sim.unflatten(1, (feat_expand.shape[1], feat_expand.shape[2]))

def get_cosine_similarity_refined_map(attn_maps, vit_feat, bboxes, 
                                      thr_pos=0.2, thr_neg=0.1, num_points=20, 
                                      thr_fg=0.7, refine_times=1, obj_tau=0.85, 
                                      gt_points=None, point_feat_extractor=None):
    # attn_maps是上采样16倍之后的，vit_feat是上采样前的，实验表明，上采样后的不太好，会使cos_sim_fg ~= cos_sim_bg
    attn_norm = norm_attns(attn_maps)
    points_bg = sample_point_grid(attn_norm, thr=thr_neg, num_points=num_points)
    points_fg = sample_point_grid(attn_norm, thr=thr_pos, num_points=num_points, is_pos=True, gt_points=gt_points)
    points_bg_supp = sample_point_grid(attn_norm.mean(0, keepdim=True), thr=thr_neg, num_points=num_points)
    # points_bg_supp = torch.cat([sample_point_grid(attn_norm[0].mean(0,keepdim=True)<thr_neg, num_points=num_points) for _ in range(3)],dim=0)
    points_fg = torch.cat((points_fg, points_bg_supp), dim=0)
    cos_sim_fg = F.interpolate(get_refined_similarity(points_fg, vit_feat[None], bboxes=bboxes, refine_times=refine_times, tau=obj_tau, is_select=True, point_feat_extractor=point_feat_extractor), attn_maps.shape[-2:], mode='bilinear')[:,:attn_norm.shape[0]]
    cos_sim_bg = F.interpolate(get_refined_similarity(points_bg, vit_feat[None], bboxes=bboxes, refine_times=refine_times, tau=obj_tau, point_feat_extractor=point_feat_extractor), attn_maps.shape[-2:], mode='bilinear')
    ret_map = (1 - cos_sim_bg) * cos_sim_fg
    map_val = ret_map.flatten(-2, -1).max(-1, keepdim=True)[0].unsqueeze(-1).clamp(1e-8)
    
    cos_sim_bg = decouple_instance(cos_sim_bg.clone(), ret_map.clone())
    max_val_bg = cos_sim_bg.flatten(-2, -1).max(-1, keepdim=True)[0].unsqueeze(-1).clamp(1e-8)
#    map_fg = torch.where(ret_map < map_val * thr_fg, torch.zeros_like(ret_map), torch.ones_like(ret_map))
    # map_bg = torch.where(ret_map > map_val * 0.1, torch.zeros_like(ret_map), torch.ones_like(ret_map))
    return ret_map / map_val, cos_sim_bg / max_val_bg, points_fg, points_bg

def cosine_shift_batch(prototypes, feats, feats_org=None, tau=0.1, temp=0.1, n_shift=5):
    # prototypes.shape: n_obj, n_block, n_dim
    # feat.shape: n_patches, n_dim
    for i_s in range(n_shift):
        sim_map = F.cosine_similarity(prototypes[:, :, None], feats[:, None], dim=-1)
        weight = F.softmax(sim_map/(temp*tau), dim=-1)
        feat_idx = weight.argmax(1, keepdim=True)
        prot_range = torch.arange(prototypes.shape[1], device=feat_idx.device, dtype=feat_idx.dtype)[None, :, None].expand(prototypes.shape[0], prototypes.shape[1], -1)
        mask_weight = torch.where(prot_range==feat_idx, torch.ones_like(weight), torch.zeros_like(weight))
        prototypes = torch.matmul(weight * mask_weight, feats)
        tau = update_density_batch(prototypes, feats, mask_weight)
    # prototypes = merge_pototypes(prototypes, thr=1-tau)
    if feats_org is not None:
        sim_map = F.cosine_similarity(prototypes[:, :, None, :], feats_org[None, None, :, :], dim=-1)
    else:
        sim_map = F.cosine_similarity(prototypes[:, :, None, :], feats[None, None, :, :], dim=-1)
    # sim_map = F.cosine_similarity(prototypes[:, None, :], feats[None, :, :], dim=-1)
    # weight = F.softmax(sim_map/(tau*0.1), dim=-1)
    return prototypes.flatten(0,1), sim_map.flatten(0,1)


# 只用来选择反例
def get_mask_points_single_box_cos_map_fg_bg(map_fg, map_bg, pos_thr=0.6, neg_thr=0.6, num_gt=10, i=0, corr_size=21):
    # Parameters:
    #     coords: num_pixels, 2
    #     attn:H, W
    #     cls: scalar,
    # Return:
    #     coords_chosen: num_gt, 2
    #     labels_chosen: num_gt
    device = map_fg.device
    # attn_pos = corrosion(attn_map.float(), corr_size=corr_size)
    # coord_pos = corrosion((map_fg > map_fg.max()*pos_thr).float(), corr_size=corr_size).nonzero(as_tuple=False)
    coord_neg = (map_bg > map_bg.max()*neg_thr).nonzero(as_tuple=False)
    # coord_pos_neg = torch.cat((coord_pos, coord_neg), dim=0)
    # print(f'coord_pos.shape[0] / coord_neg.shape[0]: {coord_pos.shape[0] / coord_neg.shape[0]}')
    # idx_chosen = torch.randperm(coord_pos_neg.shape[0], device=device)[:num_gt]
    idx_chosen = torch.randperm(coord_neg.shape[0], device=device)[:num_gt]
    # labels_pos_neg = torch.cat((torch.ones(coord_pos.shape[0], device=device, dtype=torch.bool),
    #                             torch.zeros(coord_neg.shape[0], device=device, dtype=torch.bool)), dim=0)
    if idx_chosen.shape[0] < num_gt:
        if idx_chosen.shape[0] == 0:
            coords_chosen = -torch.ones(num_gt, 2, dtype=torch.float, device=device)
            print(f'**************一个点都没有找到!**************')
            # 这些-1的点会在point ignore里被处理掉
            # return coords_chosen, torch.zeros(num_gt, dtype=torch.bool, device=device)
            return coords_chosen
        else:
            idx_chosen = fill_in_idx(idx_chosen, num_gt)
    coords_chosen = coord_neg[idx_chosen]
    # labels_chosen = labels_pos_neg[idx_chosen]

    return coords_chosen #, labels_chosen


def filter_maps(maps, pos_maps, neg_maps, pos_thr=0.85, neg_thr=0.8):
    map_pos = []
    map_neg = []
    maps_fore = torch.where(maps>0.8, torch.ones_like(maps), torch.zeros_like(maps))
    
    pos_score = (pos_maps[:, None] * maps_fore).sum(dim=[-2, -1]) / maps_fore.sum(dim=[-2, -1]).clamp(1e-6)
    neg_score = (neg_maps[:, None] * maps_fore).sum(dim=[-2, -1]) / maps_fore.sum(dim=[-2, -1]).clamp(1e-6)
    pos_idx = (pos_score >= pos_thr)
    # neg_idx = (neg_score >= neg_thr) & (pos_score < 0.5)
    split_size = pos_idx.sum(dim=-1).tolist()
    maps_fore = maps.flatten(0,1)[pos_idx.flatten()].split(split_size, dim=0)
    # maps_back = maps.flatten(0,1)[neg_idx.flatten()]
    return maps_fore, pos_idx

def merge_maps(prototypes, thr=0.95):
    prot_ret = []
    for prot in prototypes:
        prot_obj = []
        if prot.shape[0] == 0:
            prot_ret.append([])
            continue
        sim = F.cosine_similarity(prot[None], prot[:, None], dim=-1)
        sim_triu = torch.where(torch.triu(sim, diagonal=0) >= thr, torch.ones_like(sim), torch.zeros_like(sim))
        for i_p in range(sim_triu.shape[0]):
            weight = sim_triu[i_p]
            if weight.sum() > 0:
                prot_merge = torch.matmul(weight, prot) / (weight.sum() + 1e-8)
                prot_obj.append(prot_merge)
            sim_triu[weight>0] *= 0 
        prot_ret.append(torch.stack(prot_obj))
    return prot_ret

def get_center_coord(maps, rois, obj_label, num_max_keep=50, num_max_obj=3):
    coords = []
    labels = []
    split_size = [0 for _ in range(len(maps))]
    for i_obj, map_ in enumerate(maps):
        if map_.shape[0] == 0:
            continue
        top2 = map_.flatten(1).topk(dim=1, k=1)[0][:, -1, None, None]
        coord_top2 = (map_ >= top2).nonzero().float()
        xmin, ymin, xmax, ymax = rois[i_obj]
        label = obj_label[i_obj]
        map_area_idxsort = (map_>0.9).sum(dim=[-2,-1]).argsort(descending=True, dim=0)
        for i_prot in range(map_.shape[0]):
            if i_prot == num_max_obj - 1:
                break
            coord = (coord_top2[coord_top2[:, 0]==map_area_idxsort[i_prot]].mean(dim=0)[1:].flip(0) + 0.5) * 16 # patch坐标应该位于中心， 上采样16倍
            if (coord[0] >= xmin) & (coord[0] <= xmax) & (coord[1] >= ymin) & (coord[1] <= ymax):
                coords.append(coord)
                labels.append(label)
                split_size[i_obj] += 1

    if len(coords) == 0:
        return (torch.zeros(0, 2, dtype=rois[0].dtype, device=rois[0].device), torch.zeros(0, dtype=obj_label[0].dtype, device=obj_label[0].device)), []
    else:
        coords = torch.stack(coords)
        labels = torch.stack(labels)
        coord_split = coords.split(split_size, dim=0) # coord_split是用来监督Mask Decoder的，所以不需要有数量限制
        if coords.shape[0] > num_max_keep:
            idx_chosen = torch.randperm(coords.shape[0], device=coords.device)[:num_max_keep]
            coords = coords[idx_chosen]
            labels = labels[idx_chosen]
        return (coords, labels), coord_split