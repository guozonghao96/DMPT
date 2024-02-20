# Copyright (c) OpenMMLab. All rights reserved.
import copy
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob, constant_init
from mmcv.runner import force_fp32
from mmcv.cnn import (PLUGIN_LAYERS, Conv2d, ConvModule, caffe2_xavier_init,
                      normal_init, xavier_init)

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
import mmcv
import scipy.interpolate
from mmdet.core import BitmapMasks, PolygonMasks
from ..builder import HEADS, build_loss
from ..builder import DETECTORS, build_backbone, build_head, build_neck

from mmcv.ops import point_sample

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

@HEADS.register_module()
class PointFormerHead(PointDETRHead):

    def __init__(self,
                 num_query=300,
                 num_neg_points=96,
                 num_classes=20,
                 in_channels=256,
                 with_box_refine=True,
                 sync_cls_avg_factor=True,
                 as_two_stage=True,
                 num_keypoints=8,
                 foreground_topk=6,
                 init_point_mil_head=None,
                 transformer=None,
                 num_outs=3,
                 train_cfg=dict(
                    assigner=dict(
                        type='HungarianPointAssigner',
                        cls_cost=dict(type='FocalLossCost', weight=1.0),
                        reg_cost=dict(type='PointL1Cost', weight=10.0)),
                    sampler=dict(type='PointPseudoSampler'),
                 ),
                 loss_mask=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='mean',
                     loss_weight=5.0),
                 loss_dice=dict(
                     type='DiceLoss',
                     use_sigmoid=True,
                     activate=True,
                     reduction='mean',
                     naive_dice=True,
                     eps=1.0,
                     loss_weight=5.0),
                 **kwargs):
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        self.num_keypoints = num_keypoints
        self.transform_method = 'minmax'
        self.num_query = num_query
        self.num_outs = num_outs
        self.num_neg_points = num_neg_points
        self.foreground_topk = foreground_topk
        
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage

        super(PointFormerHead, self).__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            transformer=transformer,
            **kwargs)
        
        self.assigner = build_assigner(train_cfg.assigner)
        self.sampler = build_sampler(train_cfg.sampler, context=self)
        self.loss_mask = build_loss(loss_mask)
        self.loss_dice = build_loss(loss_dice)
        
        self.init_point_mil_head = build_head(init_point_mil_head)
        
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
        
        mask_embed = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims), nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims), nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims))
        
        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers
        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
            self.mask_embed = mask_embed
        else:
            assert False, 'no implementation'

        # if not self.as_two_stage:
        self.query_embedding = nn.Embedding(self.num_query, self.embed_dims * 2)
                   
    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m.bias, bias_init)
        for m in self.reg_branches:
            constant_init(m[-1], 0, bias=0)
        nn.init.constant_(self.reg_branches[0][-1].bias.data[2:], -2.0)
        for m in self.mask_embed:
            pass
        
    def discriminate_loss(self, part_feats):
        norm_part_feats = torch.norm(part_feats, p=2, dim=-1)
        norm_matrix = torch.bmm(norm_part_feats.unsqueeze(-1), norm_part_feats.unsqueeze(-2))
        dot_matrix = torch.bmm(part_feats, part_feats.transpose(-2, -1))
        eye = ~torch.eye(self.num_keypoints).bool().to(part_feats.device)
        dis_loss = ((eye * (dot_matrix / norm_matrix)).abs()).sum()
        return self.discriminate_loss_weight * dis_loss
        
    def points2bbox(self, pts):
        pts_x = pts[..., 0]
        pts_y = pts[..., 1]
        if self.transform_method == 'minmax':
            bbox_left = pts_x.min(dim=-1, keepdim=True)[0]
            bbox_right = pts_x.max(dim=-1, keepdim=True)[0]
            bbox_up = pts_y.min(dim=-1, keepdim=True)[0]
            bbox_bottom = pts_y.max(dim=-1, keepdim=True)[0]
            bbox = torch.cat([bbox_left, bbox_up, bbox_right, bbox_bottom],
                             dim=-1)
        else:
            raise NotImplementedError
        return bbox
    
    def forward_head(self, mask_embed, mask_feature):
        # shape (num_queries, batch_size, c)
        # mask_embed = mask_embed.transpose(0, 1)
        # shape (num_queries, batch_size, h, w)
        mask_pred = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_feature)
        return mask_pred
    
    def forward_init_mil(self, 
                         pos_init_points,
                         pos_gt_labels,
                         mask_features,
                         pos_point_preds,
                         instance_imgs_wh,
                         num_gts_imgs,
                         img_size=None,
                         pos_pred_masks=None):
        num_imgs = mask_features.size(0)
        num_stages, num_gts, num_heads, num_points_per_head = \
            pos_init_points.size()[:4]
        pos_init_points = pos_init_points.reshape(num_stages, num_gts, num_heads * num_points_per_head, 2)
        all_pos_scores = []
        all_bag_scores = []
        all_neg_scores = []
        all_pos_gt_scores = []
        
        selected_pos_points = [] # num_stages, num_image, num_gt, num_points, 2
        selected_pos_valid_flags = [] # num_stages, num_image, num_gt, num_points
        selected_neg_points = [] # num_stages, num_image, num_points, 2
        selected_neg_valid_flags = [] # num_stages, num_image, num_gt, num_points
        selected_pos_point_pred_masks = [] # num_stages, num_image, num_gt, num_points, dim
        selected_neg_point_pred_masks = [] # num_stages, num_image, num_points, dim
        selected_pos_bin_labels = [] # num_stages, num_image, num_gt, num_points
        selected_neg_bin_labels = [] # num_stages, num_image, num_points
        
        
        for i_stage, (pos_dcn_points, pos_pred_points, gt_bag_labels, pred_masks) in \
            enumerate(zip(pos_init_points, pos_point_preds, pos_gt_labels, pos_pred_masks)):
            pos_dcn_points_per_img = torch.split(pos_dcn_points, num_gts_imgs, dim=0)
            pos_pred_points_per_img = torch.split(pos_pred_points, num_gts_imgs, dim=0)
            instance_imgs_wh_per_img = torch.split(instance_imgs_wh[0], num_gts_imgs, dim=0)
            gt_bag_labels_per_img = torch.split(gt_bag_labels, num_gts_imgs, dim=0)
            pred_masks_per_img = torch.split(pred_masks, num_gts_imgs, dim=0)
            
            selected_pos_points_ = []
            selected_pos_valid_flags_ = []
            selected_neg_points_ = []
            selected_neg_valid_flags_ = []
            selected_pos_point_pred_masks_ = []
            selected_neg_point_pred_masks_ = []
            selected_pos_bin_labels_ = []
            selected_neg_bin_labels_ = []
            
            for i_img, (pos_dcn_points_, pos_pred_points_, features_, imgs_wh_, gt_labels_, pred_masks_) in \
                enumerate(zip(pos_dcn_points_per_img, pos_pred_points_per_img,
                             mask_features, instance_imgs_wh_per_img, gt_bag_labels_per_img,
                             pred_masks_per_img)):
                pos_points_ = pos_dcn_points_
                pos_features = point_sample(features_.unsqueeze(0).repeat(pos_points_.size(0), 1, 1, 1), 
                                            pos_points_).permute(0, 2, 1)
                
                pos_gt_features = point_sample(features_.unsqueeze(0).repeat(pos_points_.size(0), 1, 1, 1), 
                                            pos_pred_points_.unsqueeze(1)).permute(0, 2, 1)
                
                pos_points_ori_size = pos_points_ * imgs_wh_
                pos_pseudo_bboxes = self.points2bbox(pos_points_ori_size)
                white_map = torch.ones((int(img_size[1]), int(img_size[0]))).to(features_.device)
                for (bbox, ) in zip(pos_pseudo_bboxes):
                    xs = torch.clamp(bbox[0::2], min=0, max=img_size[0] - 1).long()
                    ys = torch.clamp(bbox[1::2], min=0, max=img_size[1] - 1).long()
                    white_map[ys[0]:ys[1], xs[0]:xs[1]] = 0
                    
                neg_coords = white_map.nonzero()
                neg_coords = neg_coords[torch.randperm(len(neg_coords))[:pos_points_.size(0) * self.num_neg_points]]
                neg_points_ = torch.stack([neg_coords[:, 1] / img_size[0],
                                           neg_coords[:, 0] / img_size[1]], dim=-1).float()
                neg_features = point_sample(features_.unsqueeze(0), neg_points_.unsqueeze(0)).permute(0, 2, 1)
                
                # forward in each img of each stage
                pos_scores, bag_scores, neg_scores, pos_gt_scores = self.init_point_mil_head(pos_features, 
                                                                                 neg_features, 
                                                                                 pos_gt_features)
                if len(pos_scores) == 0:
                    pos_points = -torch.ones((0, num_heads * num_points_per_head, 2)).to(pos_scores.device)
                    pos_valid_flags = torch.zeros((0, num_heads * num_points_per_head)).to(pos_scores.device).bool()
                    neg_points = -torch.ones((0, 2)).to(pos_scores.device)
                    neg_valid_flags = torch.zeros((0)).to(pos_scores.device).bool()
                else:
                # select pseudo gt labels
                    with torch.no_grad():
                        pos_points, pos_valid_flags, neg_points, neg_valid_flags  = \
                            self.init_point_mil_head.forward_select_pseudo_point_gts(pos_scores, 
                                                                                 neg_scores.squeeze(0),
                                                                                 pos_gt_scores.squeeze(1), 
                                                                                 pos_points_,
                                                                                 neg_points_,
                                                                                 pos_pred_points_,
                                                                                 pos_gt_features.squeeze(1),
                                                                                 neg_features.squeeze(0),
                                                                                 gt_labels_,
                                                                                 imgs_wh_=imgs_wh_)
                    
                # print(pos_points.shape, pos_valid_flags.shape, neg_points.shape, neg_valid_flags.shape)
                all_pos_scores.append(pos_scores)
                all_bag_scores.append(bag_scores)
                all_neg_scores.append(neg_scores.squeeze(0))
                all_pos_gt_scores.append(pos_gt_scores.squeeze(1))
                
                pos_point_pred_masks = point_sample(pred_masks_.unsqueeze(1), pos_points).permute(0, 2, 1).squeeze(-1)
                neg_point_pred_masks = point_sample(pred_masks_.unsqueeze(1), neg_points.unsqueeze(0).repeat(pos_points.size(0), 1, 1)).permute(0, 2, 1).squeeze(0).squeeze(-1)
                
                
                selected_pos_points_.append(pos_points.reshape(-1, 2))
                selected_pos_valid_flags_.append(pos_valid_flags.reshape(-1))
                selected_neg_points_.append(neg_points.reshape(-1))
                selected_neg_valid_flags_.append(neg_valid_flags.unsqueeze(0).repeat(pos_points.size(0), 1).reshape(-1))
                selected_pos_point_pred_masks_.append(pos_point_pred_masks.reshape(-1)) 
                selected_neg_point_pred_masks_.append(neg_point_pred_masks.reshape(-1))
                selected_pos_bin_labels_.append(torch.ones_like(pos_point_pred_masks.reshape(-1)))
                selected_neg_bin_labels_.append(torch.zeros_like(neg_point_pred_masks.reshape(-1)))
                
            selected_pos_points_ = torch.cat(selected_pos_points_)
            selected_pos_valid_flags_ = torch.cat(selected_pos_valid_flags_)
            selected_neg_points_ = torch.cat(selected_neg_points_)
            selected_neg_valid_flags_ = torch.cat(selected_neg_valid_flags_)
            selected_pos_point_pred_masks_ = torch.cat(selected_pos_point_pred_masks_)
            selected_neg_point_pred_masks_ = torch.cat(selected_neg_point_pred_masks_)
            selected_pos_bin_labels_ = torch.cat(selected_pos_bin_labels_)
            selected_neg_bin_labels_ = torch.cat(selected_neg_bin_labels_)
            
            selected_pos_points.append(selected_pos_points_)
            selected_pos_valid_flags.append(selected_pos_valid_flags_)
            selected_neg_points.append(selected_neg_points_)
            selected_neg_valid_flags.append(selected_neg_valid_flags_)
            selected_pos_point_pred_masks.append(selected_pos_point_pred_masks_)
            selected_neg_point_pred_masks.append(selected_neg_point_pred_masks_)
            selected_pos_bin_labels.append(selected_pos_bin_labels_)
            selected_neg_bin_labels.append(selected_neg_bin_labels_)
        all_pos_scores = torch.cat(all_pos_scores)
        all_bag_scores = torch.cat(all_bag_scores)
        all_neg_scores = torch.cat(all_neg_scores)
        all_pos_gt_scores = torch.cat(all_pos_gt_scores)
        
        selected_pos_points = torch.stack(selected_pos_points)
        selected_pos_valid_flags = torch.stack(selected_pos_valid_flags)
        selected_neg_points = torch.stack(selected_neg_points)
        selected_neg_valid_flags = torch.stack(selected_neg_valid_flags)
        selected_pos_point_pred_masks = torch.stack(selected_pos_point_pred_masks)
        selected_neg_point_pred_masks = torch.stack(selected_neg_point_pred_masks)
        selected_pos_bin_labels = torch.stack(selected_pos_bin_labels)
        selected_neg_bin_labels = torch.stack(selected_neg_bin_labels)
        if num_gts == 0:
            print('123')
        # print(selected_pos_points.shape, selected_pos_valid_flags.shape, selected_pos_point_pred_masks.shape, selected_pos_bin_labels.shape)
        # loss
        losses_mil = self.init_point_mil_head.loss(all_bag_scores, 
                                                   all_neg_scores, 
                                                   all_pos_gt_scores, 
                                                   pos_gt_labels.reshape(-1))
        mil_results = dict(
            selected_pos_points=selected_pos_points,
            selected_pos_valid_flags=selected_pos_valid_flags,
            selected_neg_points=selected_neg_points,
            selected_neg_valid_flags=selected_neg_valid_flags,
            selected_pos_point_pred_masks=selected_pos_point_pred_masks,
            selected_neg_point_pred_masks=selected_neg_point_pred_masks,
            selected_pos_bin_labels=selected_pos_bin_labels,
            selected_neg_bin_labels=selected_neg_bin_labels,
        )
        return losses_mil, mil_results
            
    def forward_train(self,
                      x,  # mapper 4 x [bs, c, h, w]
                      img_metas,
                      gt_bboxes, # gt_bboxes
                      gt_labels, # gt_labels
                      gt_points, # gt_points
                      gt_masks,
                      imgs_wh):
        
        num_gts = [len(bboxes) for bboxes in gt_bboxes]
        num_imgs = len(img_metas)
        instance_imgs_wh = [imgs_wh[i].repeat(num_gts[i], 1) for i in range(num_imgs)]
        instance_imgs_wh = torch.cat(instance_imgs_wh, dim=0).unsqueeze(0).unsqueeze(-2)
        mlvl_feats = x[-self.num_outs:]
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
            
        query_embeds = self.query_embedding.weight
        
        hs, init_reference, inter_references, \
            enc_outputs_class, enc_outputs_coord, \
            intermediate_shifted_points, \
            intermediate_attn_weights, \
            mask_features = self.transformer( # mask features resnet 是输入的1/8倍 vit是1/4倍
                    mlvl_feats,
                    mlvl_masks,
                    query_embeds,
                    mlvl_positional_encodings,
                    reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                    cls_branches=self.cls_branches if self.as_two_stage else None,  # noqa:E501
                    mask_feature=x[0],
                    imgs_wh=imgs_wh,
            )
        hs = hs.permute(0, 2, 1, 3)
        num_stage = hs.size(0)
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
        mask_embeds = self.mask_embed(hs)
        
        mask_pred_list = []
        for mask_embed in mask_embeds:
            mask_pred = self.forward_head(mask_embed, mask_features)
            mask_pred_list.append(mask_pred)
        outputs_masks = torch.stack(mask_pred_list)
        pred_h, pred_w = outputs_masks.size()[-2:]
        if self.as_two_stage:
            outs = (outputs_classes, \
                    outputs_coords, enc_outputs_class, \
                    enc_outputs_coord.sigmoid())
        else:
            assert False, 'no implementation'
        
        # deformable detr point cls + reg
        loss_inputs = outs + (gt_labels, gt_points, img_metas)
        losses, refine_targets = self.loss(*loss_inputs, gt_bboxes_ignore=None)
        
        pos_inds = []
        point_targets = []
        point_preds = []
        matched_labels = []
        pos_gt_assigned_inds = []
        for pos_inds_, point_preds_, point_targets_, labels_, pos_gt_assigned_ in refine_targets:
            point_targets_ = torch.cat(point_targets_, dim=0)
            point_preds_ = torch.cat(point_preds_, dim=0)
            labels_ = torch.cat(labels_, dim=0)
            # pos_gt_assigned_ = torch.cat(pos_gt_assigned_, dim=0)
            pos_inds.append(pos_inds_)
            point_targets.append(point_targets_[pos_inds_])
            point_preds.append(point_preds_[pos_inds_])
            matched_labels.append(labels_[pos_inds_])
            pos_gt_assigned_inds.append(pos_gt_assigned_)
        
        pos_gt_labels = torch.stack(matched_labels)
        pos_point_targets = torch.stack(point_targets)
        pos_point_preds = torch.stack(point_preds)
        pos_inds = torch.stack(pos_inds)
        pos_dcn_points = []
        pos_pred_masks = []
        pos_dcn_weights = []
        for pos_inds_, dcn_points, dcn_weights, pred_masks in \
                zip(pos_inds, intermediate_shifted_points, intermediate_attn_weights, outputs_masks):
            pos_dcn_points.append(dcn_points.reshape(num_imgs * self.num_query, 8, -1, 2)[pos_inds_])
            pos_pred_masks.append(pred_masks.reshape(num_imgs * self.num_query, *pred_masks.size()[-2:])[pos_inds_])
            pos_dcn_weights.append(dcn_weights.reshape(num_imgs * self.num_query, 8, -1)[pos_inds_])
            
        pos_dcn_points = torch.stack(pos_dcn_points).detach()
        pos_dcn_weights = torch.stack(pos_dcn_weights).detach()
        pos_pred_masks = torch.stack(pos_pred_masks)
        
        # dcn points mil loss
        losses_mil, mil_results = self.forward_init_mil(pos_dcn_points, 
                                                        pos_gt_labels,
                                                        mask_features,
                                                        pos_point_preds,
                                                        instance_imgs_wh,
                                                        num_gts,
                                                        img_size=imgs_wh[0, 0, :],
                                                        pos_pred_masks=pos_pred_masks)
        # mask2former mask loss
        
        
        # # 只取topk的点作为前景
        # _, top_index = torch.topk(pos_dcn_weights, k=self.foreground_topk, dim=-1)
        # pos_dcn_points = torch.gather(pos_dcn_points, dim=-2, 
        #                               index=top_index.unsqueeze(-1).repeat(1, 1, 1, 1, 2)).reshape(self.transformer.decoder.num_layers,
        #                                                                                            -1, self.foreground_topk * 8, 2)
        
        # loss mask
        # pseudo_bboxes = self.points2bbox(instance_imgs_wh * pos_dcn_points) / 4 # 除以4是因为map是1/4的尺度(vit是4) # 除以8的话是resnet
        # mask_targets = torch.ones_like(pos_pred_masks)
        mask_targets = []
        for i in range(self.transformer.decoder.num_layers):
            gt_masks_ = []
            for i_img, masks in enumerate(gt_masks):
                gt_mask_maps = masks.masks[pos_gt_assigned_inds[i][i_img].cpu().numpy()].astype(np.bool)
                ori_img_h, ori_img_w = gt_mask_maps.shape[-2:]
                padded_gt_mask_maps = np.zeros((num_gts[i_img], input_img_h, input_img_w)).astype(np.long)
                padded_gt_mask_maps[:, :ori_img_h, :ori_img_w] = gt_mask_maps
                padded_gt_mask_maps = torch.as_tensor(padded_gt_mask_maps).to(pos_dcn_points.device).long()
                gt_masks_.append(padded_gt_mask_maps)
            gt_masks_ = torch.cat(gt_masks_)
            mask_targets.append(gt_masks_)
        mask_targets = torch.stack(mask_targets)
        
        for i_stage, (pos_valid_flags, pos_point_pred_masks, # num_gt1 * num_pos_points + num_gt2 * num_pos_point
                      neg_valid_flags, neg_point_pred_masks, # num_gt1 * num_neg_points + num_gt2 * num_neg_points
                      pos_bin_labels, neg_bin_labels) in \
            enumerate(zip(mil_results['selected_pos_valid_flags'], mil_results['selected_pos_point_pred_masks'],
                          mil_results['selected_neg_valid_flags'], mil_results['selected_neg_point_pred_masks'], 
                          mil_results['selected_pos_bin_labels'], mil_results['selected_neg_bin_labels'])):
            # print(pos_valid_flags.shape, pos_point_pred_masks.shape,
            #      neg_valid_flags.shape, neg_point_pred_masks.shape,
            #      pos_bin_labels.shape, neg_bin_labels.shape)
            num_pos = pos_valid_flags.sum() #pos_point_pred_masks.size(0)
            mask_point_preds = torch.cat([pos_point_pred_masks, neg_point_pred_masks])
            mask_point_targets = torch.cat([pos_bin_labels, neg_bin_labels])
            valid_flags = torch.cat([pos_valid_flags, neg_valid_flags])
            total_num_points = mask_point_preds.size(-1)
            num_total_masks = reduce_mean(mask_point_preds.new_tensor([num_pos]))
            num_total_masks = max(num_total_masks, 1)
            
            if num_pos == 0:
                # zero match
                loss_dice = mask_point_preds.sum() * 0
                loss_mask = mask_point_preds.sum() * 0
            else:
                # dice loss
                # loss_dice = mask_point_preds.sum() * 0
                num_pos_splits = [num_gt * (8 * 12 + num_gt * self.num_neg_points) for num_gt in num_gts]
                num_neg_splits = [num_gt * num_gt * self.num_neg_points for num_gt in num_gts]
                pos_pred_masks_ = torch.split(pos_point_pred_masks, num_pos_splits, dim=0)
                neg_pred_masks_ = torch.split(neg_point_pred_masks, num_neg_splits, dim=0)
                pos_bin_labels_ = torch.split(pos_bin_labels, num_pos_splits, dim=0) 
                neg_bin_labels_ = torch.split(neg_bin_labels, num_neg_splits, dim=0)
                pos_valid_flags_ = torch.split(pos_valid_flags, num_pos_splits, dim=0) 
                neg_valid_flags_ = torch.split(neg_valid_flags, num_neg_splits, dim=0) 
                
                loss_dice = 0
                for i_img in range(num_imgs):
                    loss_dice += self.loss_dice(
                        torch.cat([pos_pred_masks_[i_img].reshape(num_gts[i_img], 8 * 12 + num_gts[i_img] * self.num_neg_points), 
                                   neg_pred_masks_[i_img].reshape(num_gts[i_img], num_gts[i_img] * self.num_neg_points)], dim=-1),
                        torch.cat([pos_bin_labels_[i_img].reshape(num_gts[i_img], 8 * 12 + num_gts[i_img] * self.num_neg_points),
                                   neg_bin_labels_[i_img].reshape(num_gts[i_img], num_gts[i_img] * self.num_neg_points)], dim=-1),
                        valid_flags=torch.cat([pos_valid_flags_[i_img].reshape(num_gts[i_img], 8 * 12 + num_gts[i_img] * self.num_neg_points), 
                                               neg_valid_flags_[i_img].reshape(num_gts[i_img], num_gts[i_img] * self.num_neg_points)], dim=-1),
                        avg_factor=num_total_masks)
                # mask loss
                # shape (num_queries, num_points) -> (num_queries * num_points, )
                mask_point_preds = mask_point_preds.reshape(-1)
                # shape (num_total_gts, num_points) -> (num_total_gts * num_points, )
                mask_point_targets = mask_point_targets.reshape(-1)
                loss_mask = self.loss_mask(
                    mask_point_preds,
                    mask_point_targets,
                    weight=valid_flags,
                    avg_factor=num_total_masks * total_num_points)
                
            losses[f'd{i_stage}.loss_dice'] = loss_dice
            losses[f'd{i_stage}.loss_mask'] = loss_mask
            
#         all_pred_mask_points = []
#         all_mask_point_labels = []
#         for i, (pred_masks, target_maps, pos_points, bboxes) in \
#                 enumerate(zip(pos_pred_masks, mask_targets, 
#                               pos_dcn_points, pseudo_bboxes)):
#             num_pos = pred_masks.size(0)
#             pos_labels = torch.ones_like(pos_points[..., 0])
#             neg_points = []
#             target_maps_ = torch.ones_like(pos_pred_masks[0])
#             for target, bbox in zip(target_maps_, bboxes):
#                 xs = torch.clamp(bbox[0::2], min=0, max=pred_w - 1).long()
#                 ys = torch.clamp(bbox[1::2], min=0, max=pred_h - 1).long()
#                 target[ys[0]:ys[1], xs[0]:xs[1]] = 0
#                 neg_coords = target.nonzero()
#                 neg_coords = neg_coords[torch.randperm(len(neg_coords))[:self.num_neg_points]]
#                 neg_points_ = torch.stack([neg_coords[:, 1] / pred_w,
#                                            neg_coords[:, 0] / pred_h], dim=-1)
#                 neg_points.append(neg_points_)
#             neg_points = torch.stack(neg_points)
#             neg_labels = torch.zeros_like(neg_points[..., 0])
#             point_targets = torch.cat([pos_points, neg_points], dim=1)
#             mask_point_targets = point_sample(target_maps.unsqueeze(1).float(), point_targets).squeeze(1).long()
#             # mask_point_targets = torch.cat([pos_labels, neg_labels], dim=1)
#             all_pred_mask_points.append(point_targets)
#             all_mask_point_labels.append(mask_point_targets)
#             mask_point_preds = point_sample(pred_masks.unsqueeze(1), point_targets).squeeze(1) # num_gt, num_points
#             total_num_points = mask_point_preds.size(-1)
#             num_total_masks = reduce_mean(mask_point_preds.new_tensor([num_pos]))
#             num_total_masks = max(num_total_masks, 1)
            
#             if num_pos == 0:
#                 # zero match
#                 loss_dice = mask_point_preds.sum() * 0
#                 loss_mask = mask_point_preds.sum() * 0
#             else:
#                 # dice loss
#                 loss_dice = self.loss_dice(
#                     mask_point_preds, mask_point_targets, avg_factor=num_total_masks)
#                 # mask loss
#                 # shape (num_queries, num_points) -> (num_queries * num_points, )
#                 mask_point_preds = mask_point_preds.reshape(-1)
#                 # shape (num_total_gts, num_points) -> (num_total_gts * num_points, )
#                 mask_point_targets = mask_point_targets.reshape(-1)
#                 loss_mask = self.loss_mask(
#                     mask_point_preds,
#                     mask_point_targets,
#                     avg_factor=num_total_masks * total_num_points)
                
#             losses[f'd{i}.loss_dice'] = loss_dice
#             losses[f'd{i}.loss_mask'] = loss_mask
        
        # 预测的map和gt mask的顺序不align
        # 进行miou测试
        # 只用最后一层预测来进行
        
        pseudo_masks_pred = pos_pred_masks[-1].sigmoid()
        if pseudo_masks_pred.shape[0] == 0:
            score_thresholds = [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
            mean_iou = dict()
            for threshold in score_thresholds:
                mean_iou['mIoU{:s}'.format(str(threshold)[1:])] = torch.as_tensor(0.0).cuda()
            
            results=dict(
                pos_dcn_points=pos_dcn_points,
                pos_pred_masks=pos_pred_masks,
                pos_point_preds=pos_point_preds,
                instance_imgs_wh=instance_imgs_wh,
                mean_iou=mean_iou,
                mask_targets=mask_targets
            )
            results.update(mil_results)
            losses.update(losses_mil)
            return losses, results
        
        pseudo_masks_pred = F.interpolate(pseudo_masks_pred.unsqueeze(0), scale_factor=4, mode='bicubic')[0]
        pseudo_masks_pred = torch.split(pseudo_masks_pred, num_gts, dim=0)
        pseudo_masks_pred = [pred.detach().cpu().numpy() for pred in pseudo_masks_pred]
        
        gt_masks_ = []
        gt_labels_ = []
        score_thresholds = [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        mean_iou = dict()
        for threshold in score_thresholds:
            mean_iou['mIoU{:s}'.format(str(threshold)[1:])] = []
        
        true_img_num = num_imgs
        for i_img, (masks, labels) in enumerate(zip(gt_masks, gt_labels)):
            gt_mask_maps = masks.masks[pos_gt_assigned_inds[-1][i_img].cpu().numpy()].astype(np.bool)
            if gt_mask_maps.shape[0] == 0:
                true_img_num -= 1
                continue
            ori_img_h, ori_img_w = gt_mask_maps.shape[-2:]
            padded_gt_mask_maps = np.zeros((num_gts[i_img], input_img_h, input_img_w)).astype(np.bool)
            padded_gt_mask_maps[:, :ori_img_h, :ori_img_w] = gt_mask_maps
            gt_masks_.append(padded_gt_mask_maps)
            labels = labels[pos_gt_assigned_inds[-1][i_img]]
            pred_masks_ = pseudo_masks_pred[i_img]
            for threshold in score_thresholds:
                pred_masks = pred_masks_ > threshold
                inter_sum = (pred_masks & padded_gt_mask_maps).sum()
                union_sum = (pred_masks | padded_gt_mask_maps).sum()
                # if inter_sum / union_sum == np.nan:
                mean_iou['mIoU{:s}'.format(str(threshold)[1:])].append(inter_sum / union_sum)
                
        for threshold in score_thresholds:
            mean_iou['mIoU{:s}'.format(str(threshold)[1:])] = torch.tensor(sum(mean_iou['mIoU{:s}'.format(str(threshold)[1:])]) / true_img_num).cuda()
        results=dict(
            pos_dcn_points=pos_dcn_points,
            pos_pred_masks=pos_pred_masks,
            # all_pred_mask_points=all_pred_mask_points,
            pos_point_preds=pos_point_preds,
            instance_imgs_wh=instance_imgs_wh,
            # gt_masks_=gt_masks_,
            mean_iou=mean_iou,
            # all_mask_point_labels=torch.stack(all_mask_point_labels),
            mask_targets=mask_targets
        )
        results.update(mil_results)
        losses.update(losses_mil)
        return losses, results
    
    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list'))
    def loss(self,
             all_cls_scores,
             all_points_preds,
             enc_cls_scores, 
             enc_kpt_preds,
             gt_labels_list,
             gt_points_list,
             img_metas,
             gt_bboxes_ignore=None):
        
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        num_dec_layers = len(all_cls_scores)
        all_gt_points_list = [gt_points_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]
        # all_fg_masks_list = [fg_masks_list for _ in range(num_dec_layers)]
         
        losses_cls, losses_point, \
            cls_acc, refine_targets = multi_apply(
                self.loss_single, all_cls_scores, all_points_preds,
                all_gt_labels_list, all_gt_points_list,
                img_metas_list, all_gt_bboxes_ignore_list)

        loss_dict = dict()
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(img_metas))
            ]
            enc_loss_cls, enc_losses_kpt, cls_acc_rpn, _ = \
                self.loss_single(
                    enc_cls_scores, enc_kpt_preds, binary_labels_list,
                    gt_points_list, img_metas)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_pos_acc'] = cls_acc_rpn
            loss_dict['enc_loss_kpt'] = enc_losses_kpt
            
        num_dec_layer = 0
        # for loss_cls_i, loss_point_i in zip(losses_cls[:-1], losses_point[:-1]):
        for loss_cls_i, loss_point_i, cls_acc_i in \
            zip(losses_cls, losses_point, cls_acc):
            
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.pos_acc'] = cls_acc_i
            loss_dict[f'd{num_dec_layer}.loss_point'] = loss_point_i
            num_dec_layer += 1
            
        return loss_dict, refine_targets
    
    def loss_single(self,
                    cls_scores,
                    point_preds,
                    gt_labels_list,
                    gt_points_list,
                    img_metas,
                    gt_bboxes_ignore_list=None):
        
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        point_preds_list = [point_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, 
                                           point_preds_list,
                                           gt_labels_list,
                                           gt_points_list,
                                           img_metas, gt_bboxes_ignore_list)
        (labels_list, label_weights_list, point_targets_list,
         point_weights_list, num_total_pos, num_total_neg, 
         pos_assigned_gt_inds_list) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
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
        
        point_preds = point_preds.reshape(-1, 2)
        point_targets = torch.cat(point_targets_list, 0)
        point_weights = torch.cat(point_weights_list, 0)
        loss_point = self.loss_point(
            point_preds, point_targets, point_weights, avg_factor=num_total_pos)
        
        return loss_cls, loss_point, cls_acc, \
                (pos_inds, point_preds_list, 
                 point_targets_list, labels_list, pos_assigned_gt_inds_list) 
    
    def get_targets(self,
                    cls_scores_list,
                    points_preds_list,
                    gt_labels_list,
                    gt_points_list,
                    img_metas,
                    gt_bboxes_ignore_list=None):
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, point_targets_list, 
         point_weights_list, pos_inds_list, neg_inds_list,
         pos_assigned_gt_inds_list) = multi_apply(
             self._get_target_single, cls_scores_list, 
             points_preds_list, gt_labels_list, 
             gt_points_list, 
             img_metas, gt_bboxes_ignore_list)
        
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, point_targets_list, 
                point_weights_list, num_total_pos, num_total_neg,
                pos_assigned_gt_inds_list)

        
    def _get_target_single(self,
                           cls_score,
                           point_pred,
                           gt_labels,
                           gt_points,
                           img_meta,
                           gt_bboxes_ignore=None):

        num_points = point_pred.size(0)
        # assigner and sampler
        
        assign_result = self.assigner.assign(
            point_pred, cls_score, gt_points,
            gt_labels, img_meta)
        sampling_result = self.sampler.sample(
            assign_result, point_pred, gt_points)
        
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_points.new_full((num_points, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_points.new_ones(num_points)
        
        # point targets
        point_targets = torch.zeros_like(point_pred)
        point_weights = torch.zeros_like(point_pred)
        point_weights[pos_inds] = 1.0
        
        img_h, img_w, _ = img_meta['img_shape']
        
        factor = point_pred.new_tensor([img_w, img_h]).reshape(1, 2)
        pos_gt_points_normalized = gt_points[sampling_result.pos_assigned_gt_inds] / factor        
        point_targets[pos_inds] = pos_gt_points_normalized
        
        return (labels, label_weights, point_targets, point_weights, 
                pos_inds, neg_inds, sampling_result.pos_assigned_gt_inds)
    
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
            select_coords = torch.cat([select_coords, gt_points.unsqueeze(1).flip(-1)], dim=1)
            
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
        sampled_points = sampled_points.flip(-1) # 变xy为yx
        num_gt = sampled_points.size(0)
        fg_mask = map_fg.clone()
        
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
        semantic_points = torch.zeros(num_gt, self.num_keypoints - 1, 2).to(sampled_points.device)
        
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
                    # print(f'factor adjusted from {thr * factor} to {thr * factor * 2}')
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
            # print(f'**************一个点都没有找到!**************')
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