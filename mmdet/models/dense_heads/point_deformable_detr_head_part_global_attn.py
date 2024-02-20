# Copyright (c) OpenMMLab. All rights reserved.
import copy

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
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding

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
class PointDeformableDETRHead(PointDETRHead):

    def __init__(self,
                 # instance attention map的参数
                 iam_num_points_init=10,
                 iam_thr_pos=0.35, 
                 iam_thr_neg=0.8,
                 iam_refine_times=2, 
                 iam_obj_tau=0.9,
                 # instance attention map的参数
                 point_feat_extractor=None,
                 with_gt_points=False,
                 num_classes=20,
                 in_channels=256,
                 with_box_refine=False,
                 sync_cls_avg_factor=True,
                 as_two_stage=False,
                 transformer=None,
                 **kwargs):
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        self.with_gt_points = with_gt_points
        
        # instance attention map的参数
        self.iam_thr_pos = iam_thr_pos
        self.iam_thr_neg = iam_thr_neg
        self.iam_refine_times = iam_refine_times
        self.iam_obj_tau = iam_obj_tau
        self.iam_num_points_init = iam_num_points_init
        
        self.point_feat_extractor = point_feat_extractor
        
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage

        super(PointDeformableDETRHead, self).__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            transformer=transformer, 
            **kwargs)

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        
        self.point_feat_extractor = build_roi_extractor(self.point_feat_extractor)
        fc_cls = Linear(self.embed_dims, self.cls_out_channels)
        reg_branch = []
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

        self.fc_cls = Linear(self.embed_dims, self.cls_out_channels)
        self.reg_ffn = FFN(
            self.embed_dims,
            self.embed_dims,
            self.num_reg_fcs,
            self.act_cfg,
            dropout=0.0,
            add_residual=False)
        self.fc_reg = Linear(self.embed_dims, 2)
        
#         if self.with_box_refine:
#             self.cls_branches = _get_clones(fc_cls, num_pred)
#             self.reg_branches = _get_clones(reg_branch, num_pred)
#         else:
#             self.cls_branches = nn.ModuleList(
#                 [fc_cls for _ in range(num_pred)])
#             self.reg_branches = nn.ModuleList(
#                 [reg_branch for _ in range(num_pred)])

#         if not self.as_two_stage:
#             self.query_embedding = nn.Embedding(self.num_query,
#                                                     self.embed_dims * 2)
#         self.category_embedding = nn.Embedding(self.num_classes, self.embed_dims)
            
        # self.part_embedding = nn.Embedding(self.iam_num_points_init + 1 if self.with_gt_points else self.iam_num_points_init, self.embed_dims)
        self.part_embedding = nn.Embedding(self.iam_num_points_init + 1 + 1 if self.with_gt_points else self.iam_num_points_init + 1, self.embed_dims) # 最后一个是背景
        
        
    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            nn.init.constant_(self.fc_cls.bias, bias_init)
#         if self.loss_cls.use_sigmoid:
#             bias_init = bias_init_with_prob(0.01)
#             for m in self.cls_branches:
#                 nn.init.constant_(m.bias, bias_init)
#         for m in self.reg_branches:
#             constant_init(m[-1], 0, bias=0)
#         nn.init.constant_(self.reg_branches[0][-1].bias.data[2:], -2.0)
#         if self.as_two_stage:
#             for m in self.reg_branches:
#                 nn.init.constant_(m[-1].bias.data[2:], 0.0)

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
            sampled_points, pos_point_feats = self.get_semantic_centers(map_cos_fg[-1].clone(), 
                                                                    map_cos_bg[-1].clone(), 
                                                                    gt_bboxes[i_img], 
                                                                    vit_feat[i_img], 
                                                                    pos_thr=0.35,
                                                                    n_points_sampled=self.iam_num_points_init,
                                                                    gt_points=gt_points[i_img] if self.with_gt_points else None)
            pseudo_points_ = torch.cat([sampled_points, neg_points], dim=1).float()
            pseudo_bin_labels_ = torch.cat([torch.ones_like(sampled_points)[..., 0],
                                        torch.zeros_like(neg_points)[..., 0]], dim=1).bool()
            
            pseudo_points.append(pseudo_points_)
            pseudo_bin_labels.append(pseudo_bin_labels_)
            instance_cos_fg_maps.append(map_cos_fg)
            instance_cos_bg_maps.append(map_cos_bg)
            fg_points.append(sampled_points) 
            fg_point_feats.append(pos_point_feats)
            bg_point_feats.append(neg_point_feats)
            
        # point detr
        # construct binary masks which used for the transformer.
        # NOTE following the official DETR repo, non-zero values representing
        # ignored positions, while zero values means valid positions.
        x = vit_feat
        batch_size = x.size(0)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        masks = x.new_ones((batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            masks[img_id, :img_h, :img_w] = 0

        # interpolate masks to have the same spatial shape with x
        masks = F.interpolate(
            masks.unsqueeze(1), size=x.shape[-2:]).to(torch.bool).squeeze(1)
        # position encoding
        pos_embed = self.positional_encoding(masks)  # [bs, embed_dim, h, w]
        # outs_dec: [nb_dec, bs, num_query, embed_dim]
#         category_embeds = self.category_embedding.weight
        part_embeds = self.part_embedding.weight
        
        cls_outputs = []
        reg_outputs = []
        cross_attn_weights = []
        for i_img in range(num_imgs):
            
#             gt_rois = bbox2roi([gt_bboxes[i_img].reshape(-1, 4)])
#             roi_feat = self.point_feat_extractor([x[i_img].unsqueeze(0)][:self.point_feat_extractor.num_inputs], gt_rois)
#             roi_pos_embed = self.point_feat_extractor([pos_embed[i_img].unsqueeze(0)][:self.point_feat_extractor.num_inputs], gt_rois)
#             #
    
#             pos_embed_up = F.interpolate(pos_embed[i_img].unsqueeze(0), (input_img_h, input_img_w), mode='bilinear') # bs, 256, img_h, img_w
#             location_embed = idx_by_coords(pos_embed_up.permute(0,2,3,1).expand(fg_points[i_img].shape[0],-1,-1,-1), 
#                                        (fg_points[i_img][..., 1]).long(), # 特征图尺度大小的点的y
#                                        (fg_points[i_img][..., 0]).long()).clone() # 特征图尺度大小的点的x
#             num_gt, num_points = location_embed.size()[:2]

            num_gt, num_points = fg_points[i_img].size()[:2]
            # 最后一个加入一个背景的query
            part_embeds_ = part_embeds.reshape(1, num_points + 1, -1).expand(num_gt, -1, -1)
            # query_embed = location_embed + part_embeds_
            query_embed = part_embeds_
            
            # outs_dec, _, attn_weights = self.transformer(x[i_img].unsqueeze(0), masks[i_img].unsqueeze(0),
            # outs_dec, _, attn_weights = self.transformer(roi_feat, masks[i_img].unsqueeze(0),
            outs_dec, _, attn_weights = self.transformer(x[i_img].unsqueeze(0), masks[i_img].unsqueeze(0),
                                           query_embed,
                                           pos_embed[i_img].unsqueeze(0))
                                           # roi_pos_embed)
#                                            pos_embed[i_img].unsqueeze(0))

            # attn_weights -> 6, num_gt, num_points, num_patch
            cls_scores = self.fc_cls(outs_dec)
            reg_offset = self.fc_reg(F.relu(self.reg_ffn(outs_dec))).sigmoid()
            cls_outputs.append(cls_scores)
            reg_outputs.append(reg_offset)
            
            _, num_gt, num_points, _ = attn_weights.size()
            
#             attn_weights = attn_weights.reshape(6 * num_gt, num_points, 14, 14)
            attn_weights = attn_weights.reshape(6 * num_gt, num_points, input_img_h // 16, input_img_w // 16)
            attn_weights = F.interpolate(attn_weights, (input_img_h, input_img_w), mode='bilinear').reshape(6, num_gt, num_points, input_img_h, input_img_w)
            cross_attn_weights.append(attn_weights)
            
        # 计算loss
        losses = dict(
            s0_loss_point_cls=[],
            s0_loss_point_reg=[],
            s0_pos_acc=[],
            s1_loss_point_cls=[],
            s1_loss_point_reg=[],
            s1_pos_acc=[],
            s2_loss_point_cls=[],
            s2_loss_point_reg=[],
            s2_pos_acc=[],
            s3_loss_point_cls=[],
            s3_loss_point_reg=[],
            s3_pos_acc=[],
            s4_loss_point_cls=[],
            s4_loss_point_reg=[],
            s4_pos_acc=[],
            s5_loss_point_cls=[],
            s5_loss_point_reg=[],
            s5_pos_acc=[],
        )
       
        shifted_points = []
        for cls_scores, reg_offsets, labels, target_points, dy_weights_, points, bboxes, img_wh in \
                    zip(cls_outputs, reg_outputs, gt_labels, gt_points, dy_weights, fg_points, gt_bboxes, imgs_whwh):
            num_layer, num_gt, num_points = cls_scores.size()[:-1]
            
            # labels = torch.cat([labels, 
            #                     torch.ones(1).type_as(labels) * self.num_classes], dim=0)
            # cls_scores  ->  num_layer, num_gt, num_points, num_classes
            
            repeat_labels = labels.reshape(1, -1, 1).repeat(num_layer, 1, num_points - 1)
            # repeat_labels = labels.reshape(1, -1, 1).repeat(num_layer, 1, num_points)
            shifted_points_ = []
            for i_th, (layer_logits, lay_offsets, layer_labels) in enumerate(zip(cls_scores, reg_offsets, repeat_labels)):
                # 由bboxes 生成 target points
#                 xmin, ymin, xmax, ymax = torch.split(bboxes, 1, dim=-1)
#                 xc = (bboxes[:, 2] + bboxes[:, 0]).unsqueeze(1) / 2
#                 yc = (bboxes[:, 3] + bboxes[:, 1]).unsqueeze(1) / 2
#                 width = (bboxes[:, 2] - bboxes[:, 0]).unsqueeze(1)
#                 height = (bboxes[:, 3] - bboxes[:, 1]).unsqueeze(1)
                
#                 points_1_norm = torch.cat([(xc - width / 4 - xmin) / width, 
#                                            (yc - height / 4 - ymin) / height], dim=-1) 
                
#                 points_2_norm = torch.cat([(xc + width / 4 - xmin) / width, 
#                                            (yc - height / 4 - ymin) / height], dim=-1)
                
#                 points_c_norm = torch.cat([(xc - xmin) / width, 
#                                            (yc - ymin) / height], dim=-1)
                
#                 points_3_norm = torch.cat([(xc - width / 4 - xmin) / width, 
#                                            (yc + height / 4 - ymin) / height], dim=-1)
                
#                 points_4_norm = torch.cat([(xc + width / 4 - xmin) / width, 
#                                            (yc + height / 4 - ymin) / height], dim=-1)
                
#                 t_points = torch.stack([points_1_norm, points_2_norm, 
#                                         points_c_norm, 
#                                         points_3_norm, points_4_norm], dim=1)
                # reg
                # points = points + lay_offsets
                t_points = points
                pred_points = lay_offsets[:, :-1]
                img_wh_ = img_wh.repeat(num_gt * (num_points - 1), 1)
                
                _loss_reg_ = self.loss_bbox(pred_points.reshape(-1, 2),
                                            # target_points.unsqueeze(1).repeat(1, num_points, 1).reshape(-1, 2),
                                            t_points.reshape(-1, 2) / img_wh_,
                                            reduction_override='none')
                _loss_reg_ = _loss_reg_.reshape(num_gt, num_points - 1, 2) * \
                                    dy_weights_.unsqueeze(-1).repeat(1, num_points - 1, 2)
                # 反例不参与回归，所以num_points - 1
                _loss_reg_ = _loss_reg_.sum() / max(1, num_gt * (num_points - 1))
                
                losses['s{}_loss_point_reg'.format(i_th)].append(_loss_reg_.reshape(-1))
                shifted_points_.append(pred_points.detach())
                
                # cls
                layer_labels = torch.cat([layer_labels, 
                                          torch.ones(1).type_as(layer_labels).unsqueeze(0).repeat(num_gt, 1) * self.num_classes], 
                                         dim=-1)
                _loss_ = self.loss_cls(layer_logits.reshape(-1, self.num_classes),
                                        layer_labels.reshape(-1),
                                        reduction_override='none') #.sum() / max(1, num_gt * num_points) # 可以加上mil的分数作为权重
                _loss_ = _loss_.reshape(num_gt, num_points, self.num_classes) * \
                            dy_weights_.unsqueeze(-1).repeat(1, num_points, self.num_classes)
                _loss_ = _loss_.sum() / max(1, num_gt * num_points)
                losses['s{}_loss_point_cls'.format(i_th)].append(_loss_.reshape(-1))
                losses['s{}_pos_acc'.format(i_th)].append(accuracy(layer_logits.reshape(-1, self.num_classes), 
                                                         layer_labels.reshape(-1)).reshape(-1))
            shifted_points_ = torch.stack(shifted_points_, dim=0)
            shifted_points.append(shifted_points_)
            
        for i_th in range(num_layer):
            losses['s{}_loss_point_cls'.format(i_th)] = torch.cat(losses['s{}_loss_point_cls'.format(i_th)]).mean()
            losses['s{}_loss_point_reg'.format(i_th)] = torch.cat(losses['s{}_loss_point_reg'.format(i_th)]).mean()
            losses['s{}_pos_acc'.format(i_th)] = torch.cat(losses['s{}_pos_acc'.format(i_th)]).mean()
            
        dedetr_results = dict(
            pseudo_points=pseudo_points,
            pseudo_bin_labels=pseudo_bin_labels,
            fg_points=fg_points,
#             init_reference_points=init_reference,
#             inter_reference_points=inter_references,
            shifted_points=shifted_points,
            instance_cos_fg_maps=instance_cos_fg_maps,
            cross_attn_weights=cross_attn_weights,
        )
        return losses, dedetr_results
    
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
                            gt_points=None,
                            ):
        map_cos_fg_corr = corrosion_batch(torch.where(map_cos_fg > pos_thr, 
                                                      torch.ones_like(map_cos_fg), 
                                                      torch.zeros_like(map_cos_fg))[None], corr_size=11)[0]
        fg_inter = map_cos_fg_corr # 后面直接是原图大小
#         fg_inter = F.interpolate(map_cos_fg_corr.unsqueeze(0), vit_feat.shape[-2:], mode='bilinear')[0]
        map_fg = torch.where(fg_inter > pos_thr, torch.ones_like(fg_inter), torch.zeros_like(fg_inter))
        sampled_points, point_feats = self.uniform_sample_grid(map_fg, vit_feat, rois, 
                                                               thr=pos_thr, n_points=n_points_sampled,
                                                               gt_points=gt_points)
        
        sampled_points = sampled_points.flip(-1) # 变xy为yx并变为原图大小
        # 这边再加上一个反例的特征
        return sampled_points, point_feats
    
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
