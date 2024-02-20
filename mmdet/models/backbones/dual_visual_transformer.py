# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from mmcv_custom import load_checkpoint
from mmdet.utils import get_root_logger
from mmdet.models.builder import BACKBONES
from models import VisionTransformer
from models.vision_transformer import Block
from models.utils import trunc_normal_
import torch.nn.functional as F
from functools import partial
from mmcv.runner import _load_checkpoint, load_state_dict
import copy
from mmcv.cnn import ConvModule

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=[224, 224], patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.num_patches_w = img_size[0] // patch_size
        self.num_patches_h = img_size[1] // patch_size

        num_patches = self.num_patches_w * self.num_patches_h
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
            
    def forward(self, x, mask=None):
        B, C, H, W = x.shape
        return self.proj(x)


@BACKBONES.register_module()
class DualVisionTransformer(VisionTransformer):
    def __init__(self,
                 img_size,
                 patch_size,
                 embed_dim,
                 in_chans=3,
                 with_fpn=True,
                 frozen_stages=-1,
                 out_indices=[3, 5, 7, 11],
                 use_checkpoint=True,
                 learnable_pos_embed=True,
                 point_tokens_num=100,
                 point_token_attn_mask=True,
                 return_attention=True,
                 dual_depth=8,
                 num_scale=None,
                 last_feat=True,
                 last_feat_dim=256,
                 init_cfg=None,
                 with_simple_fpn=False,
                 ratios=None,
                 **kwargs):
        super(DualVisionTransformer, self).__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim, 
            return_attention=return_attention,
            point_token_attn_mask=point_token_attn_mask,
            num_point_tokens=point_tokens_num,
            **kwargs)
        
        assert not with_fpn or (patch_size in (8, 16))
        self.patch_size = patch_size
        self.init_cfg = init_cfg
        
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim),
            requires_grad=learnable_pos_embed,
        )
        
        self.with_fpn = with_fpn
        self.with_simple_fpn = with_simple_fpn
        self.frozen_stages = frozen_stages
        self.out_indices = out_indices
        self.use_checkpoint = use_checkpoint
        self.num_scale = num_scale
        self.ratios = ratios
        if with_fpn and patch_size == 16:
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                nn.BatchNorm2d(embed_dim),
                nn.GELU(),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )

            self.fpn2 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )

            self.fpn3 = nn.Identity()

            self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            logger = get_root_logger()
            logger.info('Build model without FPN.')

#         if self.with_simple_fpn:
#             self.mappers = nn.ModuleList([])
#             self.norms = nn.ModuleList([])
#             for factor in self.ratios:
#                 self.mappers.append(
#                     nn.Linear(embed_dim, last_feat_dim, bias=True)
#                 )
#                 self.norms.append(nn.LayerNorm(last_feat_dim, eps=1e-6))
#         if with_simple_fpn and patch_size == 16:
#             self.simple_fpns = nn.ModuleList([])
#             self.norms = nn.ModuleList([])
#             for factor in self.ratios:
#                 self.simple_fpns.append(
#                     ConvModule(embed_dim, 
#                            last_feat_dim, 
#                            kernel_size=3, 
#                            padding=1)
#                 )
#                 self.norms.append(nn.LayerNorm(last_feat_dim, eps=1e-6))

        # 点监督token
        self.last_feat = last_feat
        self.dual_depth = dual_depth
        self.point_tokens_num = point_tokens_num
        self.point_token = nn.Parameter(torch.zeros(1, point_tokens_num, embed_dim))
        self.point_pos_embed = nn.Parameter(torch.zeros(1, point_tokens_num, embed_dim))
        self.return_attention = return_attention
        
        # output norm
        if last_feat:
            self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
            self.decoder_embed = nn.Linear(embed_dim, last_feat_dim, bias=True)
        
        trunc_normal_(self.point_token, std=.02)
        trunc_normal_(self.point_pos_embed, std=.02)
        
        if self.dual_depth > 0:
            self.point_branch = nn.ModuleList([
                copy.deepcopy(block) for block in self.blocks[-self.dual_depth:]
            ])
        
    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(VisionTransformer, self).train(mode)
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            self.cls_token.requires_grad = False
            self.pos_embed.requires_grad = False
            self.pos_drop.eval()

        for i in range(1, self.frozen_stages + 1):
            if i  == len(self.blocks):
                norm_layer = getattr(self, 'norm') #f'norm{i-1}')
                norm_layer.eval()
                for param in norm_layer.parameters():
                    param.requires_grad = False

            m = self.blocks[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self):
        logger = get_root_logger()
        if self.init_cfg is None:
            raise ValueError(f'No pre-trained weights for {self.__class__.__name__}')
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            pretrained = self.init_cfg['checkpoint']
            if  os.path.isfile(pretrained):
                if self.dual_depth == 0:
                    load_checkpoint(self, pretrained, strict=False, logger=logger)
                else:
                    checkpoint = _load_checkpoint(pretrained, map_location='cpu')
                    if 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    elif 'model' in checkpoint:
                        state_dict = checkpoint['model']
                    else:
                        state_dict = checkpoint
                    # TODO: match the decoder blocks, norm and head in the state_dict due to the different prefix
                    new_state_dict = state_dict.copy()
                    for k, v in state_dict.items():
                        if k.startswith('blocks'):
                            ks = k.split('.')
                            index = int(ks[1])
                            if index >= 12 - self.dual_depth:
                                new_index = index - (12 - self.dual_depth)
                                new_k = 'point_branch.{:d}.'.format(new_index)
                                for s in ks[2:]:
                                    new_k += s + '.'
                                new_k = new_k[:-1]
                                new_state_dict[new_k] = v
                    load_state_dict(self, new_state_dict, strict=False, logger=logger)
            else:
                raise ValueError(f"checkpoint path {pretrained} is invalid")

    def prepare_tokens(self, x, mask=None):
        B, nc, w, h = x.shape
        # patch linear embedding
        x = self.patch_embed(x)

        # mask image modeling
        if mask is not None:
            x = self.mask_model(x, mask)
        x = x.flatten(2).transpose(1, 2)

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)
        
#         # 增加point token
        if self.dual_depth == 0:
            point_tokens = self.point_token.expand(B, -1, -1)
            point_pos_embed = self.point_pos_embed.expand(B, -1, -1)
            point_tokens = point_tokens + point_pos_embed
            x = torch.cat((x, point_tokens), dim=1)
        return self.pos_drop(x)

    def forward(self, x):
        B, _, H, W = x.shape
        Hp, Wp = H // self.patch_size, W // self.patch_size
        x = self.prepare_tokens(x)
        
        features = []
        point_attns = []
        # reverse_features = []
        
#         if self.num_scale is not None:
#             self.index = torch.arange(12).to(x.device)[-self.num_scale:]
            
        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                if self.return_attention:
#                     x, _ = checkpoint.checkpoint(blk, x)
                    x, attn = checkpoint.checkpoint(blk, x)
                    if self.dual_depth == 0:
                        point_attns.append(attn.mean(1))
                else:
                    x = checkpoint.checkpoint(blk, x)
            else:
                if self.return_attention:
#                     x, _ = blk(x)
                    x, attn = blk(x)
                    if self.dual_depth == 0:
                        point_attns.append(attn.mean(1))
                else:
                    x = blk(x)
                    
            if i == len(self.blocks) - 1 - self.dual_depth and self.dual_depth != 0: # 12 - 1 - 8 = i == 3
                point_tokens = self.point_token.expand(B, -1, -1)
                point_pos_embed = self.point_pos_embed.expand(B, -1, -1)
                point_tokens = point_tokens + point_pos_embed
                x_for_point = x.clone()
                x_for_point = torch.cat([x_for_point, point_tokens], dim=1)
                
                for blk_ in self.point_branch:
                    if self.use_checkpoint:
                        if self.return_attention:
                            x_for_point, attn = checkpoint.checkpoint(blk_, x_for_point)
                            point_attns.append(attn.mean(1))
                        else:
                            x_for_point = checkpoint.checkpoint(blk_, x_for_point)
                    else:
                        if self.return_attention:
                            x_for_point, attn = blk_(x_for_point)
                            point_attns.append(attn.mean(1))
                        else:
                            x_for_point = blk_(x_for_point)
                
            if i in self.out_indices:
                if self.dual_depth == 0:
                    xp = x[:, 1:-self.point_tokens_num, :].permute(0, 2, 1).reshape(B, -1, Hp, Wp)
                else:
                    xp = x[:, 1:, :].permute(0, 2, 1).reshape(B, -1, Hp, Wp)
                features.append(xp.contiguous())
                
        if self.with_fpn:
            ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
            for i in range(len(features)):
                features[i] = ops[i](features[i])
        
        # ops = [self.reverse_fpn1, self.reverse_fpn2, self.reverse_fpn3, self.reverse_fpn4]
        # for i in range(len(reverse_features)):
        #     reverse_features[i] = ops[i](reverse_features[i])
        
        
        if self.dual_depth == 0:
            point_tokens = self.norm(x[:, -self.point_tokens_num:])
        else:
            point_tokens = self.norm(x_for_point[:, -self.point_tokens_num:])
            
        if self.last_feat:
#             last_feat = self.decoder_embed(self.norm(x[:, 1:-self.point_tokens_num, :]))
            if self.dual_depth == 0:
                last_feat = self.decoder_embed(self.norm(x[:, 1:-self.point_tokens_num, :]))
            else:
                last_feat = self.decoder_embed(self.norm(x[:, 1:, :]))
                
        scale_features = []
        if self.with_simple_fpn:
#             xp = x[:, 1:-self.point_tokens_num, :].permute(0, 2, 1).reshape(B, -1, Hp, Wp)
            # xp = last_feat.permute(0, 2, 1).reshape(B, -1, Hp, Wp)
            xp = x[:, 1:-self.point_tokens_num, :].permute(0, 2, 1).reshape(B, -1, Hp, Wp)
            for i_lvl, scale_factor in enumerate(self.ratios):
                feat = F.interpolate(xp, scale_factor=scale_factor, mode='bicubic', align_corners=True)
                h, w = feat.size(-2), feat.size(-1)
                # feat = self.norms[i_lvl](feat.reshape(B, -1, h * w).transpose(2, 1)).transpose(2, 1).reshape(B, -1, h, w)
#                 feat = self.norms[i_lvl](self.mappers[i_lvl](feat.flatten(-2).permute(0, 2, 1))).reshape(B, -1, h, w)
                scale_features.append(feat)
        if self.return_attention and self.last_feat:
            return tuple(features), last_feat, point_tokens, torch.stack(point_attns).detach(), tuple(scale_features), x[:, 1:-self.point_tokens_num, :]
        else:
            assert False, 'no implemention'
