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
class PointVisionTransformer(VisionTransformer):
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
                 return_attention=True,
                 last_feat=True,
                 last_feat_dim=256,
                 init_cfg=None,
                 num_classes=20,
                 **kwargs):
        super(PointVisionTransformer, self).__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim, 
            return_attention=return_attention,
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
        self.frozen_stages = frozen_stages
        self.out_indices = out_indices
        self.use_checkpoint = use_checkpoint
        self.num_classes = num_classes
        
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

        # 点监督token
        self.last_feat = last_feat
        self.return_attention = return_attention
        self.cate_token = nn.Parameter(torch.zeros(1, num_classes, embed_dim))
        trunc_normal_(self.cate_token, std=.02)
        
        # output norm
        if last_feat:
            self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
            self.decoder_embed = nn.Linear(embed_dim, last_feat_dim, bias=True)
        
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
                    load_checkpoint(self, pretrained, strict=False, logger=logger)
            else:
                raise ValueError(f"checkpoint path {pretrained} is invalid")

    def prepare_tokens(self, x, mask=None):
        B, nc, w, h = x.shape
        # patch linear embedding
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # add positional encoding to each token
        pos_embeddings = self.interpolate_pos_encoding(x, w, h)
        x = x + pos_embeddings
        
        return self.pos_drop(x), pos_embeddings

    def forward(self, x, gt_points=None, gt_labels=None):
        B, C, H, W = x.shape
        assert B == 1, 'batch_size must be 1'
        
        Hp, Wp = H // self.patch_size, W // self.patch_size
        x, pos_embeddings = self.prepare_tokens(x)
        # upsample pos_embeddings to original image size
        pos_embeddings = pos_embeddings[:, 1:]
        pos_embeddings_maps = pos_embeddings.permute(0, 2, 1).reshape(1, -1, Hp, Wp)
        pos_embeddings_maps = F.interpolate(pos_embeddings_maps, 
                                            (H, W), 
                                            mode='bicubic').expand(B, -1, -1, -1)
        cate_tokens = self.cate_token.expand(B, -1, -1)
        if gt_points is not None and gt_labels is not None:
            num_gt = len(gt_points[0])
            point_tokens = []
            for points, labels, pos_maps, cate_token \
                    in zip(gt_points, gt_labels, pos_embeddings_maps, cate_tokens):
                cate_embed = cate_token[labels]
                index = points.long()
                pos_embed = pos_maps[:, index[:, 1], index[:, 0]].transpose(1, 0)
                point_tokens.append(cate_embed + pos_embed)
            point_tokens = torch.stack(point_tokens)
            x = torch.cat([x, point_tokens], dim=1)
            
        features = []
        point_attns = []
        
        for i, blk in enumerate(self.blocks): 
            if self.use_checkpoint:
                if self.return_attention:
                    x, attn = checkpoint.checkpoint(blk, x)
                    point_attns.append(attn.mean(1))
                else:
                    x = checkpoint.checkpoint(blk, x)
            else:
                if self.return_attention:
                    x, attn = blk(x)
                    point_attns.append(attn.mean(1))
                else:
                    x = blk(x)
                    
            if i in self.out_indices:
                xp = x[:, 1:-num_gt, :].permute(0, 2, 1).reshape(B, -1, Hp, Wp)
                features.append(xp.contiguous())
                
        if self.with_fpn:
            ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
            for i in range(len(features)):
                features[i] = ops[i](features[i])
        
        point_tokens = self.norm(x[:, -num_gt:])
            
        if self.last_feat:
            last_feat = self.decoder_embed(self.norm(x[:, 1:-num_gt, :]))

        if self.return_attention and self.last_feat:
            return tuple(features), last_feat, point_tokens, torch.stack(point_attns).detach()
        else:
            assert False, 'no implemention'
