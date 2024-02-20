import os
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from functools import partial
from collections import OrderedDict

from mmcv.runner import _load_checkpoint, load_state_dict
from mmdet.utils import get_root_logger
from mmdet.models.builder import HEADS
from .bbox_head import BBoxHead
from models.vision_transformer import Block, trunc_normal_
from ...utils.positional_encoding import get_2d_sincos_pos_embed
from mmdet.models.builder import HEADS, build_loss
from mmcv.runner import auto_fp16, force_fp32

from torch.nn import functional as F

@HEADS.register_module()
class SnakeDecoderHead(BBoxHead):
    def __init__(self,
                 in_channels,
                 img_size=224,
                 patch_size=16, 
                 embed_dim=256, 
                 depth=4,
                 num_heads=8, 
                 mlp_ratio=4., 
                 qkv_bias=True, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 use_checkpoint=False,
                 init_cfg=None,
                 num_points=32,
                 loss_energy=dict(
                     type='SnakeEnergyLoss',
                     alpha=0.05,
                     beta=0.5,
                     gamma=1.0,
                     reduction='mean', 
                     loss_weight=1.0),
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patch_size = patch_size
        self.use_checkpoint = use_checkpoint
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        num_patches = (img_size // patch_size) ** 2
        self.num_patches = num_patches
        # self.det_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.init_cfg = init_cfg
        self.with_decoder_embed = False
        # MAE decoder specifics
        if in_channels != embed_dim:
            self.with_decoder_embed = True
            self.norm = norm_layer(in_channels)
            self.decoder_embed = nn.Linear(in_channels, embed_dim, bias=True)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.decoder_blocks = nn.ModuleList([
            Block(
                embed_dim, num_heads, mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)
        ])
        # self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=True)  # fixed sin-cos 
        self.circle_decoder_pos_embed = nn.Parameter(torch.zeros(1, num_points, embed_dim), requires_grad=True)  # 因为是圈特征，因此需要重新学习position embedding
        self.decoder_box_norm = norm_layer(embed_dim)
        self.snake_offset = nn.Linear(embed_dim, 2)
        self.loss_energy = build_loss(loss_energy)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        # decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.num_patches**.5), cls_token=True)
        # self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        # 

    def init_weights(self):
        logger = get_root_logger()
        if self.init_cfg is None:
            self.apply(self._init_weights)
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            pretrained = self.init_cfg['checkpoint']
            if os.path.isfile(pretrained):
                logger.info('loading checkpoint for {}'.format(self.__class__))
                checkpoint = _load_checkpoint(pretrained, map_location='cpu')
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint
                # TODO: match the decoder blocks, norm and head in the state_dict due to the different prefix
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    if k.startswith('patch_embed') or k.startswith('blocks'):
                        continue
                    elif k in ['pos_embed']:
                        continue
                    else:
                        new_state_dict[k] = v
                load_state_dict(self, new_state_dict, strict=False, logger=logger)
            else:
                raise ValueError(f"checkpoint path {pretrained} is invalid")
    
    def forward(self, x):
        circle_decoder_pos_embed = self.circle_decoder_pos_embed
        x = x + circle_decoder_pos_embed
        for blk in self.decoder_blocks:
            if self.use_checkpoint:
                x = checkpoint(blk, x)
            else:
                x = blk(x)
        x = self.decoder_box_norm(x)
        snake_offsets = self.snake_offset(x)
        return snake_offsets # offset 的 尺度偏移应该参考哪个呢？

    @force_fp32(apply_to=('decoded_snakes'))
    def loss(self,
             decoded_snakes,
             pred_offset_map,
             gt_points,
             snake_targets):
        losses = dict()        
        pred_offset_map = pred_offset_map.detach() # 应该是detach掉梯度谱 [batch, 2, h, w]
        loss_energy = 0
        for snakes, energy_map, points, snakes_t in zip(decoded_snakes, 
                                                        pred_offset_map, 
                                                        gt_points,
                                                        snake_targets):
            loss_energy += self.loss_energy(snakes, 
                                            energy_map,
                                            points,
                                            snakes_t)
        losses['loss_energy'] = loss_energy
        return losses