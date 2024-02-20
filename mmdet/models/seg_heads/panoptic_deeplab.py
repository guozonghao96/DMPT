# ------------------------------------------------------------------------------
# Panoptic-DeepLab decoder.
# Written by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from collections import OrderedDict
from functools import partial

from ..builder import HEADS, build_head
from mmdet.models.builder import HEADS, build_loss
import torch
from torch import nn
from torch.nn import functional as F

from ..builder import build_backbone, build_head, build_neck
from mmcv.runner import auto_fp16, force_fp32


def basic_conv(in_planes, out_planes, kernel_size, stride=1, padding=1, groups=1,
               with_bn=True, with_relu=True):
    """convolution with bn and relu"""
    module = []
    has_bias = not with_bn
    module.append(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                  bias=has_bias)
    )
    if with_bn:
        module.append(nn.BatchNorm2d(out_planes))
    if with_relu:
        module.append(nn.ReLU())
    return nn.Sequential(*module)


def depthwise_separable_conv(in_planes, out_planes, kernel_size, stride=1, padding=1, groups=1,
                             with_bn=True, with_relu=True):
    """depthwise separable convolution with bn and relu"""
    del groups

    module = []
    module.extend([
        basic_conv(in_planes, in_planes, kernel_size, stride, padding, groups=in_planes,
                   with_bn=True, with_relu=True),
        nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
    ])
    if with_bn:
        module.append(nn.BatchNorm2d(out_planes))
    if with_relu:
        module.append(nn.ReLU())
    return nn.Sequential(*module)


def stacked_conv(in_planes, out_planes, kernel_size, num_stack, stride=1, padding=1, groups=1,
                 with_bn=True, with_relu=True, conv_type='basic_conv'):
    """stacked convolution with bn and relu"""
    if num_stack < 1:
        assert ValueError('`num_stack` has to be a positive integer.')
    if conv_type == 'basic_conv':
        conv = partial(basic_conv, out_planes=out_planes, kernel_size=kernel_size, stride=stride,
                       padding=padding, groups=groups, with_bn=with_bn, with_relu=with_relu)
    elif conv_type == 'depthwise_separable_conv':
        conv = partial(depthwise_separable_conv, out_planes=out_planes, kernel_size=kernel_size, stride=stride,
                       padding=padding, groups=1, with_bn=with_bn, with_relu=with_relu)
    else:
        raise ValueError('Unknown conv_type: {}'.format(conv_type))
    module = []
    module.append(conv(in_planes=in_planes))
    for n in range(1, num_stack):
        module.append(conv(in_planes=out_planes))
    return nn.Sequential(*module)

################################################################################################

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.aspp_pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.ReLU()
        )

    def set_image_pooling(self, pool_size=None):
        if pool_size is None:
            self.aspp_pooling[0] = nn.AdaptiveAvgPool2d(1)
        else:
            self.aspp_pooling[0] = nn.AvgPool2d(kernel_size=pool_size, stride=1)

    def forward(self, x):
        size = x.shape[-2:]
        x = self.aspp_pooling(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=True)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super(ASPP, self).__init__()
        # out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def set_image_pooling(self, pool_size):
        self.convs[-1].set_image_pooling(pool_size)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


@HEADS.register_module()
class SinglePanopticDeepLabDecoder(nn.Module):
    def __init__(self, in_channels, feature_key, low_level_channels, low_level_key, low_level_channels_project,
                 decoder_channels, atrous_rates, aspp_channels=None):
        super(SinglePanopticDeepLabDecoder, self).__init__()
        if aspp_channels is None:
            aspp_channels = decoder_channels
        self.aspp = ASPP(in_channels, out_channels=aspp_channels, atrous_rates=atrous_rates)
        self.feature_key = feature_key
        self.decoder_stage = len(low_level_channels)
        assert self.decoder_stage == len(low_level_key)
        assert self.decoder_stage == len(low_level_channels_project)
        self.low_level_key = low_level_key
        fuse_conv = partial(stacked_conv, kernel_size=5, num_stack=1, padding=2,
                            conv_type='depthwise_separable_conv')

        # Transform low-level feature
        project = []
        # Fuse
        fuse = []
        # Top-down direction, i.e. starting from largest stride
        for i in range(self.decoder_stage):
            project.append(
                nn.Sequential(
                    nn.Conv2d(low_level_channels[i], low_level_channels_project[i], 1, bias=False),
                    nn.BatchNorm2d(low_level_channels_project[i]),
                    nn.ReLU()
                )
            )
            if i == 0:
                fuse_in_channels = aspp_channels + low_level_channels_project[i]
            else:
                fuse_in_channels = decoder_channels + low_level_channels_project[i]
            fuse.append(
                fuse_conv(
                    fuse_in_channels,
                    decoder_channels,
                )
            )
        self.project = nn.ModuleList(project)
        self.fuse = nn.ModuleList(fuse)

    def init_weights(self):
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean=0.0, std=0.001)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = 0.01
            
    def set_image_pooling(self, pool_size):
        self.aspp.set_image_pooling(pool_size)

    def forward(self, features):
        x = features[self.feature_key]
        x = self.aspp(x)
        # build decoder
        for i in range(self.decoder_stage):
            l = features[self.low_level_key[i]]
            l = self.project[i](l)
            x = F.interpolate(x, size=l.size()[2:], mode='bilinear', align_corners=True)
            x = torch.cat((x, l), dim=1)
            x = self.fuse[i](x)
        return x


@HEADS.register_module()
class SinglePanopticDeepLabHead(nn.Module):
    def __init__(self, decoder_channels, head_channels, num_classes, class_key):
        super(SinglePanopticDeepLabHead, self).__init__()
        fuse_conv = partial(stacked_conv, kernel_size=5, num_stack=1, padding=2,
                            conv_type='depthwise_separable_conv')

        self.num_head = len(num_classes)
        assert self.num_head == len(class_key)

        classifier = {}
        for i in range(self.num_head):
            classifier[class_key[i]] = nn.Sequential(
                fuse_conv(
                    decoder_channels,
                    head_channels[i],
                ),
                nn.Conv2d(head_channels[i], num_classes[i], 1)
            )
        self.classifier = nn.ModuleDict(classifier)
        self.class_key = class_key
        
    def init_weights(self):
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean=0.0, std=0.001)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = 0.01
            
    def forward(self, x):
        pred = OrderedDict()
        # build classifier
        for key in self.class_key:
            pred[key] = self.classifier[key](x)
        return pred


@HEADS.register_module()
class PanopticDeepLabDecoder(nn.Module):
    def __init__(self, 
                 instance_decoder=dict(
                    type='SinglePanopticDeepLabDecoder',
                    in_channels=384,
                    feature_key="res5",
                    low_level_channels=(384, 384, 384), 
                    low_level_key=["res4", "res3", "res2"], 
                    low_level_channels_project=(64, 32, 16),
                    decoder_channels=128, 
                    atrous_rates=(3, 6, 9),
                    aspp_channels=256), 
                instance_head=dict(
                    type="SinglePanopticDeepLabHead",
                    decoder_channels=128,
                    head_channels=(32),
                    num_classes=(2),
                    class_key=["offset"]),
                loss_offset=dict(type='L1Loss', loss_weight=1.0)):
        
        super(PanopticDeepLabDecoder, self).__init__()
        self.instance_decoder = build_head(instance_decoder)
        self.instance_head = build_head(instance_head)
        self.loss_offset = build_loss(loss_offset)
                
    def forward_train(self, features, offset_targets, weights=None):
        _, _, patch_h, patch_w = features[2].shape
        # forward 
        features_ = dict()
        features_['res2'] = features[0]
        features_['res3'] = features[1]
        features_['res4'] = features[2]
        features_['res5'] = features[3]
        
        fused_feature = self.instance_decoder(features_)
        pred_offset_maps = self.instance_head(fused_feature)['offset']
        # upsample to original size
        pred_offset_maps = F.interpolate(pred_offset_maps, 
                                         size=(patch_h * 16, patch_w * 16), 
                                         mode='bilinear')
        loss_offset = self.loss(pred_offset_maps, 
                                offset_targets,
                                weights)
        
        losses = dict()
        losses.update(loss_offset)
        
        return pred_offset_maps, losses
        
    def set_image_pooling(self, pool_size):
        self.instance_decoder.set_image_pooling(pool_size)
        
    def init_weights(self):
        self.instance_decoder.init_weights()
        self.instance_head.init_weights()
        
    @force_fp32(apply_to=('pred_offset_maps'))
    def loss(self,
             pred_offset_maps, # batch, 2, H, W
             offset_targets,
             weights):
        
        weights = torch.stack(weights) # batch, num_scale, H, W
        offset_targets = torch.stack(offset_targets) # batch, num_scale, 2, H, W
        num_scale = weights.size(1)
        # weights = weights.unsqueeze(1).repeat(1, 2, 1, 1)
        # pred_offset_maps = pred_offset_maps.flatten(1)
        # offset_targets = offset_targets.flatten(1)
        # weights = weights.flatten(1)
        
        pred_offset_maps = pred_offset_maps.unsqueeze(1).repeat(1, num_scale, 1, 1, 1) # batch, 1, 2, H, W
        weights = weights.unsqueeze(2)                   # batch, num_scale, 1, H, W
        offset_targets = offset_targets                  # batch, num_scale 2, H, W
        
        losses = dict()
        losses['loss_offset'] = self.loss_offset(pred_offset_maps, 
                                                 offset_targets,
                                                 weights)
        return losses