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
class SinglePanopticDeepLabDecoderSame(nn.Module):
    def __init__(self, in_channels, decoder_channels, atrous_rates, num_scale=None):
        super(SinglePanopticDeepLabDecoderSame, self).__init__()
        
        aspp_modules = []
        for _ in range(num_scale):
            aspp_modules.append(
                ASPP(in_channels, out_channels=decoder_channels, atrous_rates=atrous_rates)
            )
        self.aspp_modules = nn.ModuleList(aspp_modules)

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
        num_scale = features.size(0)
        outputs = []
        for i_scale in range(num_scale):
            feat = self.aspp_modules[i_scale](features[i_scale])
            outputs.append(feat)
        return outputs 

@HEADS.register_module()
class SinglePanopticDeepLabHeadSame(nn.Module):
    def __init__(self, decoder_channels, head_channels, num_classes, class_key):
        super(SinglePanopticDeepLabHeadSame, self).__init__()
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
        for key, feat in zip(self.class_key, x):
            pred[key] = self.classifier[key](feat)
        return pred


@HEADS.register_module()
class PanopticDeepLabDecoderSame(nn.Module):
    def __init__(self, 
                 instance_decoder=dict(
                    type='SinglePanopticDeepLabDecoderSame',
                    in_channels=384,
                    decoder_channels=128, 
                    atrous_rates=(3, 6, 9)
                 ), 
                instance_head=dict(
                    type="SinglePanopticDeepLabHeadSame",
                    decoder_channels=128,
                    head_channels=(32),
                    num_classes=(2),
                    class_key=["offset"]),
                loss_pred=dict(type='L1Loss', loss_weight=0.05)):
        
        super(PanopticDeepLabDecoderSame, self).__init__()
        self.instance_decoder = build_head(instance_decoder)
        self.instance_head = build_head(instance_head)
        self.loss_pred = build_loss(loss_pred)
                
    def forward_train(self, features, patch_size=None): #, targets_maps, weights=None):
        # temp = torch.stack(targets_maps)
        patch_h, patch_w = patch_size[0], patch_size[1]
        img_h, img_w = patch_h * 16, patch_w * 16
        features = torch.stack(features)
        num_scale, num_imgs, channel, patch_h, patch_w  = features.size()
        features = features.reshape(-1, channel, patch_h, patch_w)
        features = F.interpolate(features, size=(patch_h * 4, patch_w * 4), mode='bilinear')
        features = features.reshape(num_scale, num_imgs, channel, patch_h * 4, patch_w * 4)
        fused_feature = self.instance_decoder(features)
        pred_maps_ = self.instance_head(fused_feature)
        pred_maps = []
        for key in pred_maps_.keys():
            temp_maps = F.interpolate(pred_maps_[key], 
                                      size=(img_h, img_w), 
                                      mode='bilinear')
            pred_maps.append(temp_maps)
        pred_maps = torch.stack(pred_maps).permute(1, 0, 2, 3, 4)
        # loss_pred = self.loss(pred_maps, 
        #                         targets_maps,
        #                         weights,
        #                         style=key[:-1])
        # losses = dict()
        # losses.update(loss_pred)
        # return pred_maps, losses
        return pred_maps
        
    def set_image_pooling(self, pool_size):
        self.instance_decoder.set_image_pooling(pool_size)
        
    def init_weights(self):
        self.instance_decoder.init_weights()
        self.instance_head.init_weights()
        
    @force_fp32(apply_to=('pred_maps'))
    def loss(self,
             pred_maps, # batch, 2, H, W
             targets_maps,
             weights,
             style=None,
            ):
        if style == 'offset':
            weights = torch.stack(weights) # batch, num_scale, H, W
            targets_maps = torch.stack(targets_maps) # batch, num_scale, 2, H, W
            num_scale = weights.size(1)
            # weights = weights.unsqueeze(1).repeat(1, 2, 1, 1)
            # pred_maps = pred_maps.flatten(1)
            # targets_maps = targets_maps.flatten(1)
            # weights = weights.flatten(1)

            weights = weights.unsqueeze(2)                   # batch, num_scale, 1, H, W
            targets_maps = targets_maps                  # batch, num_scale 2, H, W

            # pred_maps = torch.flip(pred_maps, dims=[1]) # 大目标学上面的，小目标学下面的
            #                                                           # 好像没用，但是可能有insight
            losses = dict()
            losses['loss_offset'] = self.loss_pred(pred_maps, 
                                                   targets_maps,
                                                   weights)
            return losses
        elif style == 'seg':
            weights = torch.stack(weights) # batch, num_scale, H, W
            batch, num_scale, img_h, img_w = weights.size()
            
            targets_maps = torch.stack(targets_maps) # batch, num_scale, 2, H, W
            # weights = weights.unsqueeze(1).repeat(1, 2, 1, 1)
            # pred_maps = pred_maps.flatten(1)
            # targets_maps = targets_maps.flatten(1)
            # weights = weights.flatten(1)

            weights = weights.unsqueeze(2)                   # batch, num_scale, 1, H, W
            targets_maps = targets_maps                  # batch, num_scale 2, H, W
            
            # pred_maps batch, num_scale, channel, img_h, img_w
            cls_channel = pred_maps.size(2)
            # pred_maps = torch.flip(pred_maps, dims=[1]) # 大目标学上面的，小目标学下面的
            #                                                           # 好像没用，但是可能有insight
            losses = dict()
            losses['loss_seg'] = self.loss_pred(pred_maps.reshape(-1, cls_channel, img_h, img_w), 
                                                   targets_maps.reshape(-1, img_h, img_w),
                                                   weights)
            return losses