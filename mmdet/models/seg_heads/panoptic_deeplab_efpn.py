# ------------------------------------------------------------------------------
# Panoptic-DeepLab decoder.
# Written by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from collections import OrderedDict
from functools import partial
import math
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


class SeparableConvBlock(nn.Module):
    """
    created by Zylo117
    """

    def __init__(self, in_channels, out_channels=None, norm=True, activation=False, onnx_export=False):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        # Q: whether separate conv
        #  share bias between depthwise_conv and pointwise_conv
        #  or just pointwise_conv apply bias.
        # A: Confirmed, just pointwise_conv applies bias, depthwise_conv has no bias.

        self.depthwise_conv = Conv2dStaticSamePadding(in_channels, in_channels,
                                                      kernel_size=3, stride=1, groups=in_channels, bias=False)
        self.pointwise_conv = Conv2dStaticSamePadding(in_channels, out_channels, kernel_size=1, stride=1)

        self.norm = norm
        if self.norm:
            # Warning: pytorch momentum is different from tensorflow's, momentum_pytorch = 1 - momentum_tensorflow
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)

        self.activation = activation
        if self.activation:
            self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.bn(x)

        if self.activation:
            x = self.swish(x)

        return x
    
class Conv2dStaticSamePadding(nn.Module):
    """
    created by Zylo117
    The real keras/tensorflow conv2d with same padding
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, groups=1, dilation=1, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=bias, groups=groups)
        self.stride = self.conv.stride
        self.kernel_size = self.conv.kernel_size
        self.dilation = self.conv.dilation

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        h, w = x.shape[-2:]
        
        extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]
        
        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        x = F.pad(x, [left, right, top, bottom])

        x = self.conv(x)
        return x


@HEADS.register_module()
class EFPN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_levels
                ):
        super(EFPN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels            
        self.num_levels = num_levels
        
# up_small, low_large
        self.upsample1 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(out_channels, out_channels, 1),
            )
        self.upsample2 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
            )
        self.upsample3 = nn.Sequential(
            nn.Identity(),
        )
        self.upsample4 = nn.Sequential(
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        
#         self.pooling1_2 = nn.MaxPool2d(3, stride=2, padding=1)
#         self.fuse_conv1_2_down = SeparableConvBlock(out_channels)
#         self.sampler2_1 = nn.Upsample(scale_factor=2, mode='nearest')
#         self.fuse_conv1_2_up = SeparableConvBlock(out_channels)
        
        
#         self.pooling2_3_1 = nn.MaxPool2d(3, stride=2, padding=1)
#         self.fuse_conv2_3_down_1 = SeparableConvBlock(out_channels)
#         self.sampler3_2_1 = nn.Upsample(scale_factor=2, mode='nearest')
#         self.fuse_conv2_3_up_1 = SeparableConvBlock(out_channels)
#         self.pooling2_3_2 = nn.MaxPool2d(3, stride=2, padding=1)
#         self.fuse_conv2_3_down_2 = SeparableConvBlock(out_channels)
#         self.sampler3_2_2 = nn.Upsample(scale_factor=2, mode='nearest')
#         self.fuse_conv2_3_up_2 = SeparableConvBlock(out_channels)      
        
        
#         self.pooling3_4_1 = nn.MaxPool2d(3, stride=2, padding=1)
#         self.fuse_conv3_4_down_1 = SeparableConvBlock(out_channels)
#         self.sampler4_3_1 = nn.Upsample(scale_factor=2, mode='nearest')
#         self.fuse_conv3_4_up_1 = SeparableConvBlock(out_channels)
        
#         self.pooling3_4_2 = nn.MaxPool2d(3, stride=2, padding=1)
#         self.fuse_conv3_4_down_2 = SeparableConvBlock(out_channels)
#         self.sampler4_3_2 = nn.Upsample(scale_factor=2, mode='nearest')
#         self.fuse_conv3_4_up_2 = SeparableConvBlock(out_channels)
        
#         self.pooling3_4_3 = nn.MaxPool2d(3, stride=2, padding=1)
#         self.fuse_conv3_4_down_3 = SeparableConvBlock(out_channels)
#         self.sampler4_3_3 = nn.Upsample(scale_factor=2, mode='nearest')
#         self.fuse_conv3_4_up_3 = SeparableConvBlock(out_channels)
        
        
        
#         self.sampler1 = nn.Sequential(
#             nn.MaxPool2d(3, stride=2, padding=1),
#             nn.Conv2d(out_channels, out_channels, 1),
#             nn.BatchNorm2d(out_channels),
#             nn.GELU(),
#             nn.Conv2d(out_channels, out_channels, 1),
#         )
#         self.sampler2 = nn.MaxPool2d(3, stride=2, padding=1)
#         self.sampler3 = nn.Upsample(scale_factor=2, mode='nearest')
#         self.sampler4 = nn.Upsample(scale_factor=2, mode='nearest')

#         self.fuse_conv2_1 = SeparableConvBlock(out_channels)
#         self.fuse_conv3_2 = SeparableConvBlock(out_channels)
#         self.fuse_conv4_3 = SeparableConvBlock(out_channels)
# #         self.fuse_conv4 = SeparableConvBlock(out_channels)

#         self.fuse_conv1 = SeparableConvBlock(out_channels)
#         self.fuse_conv2 = SeparableConvBlock(out_channels)
#         self.fuse_conv3 = SeparableConvBlock(out_channels)
#         self.fuse_conv4 = SeparableConvBlock(out_channels)
        
#         self.activation = nn.GELU()

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
                
    def forward(self, inputs):
        assert len(inputs) == self.num_levels
#         patch_h, patch_w = inputs[0].size(-2), inputs[0].size(-1)
        
#         feat1_in = inputs[0]
#         feat2_in = inputs[1]
#         feat3_in = inputs[2]
#         feat4_in = inputs[3]
        
#         feat_out1 = self.upsample1(feat1_in) # 　第3层特征 resize 后
        
#         feat2 = self.upsample2(feat2_in) #　　第5层特征
#         feat2_mid = self.fuse_conv1_2_down(self.activation(self.pooling1_2(feat_out1) + feat2))
#         feat_out2 = self.fuse_conv1_2_up(self.activation(self.sampler2_1(feat2_mid) + feat_out1))
        
        
        
#         feat3 = self.upsample3(feat3_in) #　　第5层特征
#         feat3_mid_1 = self.fuse_conv2_3_down_1(self.activation(self.pooling2_3_1(feat2_mid) + feat3))
#         feat_out3_1 = self.fuse_conv2_3_up_1(self.activation(self.sampler3_2_1(feat3_mid_1) + feat2_mid))
#         feat3_mid_2 = self.fuse_conv2_3_down_2(self.activation(self.pooling2_3_2(feat_out2) + feat_out3_1))
#         feat_out3_2 = self.fuse_conv2_3_up_2(self.activation(self.sampler3_2_2(feat3_mid_2) + feat_out2))
        

#         feat4 = self.upsample4(feat4_in) #　　第5层特征
#         feat4_mid_1 = self.fuse_conv3_4_down_1(self.activation(self.pooling3_4_1(feat3_mid_1) + feat4))
#         feat_out4_1 = self.fuse_conv3_4_up_1(self.activation(self.sampler4_3_1(feat4_mid_1) + feat3_mid_1))
#         feat4_mid_2 = self.fuse_conv3_4_down_2(self.activation(self.pooling3_4_2(feat3_mid_2) + feat_out4_1))
#         feat_out4_2 = self.fuse_conv3_4_up_2(self.activation(self.sampler4_3_2(feat4_mid_2) + feat3_mid_2))
#         feat4_mid_3 = self.fuse_conv3_4_down_3(self.activation(self.pooling3_4_3(feat_out3_2) + feat_out4_2))
#         feat_out4_3 = self.fuse_conv3_4_up_3(self.activation(self.sampler4_3_2(feat4_mid_3) + feat_out3_2))
        
#         outputs = [feat_out1, feat_out2, feat_out3_2, feat_out4_3]
#         return outputs
        
#         feat3 = self.upsample3(feat3_in) #     7
#         feat4 = self.upsample4(feat4_in) #    11
        
#         feat3 = self.fuse_conv4_3(self.activation(self.up4_3(feat4) + feat3))
#         feat2 = self.fuse_conv3_2(self.activation(self.up3_2(feat3) + feat2))
#         feat1 = self.fuse_conv2_1(self.activation(self.up2_1(feat2) + feat1))
        
#         feat1_in = F.interpolate(feat1_in, size=(patch_h * 4, patch_w * 4), mode='bilinear')
#         feat2_in = F.interpolate(feat2_in, size=(patch_h * 2, patch_w * 2), mode='bilinear')
# #         feat3_in = F.interpolate(feat3_in, size=(patch_h, patch_w), mode='bilinear')
# #         feat4_in = F.interpolate(feat4_in, size=(patch_h // 2, patch_w // 2), mode='bilinear')

#         feat3 = F.interpolate(feat3, size=(patch_h, patch_w), mode='nearest')
#         feat4 = F.interpolate(feat4, size=(patch_h, patch_w), mode='nearest')
        
#         print(feat1_in.size(), feat2_in.size(), feat3_in.size(), feat4_in.size(), )
#         print(feat1.size(), feat2.size(), feat3.size(), feat4.size(), )
#         exit()
#         feat1 = self.fuse_conv1(self.activation(feat1_in + feat1))
#         feat2 = self.fuse_conv2(self.activation(feat2_in + feat2))
#         feat3 = self.fuse_conv3(self.activation(feat3_in + feat3))
#         feat4 = self.fuse_conv4(self.activation(feat4_in + feat4))

        feat1 = self.upsample1(inputs[0]) 
        feat2 = self.upsample2(inputs[1]) 
        feat3 = self.upsample3(inputs[2]) 
        feat4 = self.upsample4(inputs[3])
        
        outputs = [feat1, feat2, feat3, feat4]
        return outputs

@HEADS.register_module()
class SinglePanopticDeepLabHeadEFPN(nn.Module):
    def __init__(self, decoder_channels, head_channels, num_classes, class_key):
        super(SinglePanopticDeepLabHeadEFPN, self).__init__()
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
class PanopticDeepLabDecoderEFPN(nn.Module):
    def __init__(self,
                instance_head=dict(
                    type="SinglePanopticDeepLabHeadEFPN",
                    decoder_channels=256,
                    head_channels=(32),
                    num_classes=(2),
                    class_key=["offset"]),
                loss_pred=dict(type='L1Loss', loss_weight=0.05)):
        
        super(PanopticDeepLabDecoderEFPN, self).__init__()
        # self.instance_decoder = build_head(instance_decoder)
        self.instance_head = build_head(instance_head)
        self.loss_pred = build_loss(loss_pred)
                
    def forward_train(self, features, patch_size=None): #, targets_maps, weights=None):
        # temp = torch.stack(targets_maps)
        patch_h, patch_w = patch_size[0], patch_size[1]
        img_h, img_w = patch_h * 16, patch_w * 16
        features = torch.stack(features) # num_scale, bs, channel, h/4, w/4
        # fused_feature = self.instance_decoder(features)
        pred_maps_ = self.instance_head(features)
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
        
    # def set_image_pooling(self, pool_size):
    #     self.instance_decoder.set_image_pooling(pool_size)
        
    def init_weights(self):
        # self.instance_decoder.init_weights()
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