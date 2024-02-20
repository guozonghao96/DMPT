# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.utils import Registry, build_from_cfg

TRANSFORMER = Registry('Transformer')
LINEAR_LAYERS = Registry('linear layers')


def build_transformer(cfg, default_args=None):
    """Builder for Transformer."""
    return build_from_cfg(cfg, TRANSFORMER, default_args)


LINEAR_LAYERS.register_module('Linear', module=nn.Linear)


def build_linear_layer(cfg, *args, **kwargs):
    """Build linear layer.
    Args:
        cfg (None or dict): The linear layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate an linear layer.
        args (argument list): Arguments passed to the `__init__`
            method of the corresponding linear layer.
        kwargs (keyword arguments): Keyword arguments passed to the `__init__`
            method of the corresponding linear layer.
    Returns:
        nn.Module: Created linear layer.
    """
    if cfg is None:
        cfg_ = dict(type='Linear')
    else:
        if not isinstance(cfg, dict):
            raise TypeError('cfg must be a dict')
        if 'type' not in cfg:
            raise KeyError('the cfg dict must contain the key "type"')
        cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in LINEAR_LAYERS:
        raise KeyError(f'Unrecognized linear type {layer_type}')
    else:
        linear_layer = LINEAR_LAYERS.get(layer_type)

    layer = linear_layer(*args, **kwargs, **cfg_)

    return layer


# Copyright (c) Hikvision Research Institute. All rights reserved.
from mmcv.utils import Registry, build_from_cfg
from mmcv.cnn.bricks.transformer import ATTENTION as MMCV_ATTENTION
from mmcv.cnn.bricks.transformer import POSITIONAL_ENCODING as \
    MMCV_POSITIONAL_ENCODING
from mmcv.cnn.bricks.transformer import TRANSFORMER_LAYER_SEQUENCE as \
    MMCV_TRANSFORMER_LAYER_SEQUENCE
from mmdet.models.utils.builder import TRANSFORMER as MMDET_TRANSFORMER


ATTENTION = Registry('attention', parent=MMCV_ATTENTION)
POSITIONAL_ENCODING = Registry('Position encoding',
                               parent=MMCV_POSITIONAL_ENCODING)
TRANSFORMER_LAYER_SEQUENCE = Registry('transformer-layers sequence',
                                      parent=MMCV_TRANSFORMER_LAYER_SEQUENCE)
TRANSFORMER = Registry('Transformer', parent=MMDET_TRANSFORMER)


# def build_attention(cfg, default_args=None):
#     """Builder for attention."""
#     return build_from_cfg(cfg, ATTENTION, default_args)


# def build_positional_encoding(cfg, default_args=None):
#     """Builder for Position Encoding."""
#     return build_from_cfg(cfg, POSITIONAL_ENCODING, default_args)


# def build_transformer_layer_sequence(cfg, default_args=None):
#     """Builder for transformer encoder and transformer decoder."""
#     return build_from_cfg(cfg, TRANSFORMER_LAYER_SEQUENCE, default_args)


# def build_transformer(cfg, default_args=None):
#     """Builder for Transformer."""
#     return build_from_cfg(cfg, TRANSFORMER, default_args)


