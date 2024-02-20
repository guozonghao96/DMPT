_base_ = [
    '../_base_/datasets/voc0712.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

pretrained = '../pretrain/mae_vit_small_800e.pth'

# model settings
model = dict(
    type='FCOS',
    backbone=dict(
        type='VisionTransformer',
        init_cfg=dict(type='Pretrained', checkpoint=pretrained),
        img_size=224,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_path_rate=0.1,
        out_indices=(3, 5, 7, 11),
        learnable_pos_embed=True,
        use_checkpoint=True,
        with_fpn=True,
        last_feat=False,
        last_feat_dim=256,
    ),
    neck=dict(
        type='FPN',
        in_channels=[384, 384, 384, 384],
        out_channels=256,
        num_outs=5),
    bbox_head=dict(
        type='FCOSHead',
        num_classes=20,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True, with_point=True),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(type='AutoAugment',
#          policies=[
#              [
#                  dict(type='Resize',
#                       img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
#                                  (608, 1333), (640, 1333), (672, 1333), (704, 1333),
#                                  (736, 1333), (768, 1333), (800, 1333)],
#                       multiscale_mode='value',
#                       keep_ratio=True)
#              ],
#              [
#                  dict(type='Resize',
#                       img_scale=[(400, 1333), (500, 1333), (600, 1333)],
#                       multiscale_mode='value',
#                       keep_ratio=True),
#                  dict(type='RandomCrop',
#                       crop_type='absolute_range',
#                       crop_size=(384, 600),
#                       allow_negative_crop=True),
#                  dict(type='Resize',
#                       img_scale=[(480, 1333), (512, 1333), (544, 1333),
#                                  (576, 1333), (608, 1333), (640, 1333),
#                                  (672, 1333), (704, 1333), (736, 1333),
#                                  (768, 1333), (800, 1333)],
#                       multiscale_mode='value',
#                       override=True,
#                       keep_ratio=True)
#              ]
#          ]),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size_divisor=32),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_points']),
# ]

# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True, with_point=True),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(1000, 600),
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='Pad', size_divisor=32),
#             dict(type='DefaultFormatBundle'),
#             dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_points']),
#         ])
# ]

# data = dict(
#     samples_per_gpu=2,
#     workers_per_gpu=2,
#     train=dict(pipeline=train_pipeline),
#     val=dict(pipeline=test_pipeline),
#     test=dict(pipeline=test_pipeline))
# optimizer

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    constructor='LayerDecayOptimizerConstructor', 
                 paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.75))
# learning policy
lr_config = dict(policy='step', step=[9, 11])
runner = dict(type='EpochBasedRunnerAmp', max_epochs=12)
# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)
# find_unused_parameters=True


# optimizer = dict(
#     lr=0.01, paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
# optimizer_config = dict(
#     _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
# lr_config = dict(
#     policy='step',
#     warmup='constant',
#     warmup_iters=500,
#     warmup_ratio=1.0 / 3,
#     step=[8, 11])
# runner = dict(type='EpochBasedRunner', max_epochs=12)

