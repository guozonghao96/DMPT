_base_ = [
    '../_base_/datasets/voc2012aug.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
pretrained = '../pretrain/mae_vit_small_800e.pth'
model = dict(
    type='PointFormer',
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
        use_checkpoint=True,
        with_fpn=False,
        with_simple_fpn=True,
        last_feat=False,
        ratios=[4, 2, 1, 0.5] # 16
    ),
    # neck=None,
    neck=dict(
        type='ChannelMapper',
        in_channels=[384, 384, 384, 384],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    point_head=dict(
       type='PointFormerHead',
        num_query=300,
        num_neg_points=96,
        num_classes=20,
        in_channels=256,
        sync_cls_avg_factor=True,
        as_two_stage=True,
        with_box_refine=True,
        num_keypoints=8,
        foreground_topk=12,
        num_outs=3,
        init_point_mil_head=dict(
            type='PointMILHead',
            in_channels=256,
            num_classes=20,
            cls_threshold=0.1,
            neg_cls_threshold=0.2,
            neg_loss_weight=0.75,
            pos_gt_weight=0.125,
            loss_mil=dict(
                type='MILLoss', 
                use_sigmoid=False, # Focal loss 
                reduction='None',
                loss_weight=0.25),
        ),
        transformer=dict(
            type='PointTransformer',
            num_feature_levels=3,
            two_stage_num_proposals=300,
            num_keypoints=8,
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        num_levels=3,
                        num_points=4,
                        embed_dims=256),
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='PointDeformableDetrTransformerDecoder',
                num_layers=3,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='MultiScaleDeformableAttention',
                            decoder=True,
                            refine=False,
                            embed_dims=256,
                            # num_levels=1,
                            num_levels=3,
                            num_points=4,
                        )
                    ],
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, 
                      gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_point=dict(type='L1Loss', loss_weight=10.0),
        loss_mask=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='NewDiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0),
    ),
    train_cfg=dict(
        assigner=dict(
            type='HungarianPointAssigner',
            cls_cost=dict(type='FocalLossCost', weight=1.0),
            reg_cost=dict(type='PointL1Cost', weight=10.0)
        ),
        sampler=dict(type='PointPseudoSampler'),
        pos_weight=1),
    test_cfg=dict(
        lss=dict(
            assigner=dict(
                type='HungarianPointAssigner',
                cls_cost=dict(type='FocalLossCost', weight=1.0),
                reg_cost=dict(type='PointL1Cost', weight=10.0)
            ),
            sampler=dict(type='PointPseudoSampler'),
            pos_weight=1,
            # pooling_method
            pooling=dict(
                type='roi',
                filter=False,
                discard=0.9,
                multiple=True,
                scale=7, # when "scale_method" is "auto" or "single" this item is available
                scale_method='auto', # "single" or "auto" or "average"
            ),
            merge_method='top_merge',
            erode=False,
            topk=1,
            topk_=1,
            cam_thr=0.2,
            area_ratio=0.5,
            box_method='expand', # min-max, shift
            point_method='gt_center', # pred, gt_coarse,
        ),
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    )
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='AutoAugment',
         policies=[
             [
                 dict(type='Resize',
                      img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                                 (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                                 (736, 1333), (768, 1333), (800, 1333)],
                      multiscale_mode='value',
                      keep_ratio=True)
             ],
             [
                 dict(type='Resize',
                      img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                      multiscale_mode='value',
                      keep_ratio=True),
                 dict(type='RandomCrop',
                      crop_type='absolute_range',
                      crop_size=(384, 600),
                      allow_negative_crop=True),
                 dict(type='Resize',
                      img_scale=[(480, 1333), (512, 1333), (544, 1333),
                                 (576, 1333), (608, 1333), (640, 1333),
                                 (672, 1333), (704, 1333), (736, 1333),
                                 (768, 1333), (800, 1333)],
                      multiscale_mode='value',
                      override=True,
                      keep_ratio=True)
             ]
         ]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1000, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# data = dict(
#     samples_per_gpu=2,
#     workers_per_gpu=2,
#     train=dict(pipeline=train_pipeline),
#     test=dict(pipeline=test_pipeline),
# )
# optimizer
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.05,
    constructor='LayerDecayOptimizerConstructor', 
    paramwise_cfg=dict(
        num_layers=12, 
        layer_decay_rate=0.75,
        custom_keys={
            # 'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)}
    ))
# optimizer_config = dict(_delete_=True,
#                         grad_clip=dict(max_norm=0.01, norm_type=2))
# learning policy
# lr_config = dict(policy='step', step=[40])
# runner = dict(type='EpochBasedRunner', max_epochs=50)

# learning policy
lr_config = dict(policy='step', step=[9, 11])
# runner = dict(type='EpochBasedRunnerAmp', max_epochs=12)
runner = dict(type='EpochBasedRunner', max_epochs=12)
# fp16 = None
# optimizer_config = dict(
#     type="DistOptimizerHook",
#     update_interval=1,
#     grad_clip=None,
#     coalesce=True,
#     bucket_size_mb=-1,
#     # use_fp16=True,
#     use_fp16=False,
# )
# find_unused_parameters=True