_base_ = [
    '../_base_/datasets/voc0712point.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
INF = 1e8
pretrained = '../pretrain/mae_vit_small_800e.pth'
num_scale = 4
model = dict(
    type='LSS',
    backbone=dict(
        type='DualVisionTransformer',
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
        point_tokens_num=100,
        return_attention=True,
        point_token_attn_mask=False,
        dual_depth=0,
        num_scale=num_scale,
        last_feat=True,
        last_feat_dim=256,
        with_simple_fpn=True,
        ratios=[2, 1, 0.5, 0.25, 0.125]
    ),
    neck=[
    dict(
        type='FPN',
        in_channels=[384, 384, 384, 384],
        out_channels=256,
        num_outs=5),
    ],
    lss_head=dict(
        type='LatentScaleSelectionHead',
        instance_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[16]),
        point_head=dict(
            type='MLPHead',
            in_channels=384,
            num_classes=20,
            cls_mlp_depth=3,
            reg_mlp_depth=3,
            loss_point=dict(type='L1Loss', loss_weight=10.0),
            loss_cls=dict(type='FocalLoss', 
                          use_sigmoid=True, 
                          gamma=2.0, 
                          alpha=0.25, 
                          loss_weight=1.0)
        ),
        mil_head=dict(
            type='MILHead',
            in_channels=256,
            hidden_channels=1024,
            pooling_type='roi',
            roi_size=7,
            num_classes=20,
            loss_mil=dict(
                type='MILLoss', 
#                 use_sigmoid=True, # BCE loss 
                use_sigmoid=False, # gfocal
                reduction='mean',
                loss_weight=1.0))
    ),
    scale_head=None,
#     scale_head=dict(
# #         _delete_=True,
#         type='ScaleFreeAnchorRetinaHead',
#         num_classes=20,
#         in_channels=256,
#         stacked_convs=4,
#         feat_channels=256,
#         pre_anchor_topk=50,
#         num_proposals_per_gt=4,
#         anchor_generator=dict(
#             type='AnchorGenerator',
#             octave_base_scale=4,
#             scales_per_octave=3,
#             ratios=[0.5, 1.0, 2.0],
#             strides=[8, 16, 32, 64, 128]),
#         # anchor_generator=dict(
#         #     type='AnchorGenerator',
#         #     octave_base_scale=8,
#         #     scales_per_octave=1,
#         #     ratios=[1.0],
#         #     strides=[8, 16, 32, 64, 128]),
#         bbox_coder=dict(
#             type='DeltaXYWHBBoxCoder',
#             target_means=[.0, .0, .0, .0],
#             target_stds=[0.1, 0.1, 0.2, 0.2]),
#         loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=0.75),
#         loss_mil=dict(
#                 type='MILLoss', 
#                 use_sigmoid=True, # BCE loss 
#                 # use_sigmoid=False, # gfocal_loss 
#                 reduction='mean',
#                 loss_weight=0.0),
#         loss_cls=dict(type='FocalLoss', 
#                     use_sigmoid=True, 
#                     gamma=2.0, 
#                     alpha=0.25, 
#                     loss_weight=0.0)
#     ),
    decoder_head=None,
    offset_head=None,
    snake_head=None,
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_skip_fpn=True,
    test_wo_detector=False,
    test_on_fcos=True,
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[16]),
        bbox_head=dict(
            type='MAEBoxHead',
            init_cfg=dict(type='Pretrained', checkpoint=pretrained),
            use_checkpoint=False,
            in_channels=256,
            img_size=224,
            patch_size=16, 
            embed_dim=256, 
            depth=4,
            num_heads=8, 
            mlp_ratio=4., 
            num_classes=20,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            reg_decoded_bbox=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='GIoULoss', loss_weight=10.0),
        )
    ),
    train_cfg=dict(
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
                scale=4, # when "scale_method" is "auto" or "single" this item is available
                scale_method='scale', # "single" or "auto" or "average"
            ),
            merge_method='top_merge',
            erode=False,
            topk=1,
            topk_=4,
            cam_thr=0.2,
            area_ratio=0.5,
            box_method='expand', # min-max, shift
            point_method='gt_center', # pred, gt_coarse,
        ),
        scale_head=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.4,
                min_pos_iou=0,
                ignore_iof_thr=-1),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
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
                scale=4, # when "scale_method" is "auto" or "single" this item is available
                scale_method='scale', # "single" or "auto" or "average"
            ),
            merge_method='top_merge',
            erode=False,
            topk=1,
            topk_=4,
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
            max_per_img=100)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    )
)
# augmentation strategy originates from DETR / Sparse RCNN
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_point=True),
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
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_points']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_point=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1000, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_points']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(pipeline=train_pipeline),
    test=dict(pipeline=test_pipeline),
)
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
# runner = dict(type='EpochBasedRunner', max_epochs=12)
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