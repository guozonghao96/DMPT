_base_ = [
    '../_base_/datasets/voc2012aug.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
decoder_mask_loss_weight = 1.0
decoder_bbox_loss_weight = 10.0
decoder_cls_loss_weight = 1.0
INF = 1e8
custom_hooks = [
   dict(type='LossWeightAdjustHook', 
        start_epoch=1, 
        decoder_mask_loss_weight=decoder_mask_loss_weight,
        decoder_bbox_loss_weight=decoder_bbox_loss_weight,
        decoder_cls_loss_weight=decoder_cls_loss_weight,
        priority='NORMAL') 
]
pretrained = '../pretrain/mae_vit_small_800e.pth'
num_scale = 4
model = dict(
    type='PointDetr',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    lss_head=dict(
        type='AttnLatentScaleSelectionHead',
        instance_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[16]),
        point_head=dict(
            type='PDETRHead',
            num_classes=20,
            in_channels=1024,
            # in_channels=256,
            transformer=dict(
                type='Transformer',
                debug=True,
                encoder=dict(
                    type='DetrTransformerEncoder',
                    num_layers=6,
                    transformerlayers=dict(
                        type='BaseTransformerLayer',
                        attn_cfgs=[
                            dict(
                                type='MultiheadAttention',
                                embed_dims=256,
                                num_heads=8,
                                dropout=0.1)
                        ],
                        feedforward_channels=2048,
                        ffn_dropout=0.1,
                        operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
                decoder=dict(
                    type='DetrTransformerDecoder',
                    return_intermediate=True,
                    num_layers=6,
                    transformerlayers=dict(
                        type='DetrTransformerDecoderLayer',
                        attn_cfgs=dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        feedforward_channels=2048,
                        ffn_dropout=0.1,
                        operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                         'ffn', 'norm')),
                )),
            positional_encoding=dict(
                type='SinePositionalEncoding', num_feats=128, normalize=True),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_point=dict(type='L1Loss', loss_weight=10.0)),
        mil_head=dict(
            type='MILHead',
            in_channels=256,
            hidden_channels=1024,
            pooling_type='roi',
            roi_size=7,
            num_classes=20,
            loss_mil=dict(
                type='MILLoss', 
                use_sigmoid=True, # BCE loss 
                # use_sigmoid=False, # gfocal
                reduction='mean',
                loss_weight=1.0))
    ),
    scale_head=dict(
        type='AttentionShift',
        # iam 参数
        iam_num_points_init=5,
        iam_thr_pos=0.35, 
        iam_thr_neg=0.8,
        iam_refine_times=2, 
        iam_obj_tau=0.9,
        # iam 参数
        num_query=100,
        point_feat_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[16]),
#         with_gt_points=False,
        discriminate_loss_weight=0.2,
        with_gt_points=True,
        num_classes=20,
        in_channels=256,
        sync_cls_avg_factor=True,
        as_two_stage=False,
        with_box_refine=True,
        num_keypoints=6,
        transform_method='minmax'),
    scale_head1=None,
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
    roi_skip_fpn=False,
    test_wo_detector=False,
    test_on_fcos=False,
    roi_head=dict(
        type='PointStandardRoIHead',
        # type='MAEDecoderHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=20,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=decoder_cls_loss_weight),
            loss_bbox=dict(type='L1Loss', loss_weight=decoder_bbox_loss_weight)), 
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead_',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=20,
            loss_mask=dict(
                type='MaskCrossEntropyLoss', use_mask=True,
                loss_weight=decoder_mask_loss_weight))),
            # loss_mask=dict(
            #     type='CrossEntropyLoss', use_mask=True, loss_weight=decoder_mask_loss_weight))),
    train_cfg=dict(
        lss=dict(
            assigner=dict(
                type='HungarianPointAssigner',
                cls_cost=dict(type='FocalLossCost', weight=1.),
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
                scale=6, # when "scale_method" is "auto" or "single" this item is available
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
        scale_head=dict(
            assigner=dict(
                type='HungarianKeyPointAssigner',
                cls_cost=dict(type='FocalLossCost', weight=1.0),
                reg_cost=dict(type='KeyPointL1Cost', weight=10.0),
                # bbox_reg_cost=dict(type='BBoxL1Cost', weight=2.5),
                # bbox_iou_cost=dict(type='IoUCost', weight=1.0)
                bbox_reg_cost=dict(type='BBoxL1Cost', weight=0.0),
                bbox_iou_cost=dict(type='IoUCost', weight=0.0)
            ),
            sampler=dict(type='PointPseudoSampler'),
        ),
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
    # betas=(0.9, 0.999),
    # weight_decay=0.05,
    weight_decay=0.0001,
    # constructor='LayerDecayOptimizerConstructor', 
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}))
# learning policy
lr_config = dict(policy='step', step=[9, 11])
# runner = dict(type='EpochBasedRunnerAmp', max_epochs=12)
runner = dict(type='EpochBasedRunner', max_epochs=12)
# do not use mmdet version fp16
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