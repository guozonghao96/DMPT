# dataset settings
dataset_type = 'VOCDatasetInstance'
# dataset_type = 'VOCPointDataset'
# data_root = 'data/VOCdevkit/'
# data_root = '/home/YaoYuan/Dataset/VOCdevkit/'
data_root = '/home/GuoZonghao/Dataset/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
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
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1000, 600),
        # img_scale=(3000, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            # dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
            # dict(type='Collect', keys=['img']),
        ])
]

data_train_12 = dict(
    type='RepeatDataset',
    times=3,
    dataset=dict(
        type='VOCDatasetInstance',
        # ann_subdir='Annotations-QC',
        ann_file=data_root + 'VOC2012/ImageSets/Segmentation/train_aug_12.txt',
        img_prefix=data_root + 'VOC2012/',
        pipeline=train_pipeline
    )
)
data_train_sbd= dict(
    type='RepeatDataset',
    times=3,
    dataset=dict(
        type='SBDDatasetInstance',
        # ann_subdir='Annotations-QC',
        ann_file=data_root + 'VOC2012/ImageSets/Segmentation/train_aug_sbd.txt',
        img_prefix=data_root + 'VOC2012/',
        pipeline=train_pipeline
    )
)

data_test_12 = dict(
    type='VOCDatasetInstance',
    test_mode=False,
    # ann_subdir='Annotations-QC',
    ann_file=data_root + 'VOC2012/ImageSets/Segmentation/train_aug_12.txt',
    img_prefix=data_root + 'VOC2012/',
    pipeline=test_pipeline)

data_test_sbd= dict(
    type='SBDDatasetInstance',
    test_mode=False,
    ann_file=data_root + 'VOC2012/ImageSets/Segmentation/train_aug_sbd.txt',
    img_prefix=data_root + 'VOC2012/',
    pipeline=test_pipeline
)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=[data_train_12, data_train_sbd],
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2012/ImageSets/Segmentation/val.txt',
        img_prefix=data_root + 'VOC2012/',
        pipeline=test_pipeline),
    test=[data_test_12, data_test_sbd]
)

evaluation = dict(interval=1, metric='mAP_Segm')