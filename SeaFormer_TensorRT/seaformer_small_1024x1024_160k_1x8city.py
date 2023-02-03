log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(
    type='AdamW',
    lr=0.0003,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            head=dict(lr_mult=10.0),
            norm=dict(decay_mult=0.0))))
optimizer_config = dict()
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=160000)
checkpoint_config = dict(by_epoch=False, interval=4000)
evaluation = dict(interval=4000, metric='mIoU', pre_eval=True)
norm_cfg = dict(type='SyncBN', requires_grad=True)
model_cfgs = dict(
    cfg1=[[3, 1, 16, 1], [3, 4, 24, 2], [3, 3, 24, 1]],
    cfg2=[[5, 3, 48, 2], [5, 3, 48, 1]],
    cfg3=[[3, 3, 96, 2], [3, 3, 96, 1]],
    cfg4=[[5, 4, 160, 2]],
    cfg5=[[3, 6, 192, 2]],
    channels=[16, 24, 48, 96, 160, 192],
    depths=[3, 3],
    key_dims=[16, 24],
    emb_dims=[160, 192],
    num_heads=6,
    drop_path_rate=0.1)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='SeaFormer',
        cfgs=[[[3, 1, 16, 1], [3, 4, 24, 2], [3, 3, 24, 1]],
              [[5, 3, 48, 2], [5, 3, 48, 1]], [[3, 3, 96, 2], [3, 3, 96, 1]],
              [[5, 4, 160, 2]], [[3, 6, 192, 2]]],
        channels=[16, 24, 48, 96, 160, 192],
        emb_dims=[160, 192],
        key_dims=[16, 24],
        depths=[3, 3],
        num_heads=6,
        drop_path_rate=0.1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='modelzoos/classification/SeaFormer_S.pth')),
    decode_head=dict(
        type='LightHead',
        in_channels=[48, 160, 192],
        in_index=[0, 1, 2],
        channels=128,
        dropout_ratio=0.1,
        embed_dims=[96, 128],
        num_classes=19,
        is_dw=True,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
dataset_type = 'CityscapesDataset'
data_root = '/dockerdata/CityScapes/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (1024, 1024)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(1024, 1024), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(1024, 1024), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=500,
        dataset=dict(
            type='CityscapesDataset',
            data_root='/dockerdata/CityScapes/',
            img_dir='leftImg8bit/train',
            ann_dir='gtFine/train',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations'),
                dict(
                    type='Resize',
                    img_scale=(2048, 1024),
                    ratio_range=(0.5, 2.0)),
                dict(
                    type='RandomCrop',
                    crop_size=(1024, 1024),
                    cat_max_ratio=0.75),
                dict(type='RandomFlip', prob=0.5),
                dict(type='PhotoMetricDistortion'),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(
                    type='Pad', size=(1024, 1024), pad_val=0, seg_pad_val=255),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_semantic_seg'])
            ])),
    val=dict(
        type='CityscapesDataset',
        data_root='/dockerdata/CityScapes/',
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 1024),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CityscapesDataset',
        data_root='/dockerdata/CityScapes/',
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 1024),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
find_unused_parameters = True

