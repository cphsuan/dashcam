# model settings
model = dict(
    type='FasterRCNN', #模型類型
    backbone=dict(
        type='ResNet', #類型
        depth=50, #網路層數
        num_stages=4, #resnet的stage数量
        out_indices=(0, 1, 2, 3), #输出的stage序號
        frozen_stages=1, #凍結的stage數量，即該stage不更新參數，-1表示所有的stage都更新參數
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch', # 如果設置pytorch，則stride為2的層是conv3x3的卷積層；如果設置caffe，则stride為2的層是第一個conv1x1的卷積層
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048], #輸入的各個stage的通道數
        out_channels=256, #輸出的特徵層的通道數
        num_outs=5), #輸出的數量
    rpn_head=dict(
        type='RPNHead',
        in_channels=256, #輸入通道數
        feat_channels=256, #特徵層通道數
        anchor_generator=dict( #生成的anchor
            type='AnchorGenerator',
            scales=[8], #=baselen = sqrt(w*h)
            ratios=[0.5, 1.0, 2.0], # anchor's w and h ratio 
            strides=[4, 8, 16, 32, 64]), #每個特徵層上的anchor步長
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0], #均值
            target_stds=[1.0, 1.0, 1.0, 1.0]), #標準差
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0), #是否使用sigmoid来进行分類，如果False则使用softmax来分類
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead', #全連接層類型
            in_channels=256, #輸入通道數
            fc_out_channels=1024, #輸出通道數
            roi_feat_size=7, #roi特徵尺寸
            num_classes=6, #80
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False, # class_agnostic表示輸出bbox時只考慮是否為前景，後續分類的時候再根据该bbox在網路中的類別得分來分類，一個框可以對應多個類別
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner', #RPN網路的正負樣本劃分
                pos_iou_thr=0.7, #正樣本的iou閾值
                neg_iou_thr=0.3, #負樣本的iou閾值
                min_pos_iou=0.3, ##正樣本的iou最小值 如果assign给ground truth的anchors中最大的IOU低於0.3，则忽略所有的anchors，否則保留最大IOU的anchor
                match_low_quality=True,
                ignore_iof_thr=-1), #忽略bbox的閾值
            sampler=dict(
                type='RandomSampler', #正負樣本提取器類型
                num=256, #需提取正負樣本的數量
                pos_fraction=0.5, #正負樣本的比例
                neg_pos_ub=-1,
                add_gt_as_proposals=False), #把ground truth加入proposal作為幀樣本
            allowed_border=-1,
            pos_weight=-1, #正樣本權重，-1表示不改變原始權重
            debug=False),
        rpn_proposal=dict( #推斷時的RPN參數
            nms_pre=2000, #在所有的fpn層內做nms
            max_per_img=1000, #max_per_img表示最终输出的det bbox数量
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
    ))
