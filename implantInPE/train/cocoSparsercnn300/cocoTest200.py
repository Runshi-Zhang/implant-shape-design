_base_ = './sparse_rcnn_r101_fpn_300_proposals_mstrain_480-800_3x_coco.py'

#python tools/train.py configs/cocoSparsercnn300/cocoTest200.py

# dataset
data_root = './objectdetectiondata'
dataset_type = 'CocoDataset'
classes = ('0', '1')
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=24,
    train=dict(
        type=dataset_type,
        img_prefix= data_root + '/train/',
        classes=classes,
        ann_file=data_root + '/annotations/train.json'),
    val=dict(
        type=dataset_type,
        img_prefix=data_root + '/val/',
        classes=classes,
        ann_file=data_root + '/annotations/val.json'),
    test=dict(
        type=dataset_type,
        img_prefix=data_root + '/val/',
        classes=classes,
        ann_file=data_root + '/annotations/val.json'))

#pre-training
load_from = './sparse_rcnn_r101_fpn_mstrain_480-800_3x_coco_20201223_121552-6c46c9d6.pth'
