_base_ = [
    './cascade_mask_rcnn_x101_64x4d_fpn_mstrain_3x_coco.py'
]

#python tools/train.py configs/cascade_rcnn/cocoCascade.py
#pip install -v -e .


# dataset
data_root = ./segmentation'
dataset_type = 'CocoInstance'
#classes = ('ribs', 'sternum','gristle')
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=12,
    train=dict(
    dataset=dict(type=dataset_type,
        img_prefix= data_root + '/train2017/',
        #classes=classes,
        ann_file=data_root + '/annotations/train2017.json')
),
    val=dict(
        type=dataset_type,
        img_prefix=data_root + '/val2017/',
        #classes=classes,
        ann_file=data_root + '/annotations/val2017.json'),
    test=dict(
        type=dataset_type,
        img_prefix=data_root + '/val2017/',
        #classes=classes,
        ann_file=data_root + '/annotations/val2017.json'))

# pre-training
load_from = './cascade_mask_rcnn_x101_64x4d_fpn_mstrain_3x_coco_20210719_210311-d3e64ba0.pth'
