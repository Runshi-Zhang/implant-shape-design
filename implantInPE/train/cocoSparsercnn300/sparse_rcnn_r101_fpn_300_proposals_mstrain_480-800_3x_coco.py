_base_ = './sparse_rcnn_r101_fpn_mstrain_480-800_3x_coco.py'
num_proposals = 200
model = dict(
    rpn_head=dict(num_proposals=num_proposals),
    test_cfg=dict(
        _delete_=True, rpn=None, rcnn=dict(max_per_img=num_proposals)))

