# implant-shape-design
An end-to-end framework for intelligent diagnosis and outputting surgical plans for pectus excavatum in minimally invasive repair of pectus excavatum .

Code and data for paper: Automatic implant shape design for minimally invasive repair of pectus excavatum using deep learning and shape registration

Our code is based on the <a href="https://download.openmmlab.com/mmdetection" rel="nofollow">mmdetection</a> of openmmlab.

#Requirements
Some important required packages include:

Linux

Pytorch version >=1.5.

Python 3.6+

mmdetection 2.28.1

Some basic python packages such as Numpy,pydicom,scipy,pycpd...

#Object Detection Module

The pre-training model is <a href="https://download.openmmlab.com/mmdetection/v2.0/sparse_rcnn/sparse_rcnn_r101_fpn_mstrain_480-800_3x_coco/sparse_rcnn_r101_fpn_mstrain_480-800_3x_coco_20201223_121552-6c46c9d6.pth" rel="nofollow">Sparse R-CNN</a>.

The network of the Object Detection Module is <a href="https://pan.baidu.com/s/1cPepyFn7vTfCzsEcxaLWrA">Sparse R-CNN-R101 with 200 proposal boxes</a>.

#Instance Segmentation Module

The pre-training model is <a href="https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_r101_fpn_mstrain_3x_coco/cascade_mask_rcnn_r101_fpn_mstrain_3x_coco_20210628_165236-51a2d363.pth" rel="nofollow">Cascade Mask R-CNN</a>.

#Requirements
