_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

seed = 2022
gpu_ids = [0]
checkpoint_config = dict(max_keep_ckpts=3, interval=1)
work_dir = './work_dirs/faster_rcnn_r50_fpn_1x_trash'