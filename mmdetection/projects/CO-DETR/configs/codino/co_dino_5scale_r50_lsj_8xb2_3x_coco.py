_base_ = ['co_dino_5scale_r50_lsj_8xb2_1x_coco.py']

param_scheduler = [dict(type='MultiStepLR',milestones=[30])]
train_cfg = dict(max_epochs=36)
seed = 42
gpu_ids = [0]
checkpoint_config = dict(max_keep_ckpts=3, interval=1)
work_dir = './work_dirs/faster_rcnn_r50_fpn_1x_trash'