#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# Lightning Trainer should be considered beta at this point
# We have confirmed that training and validation run correctly and produce correct results
# Depending on how you launch the trainer, there are issues with processes terminating correctly
# This module is still dependent on D2 logging, but could be transferred to use Lightning logging

import logging
import os
import datetime
import time
import copy
import wandb
import weakref
from collections import OrderedDict
from typing import Any, Dict, List
import pytorch_lightning as pl  # type: ignore
from pytorch_lightning import LightningDataModule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

import torch 

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedGroupKFold
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, build_detection_train_loader, detection_utils
from detectron2.engine import (
    DefaultTrainer,
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
)
from detectron2.evaluation import print_csv_format,COCOEvaluator, DatasetEvaluator
from detectron2.evaluation.testing import flatten_results_dict
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage
from detectron2.utils.logger import setup_logger
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2 import model_zoo
import detectron2.data.transforms as T



def build_evaluator(cfg, dataset_name, output_folder=None):
    if output_folder is None:
        os.makedirs('./output_eval', exist_ok = True)
        output_folder = './output_eval'
        
    return COCOEvaluator(dataset_name, cfg, False, output_folder)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("detectron2")


class TrainingModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        self.cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        self.storage: EventStorage = None
        self.model = build_model(self.cfg)

        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER

        self.val_loss = []

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["iteration"] = self.storage.iter

    def on_load_checkpoint(self, checkpointed_state: Dict[str, Any]) -> None:
        self.start_iter = checkpointed_state["iteration"]
        self.storage.iter = self.start_iter

    def setup(self, stage: str):
        if self.cfg.MODEL.WEIGHTS:
            self.checkpointer = DetectionCheckpointer(
                # Assume you want to save checkpoints together with logs/statistics
                self.model,
                self.cfg.OUTPUT_DIR,
            )
            logger.info(f"Load model weights from checkpoint: {self.cfg.MODEL.WEIGHTS}.")
            # Only load weights, use lightning checkpointing if you want to resume
            self.checkpointer.load(self.cfg.MODEL.WEIGHTS)

        self.iteration_timer = hooks.IterationTimer()
        self.iteration_timer.before_train()
        self.data_start = time.perf_counter()
        self.writers = None

    def training_step(self, batch, batch_idx):
        data_time = time.perf_counter() - self.data_start
        # Need to manually enter/exit since trainer may launch processes
        # This ideally belongs in setup, but setup seems to run before processes are spawned
        if self.storage is None:
            self.storage = EventStorage(0)
            self.storage.__enter__()
            self.iteration_timer.trainer = weakref.proxy(self)
            self.iteration_timer.before_step()
            self.writers = (
                default_writers(self.cfg.OUTPUT_DIR, self.max_iter)
                if comm.is_main_process()
                else {}
            )

        loss_dict = self.model(batch)
        # total_loss = sum(loss_dict.values())
        # self.train_loss.append(total_loss.item())
        SimpleTrainer.write_metrics(loss_dict, data_time,self.storage.iter) # ITER

        # WandB에 iteration, total_loss, 개별 손실 기록
        log_data = {
            "iteration": self.storage.iter,
            "total_loss": sum(loss_dict.values()).item()
        }
        # 개별 손실 추가
        for loss_name, loss_value in loss_dict.items():
            log_data[loss_name] = loss_value.item()
        
        wandb.log(log_data)

        opt = self.optimizers()
        self.storage.put_scalar(
            "lr",
            opt.param_groups[self._best_param_group_id]["lr"],
            smoothing_hint=False,
        )
        self.iteration_timer.after_step()
        self.storage.step()
        # A little odd to put before step here, but it's the best way to get a proper timing
        self.iteration_timer.before_step()

        if self.storage.iter % 20 == 0:
            for writer in self.writers:
                writer.write()
        return sum(loss_dict.values())

    def training_step_end(self, training_step_outpus):
        self.data_start = time.perf_counter()
        return training_step_outpus

    def training_epoch_end(self, training_step_outputs):
        # 에포크마다 손실, 학습률 등을 로깅
        # avg_train_loss = sum(self.train_loss) / len(self.train_loss) if self.train_loss else 0
        # wandb.log({"train_loss": avg_train_loss})
                
        # 저장한 학습 손실 목록 초기화
        # self.train_loss = []

        self.iteration_timer.after_train()
        if comm.is_main_process():
            self.checkpointer.save("model_final")
        for writer in self.writers:
            writer.write()
            writer.close()
        self.storage.__exit__(None, None, None)

    def _process_dataset_evaluation_results(self) -> OrderedDict:
        results = OrderedDict()
        for idx, dataset_name in enumerate(self.cfg.FOLD_VAL_NAME):
            results[dataset_name] = self._evaluators[idx].evaluate()
            if comm.is_main_process():
                print_csv_format(results[dataset_name])

        if len(results) == 1:
            results = list(results.values())[0]
        return results

    def _reset_dataset_evaluators(self):
        self._evaluators = []
        for dataset_name in self.cfg.FOLD_VAL_NAME:
            evaluator = build_evaluator(self.cfg, dataset_name)
            evaluator.reset()
            self._evaluators.append(evaluator)

    def on_validation_epoch_start(self):
        self._reset_dataset_evaluators()

    def validation_epoch_end(self, _outputs):
        results = self._process_dataset_evaluation_results()

        flattened_results = flatten_results_dict(results)

        # Validation 손실 평균 계산 및 로깅
        # avg_val_loss = sum(self.val_loss) / len(self.val_loss) if self.val_loss else 0
        # wandb.log({"avg_val_loss": avg_val_loss})

        # Validation 손실 초기화
        self.val_loss = []

        # Extract mAP, mAP50, and mAP75 and store them in instance variables
        if 'bbox/AP' in flattened_results:
            self.mAP = flattened_results['bbox/AP']
            self.log('mAP', self.mAP)  # Log for EarlyStopping and Checkpointing
        if 'bbox/AP50' in flattened_results:
            self.mAP50 = flattened_results['bbox/AP50']
            self.log('mAP50', self.mAP50)  # Log for EarlyStopping and Checkpointing
        if 'bbox/AP75' in flattened_results:
            self.mAP75 = flattened_results['bbox/AP75']
            self.log('mAP75', self.mAP75)  # Log for EarlyStopping and Checkpointing

        # avg_val_loss 계산 및 Wandb에 즉시 로그 기록
        # avg_val_loss = sum(self.val_loss) / len(self.val_loss) if self.val_loss else 0
        wandb.log({
            # "avg_val_loss": avg_val_loss,
            "mAP": self.mAP,
            "mAP50": self.mAP50,
            "mAP75": self.mAP75,
            "current_step": self.storage.iter,
            "learning_rate": self.optimizers().param_groups[0]["lr"]
        })

        for k, v in flattened_results.items():
            try:
                v = float(v)
            except Exception as e:
                raise ValueError(
                    "[EvalHook] eval_function should return a nested dict of float. "
                    "Got '{}: {}' instead.".format(k, v)
                ) from e
        self.storage.put_scalars(**flattened_results, smoothing_hint=False)

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        if not isinstance(batch, List):
            batch = [batch] 

         # 이미지 데이터를 torch.Tensor로 변환하고, 채널 순서 변경 (HWC -> CHW)
        # for x in batch:
            # x['image'] = torch.from_numpy(x['image'].astype('float32')).permute(2, 0, 1)

        outputs = self.model(batch)
        self._evaluators[dataloader_idx].process(batch, outputs)
        
    
    def on_predict_epoch_start(self):
        # return super().on_predict_epoch_start()
        self.model.eval()

    # def predict_step(self, batch, batch_idx):

    #     # test dataloder의 batch_size는 1
    #     # TODO : batch size 변경 
    #     prediction_string = ''
    #     data = batch
    #     data = data[0]
    #     data_file_name,data = data['file_name'],data['image']
    #     height, width = data.shape[:2]
        
    #     #TODO? 
    #     # if self.input_format == "RGB":
    #     #     # whether the model expects BGR inputs or RGB
    #     #     original_image = original_image[:, :, ::-1]


    #     data = self.aug.get_transform(data).apply_image(data)
    #     data = torch.as_tensor(data.astype("float32").transpose(2, 0, 1))
    #     data.to(self.cfg.MODEL.DEVICE)

    #     inputs = {"image": data, "height": height, "width": width}

    #     outputs = self.model([inputs])[0]['instances'] #==self.model at eval  

    #     targets = outputs.pred_classes.cpu().tolist()
    #     boxes = [i.cpu().detach().numpy() for i in outputs.pred_boxes]
    #     scores = outputs.scores.cpu().tolist()

    #     for target, box, score in zip(targets,boxes,scores):
    #         prediction_string += (str(target) + ' ' + str(score) + ' ' + str(box[0]) + ' ' 
    #         + str(box[1]) + ' ' + str(box[2]) + ' ' + str(box[3]) + ' ')

    #     return prediction_string, data_file_name
    
    def predict_step(self, batch, batch_idx):
        prediction_results = []
    
        for data in batch:
            prediction_string = ''
            data_file_name, data_image = data['file_name'], data['image']
            height, width = data_image.shape[:2]

            # 이미지 변환
            data_image = self.aug.get_transform(data_image).apply_image(data_image)
            data_image = torch.as_tensor(data_image.astype("float32").transpose(2, 0, 1))
            data_image = data_image.to(self.cfg.MODEL.DEVICE)

            inputs = {"image": data_image, "height": height, "width": width}
            outputs = self.model([inputs])[0]['instances']  # 예측

            # 예측 결과 추출
            targets = outputs.pred_classes.cpu().tolist()
            boxes = [i.cpu().detach().numpy() for i in outputs.pred_boxes]
            scores = outputs.scores.cpu().tolist()

            # 예측 문자열 생성
            for target, box, score in zip(targets, boxes, scores):
                prediction_string += (f"{target} {score} {box[0]} {box[1]} {box[2]} {box[3]} ")

            # 결과 저장
            prediction_results.append((prediction_string, data_file_name))
    
        return prediction_results


    def configure_optimizers(self):
        optimizer = build_optimizer(self.cfg, self.model)
        self._best_param_group_id = hooks.LRScheduler.get_best_param_group_id(optimizer)
        scheduler = build_lr_scheduler(self.cfg, optimizer)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        # 사용자 정의 스케줄러를 수동으로 업데이트합니다.
        scheduler.step()

##### Mapper 
def TrainMapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    image = detection_utils.read_image(dataset_dict['file_name'], format='BGR')
    
    transform_list = [
        T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
        T.RandomBrightness(0.8, 1.8),
        T.RandomContrast(0.6, 1.3)
    ]
    
    image, transforms = T.apply_transform_gens(transform_list, image)
    
    dataset_dict['image'] = torch.as_tensor(image.transpose(2,0,1).astype('float32')) # (H,W,C) -> (C,H,W)
    
    annos = [   # 변환된 이미지에 맞게 annotation 변경                
        detection_utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop('annotations')
        if obj.get('iscrowd', 0) == 0
    ]
    
    instances = detection_utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict['instances'] = detection_utils.filter_empty_instances(instances)
    
    return dataset_dict

def ValMapper(dataset_dict):
    
    dataset_dict = copy.deepcopy(dataset_dict)
    image = detection_utils.read_image(dataset_dict['file_name'], format='BGR')
    
    dataset_dict['image'] = torch.as_tensor(image.transpose(2,0,1).astype('float32')) # (H,W,C) -> (C,H,W)
    
    return dataset_dict

def TestMapper(dataset_dict):
    
    dataset_dict = copy.deepcopy(dataset_dict)
    image = detection_utils.read_image(dataset_dict['file_name'], format='BGR')
    
    dataset_dict['image'] = image
    
    return dataset_dict


class DataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        self.sampler = None #TODO?

        print(self.cfg.DATASETS)


    def train_dataloader(self):
        return build_detection_train_loader(self.cfg,mapper = TrainMapper,sampler=self.sampler)

    def val_dataloader(self):
        # dataloaders = []
        # for dataset_name in self.cfg.DATASETS.TEST: #TODO?
            # dataloaders.append(build_detection_test_loader(self.cfg, dataset_name,ValMapper))
        dataloaders = [build_detection_test_loader(self.cfg,fold_val_name,ValMapper)]
        return dataloaders
    
    def predict_dataloader(self):
        return build_detection_test_loader(self.cfg, 'coco_trash_test', TestMapper) #TODO? 


    


def main(args):

    wandb.init(project='trash object detection',entity='tayoung1005-aitech',config=args,reinit=True)

    global base_dir
    base_dir = '/home/minipin/cv-01/level2-objectdetection-cv-01/'
    # 학습 및 테스트 데이터셋 등록
    try:
        register_coco_instances('coco_trash_train', {}, base_dir + 'dataset/train.json', base_dir + 'dataset')
    except AssertionError:
        pass

    try:
        register_coco_instances('coco_trash_test', {}, base_dir + 'dataset/test.json', base_dir + 'dataset')
    except AssertionError:
        pass

    MetadataCatalog.get('coco_trash_train').thing_classes = ["General trash", "Paper", "Paper pack", "Metal", 
                                                        "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]
    
    # 원본 학습 데이터셋 로드
    original_dataset = DatasetCatalog.get('coco_trash_train')
    original_metadata = MetadataCatalog.get('coco_trash_train')
    

    # 원본 학습 데이터셋 불러오기
    import json
    with open(base_dir + 'dataset/train.json') as f:
        data = json.load(f)

    var = [(ann['image_id'], ann['category_id']) for ann in data['annotations']]
    X = np.ones((len(data['annotations']),1))
    y = np.array([v[1] for v in var])
    groups = np.array([v[0] for v in var])

    # StratifiedGroupKFold 적용
    num_folds, random_state = 5, 7
    skf = StratifiedGroupKFold(n_splits=num_folds, shuffle=True, random_state=random_state)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y, groups)):
        # Train, Validation 데이터셋 생성
        train_image_ids = groups[train_idx]
        val_image_ids = groups[val_idx]

        train_dataset = [item for item in original_dataset if item['image_id'] in train_image_ids]
        val_dataset = [item for item in original_dataset if item['image_id'] in val_image_ids]

        # Fold별로 데이터셋 등록
        global fold_train_name
        fold_train_name= f'coco_trash_train_fold_{fold+1}'
        global fold_val_name 
        fold_val_name = f'coco_trash_val_fold_{fold+1}'

        # 기존에 동일한 이름으로 등록된 데이터셋이 있다면 삭제
        # if fold_train_name in DatasetCatalog:
        #     DatasetCatalog.remove(fold_train_name)
        # if fold_val_name in DatasetCatalog:
        #     DatasetCatalog.remove(fold_val_name)    

        # 새로운 데이터셋 등록
        DatasetCatalog.register(fold_train_name, lambda d=train_dataset: d)
        MetadataCatalog.get(fold_train_name).set(thing_classes=original_metadata.thing_classes)
        
        DatasetCatalog.register(fold_val_name, lambda d=val_dataset: d)
        MetadataCatalog.get(fold_val_name).set(thing_classes=original_metadata.thing_classes)

        fold_val_name = [f'coco_trash_val_fold_{fold + 1}'] 

        cfg = setup(args)
        cfg.defrost()
        cfg.FOLD_VAL_NAME = fold_val_name
        cfg.DATASETS.TRAIN = (fold_train_name,)
        cfg.DATASETS.TEST = ('coco_trash_test',)  # 테스트 데이터셋 사용

        # 현재 fold에 대한 학습 수행
        train(cfg, args, fold)

        # Fold별로 Wandb 로그
        wandb.log({"fold_completed": fold})


def train(cfg, args, fold):


    trainer_params = {
        # training loop is bounded by max steps, use a large max_epochs to make
        # sure max_steps is met first
        "max_epochs": 10**8,
        "max_steps": cfg.SOLVER.MAX_ITER,
        "val_check_interval": cfg.TEST.EVAL_PERIOD if cfg.TEST.EVAL_PERIOD > 0 else 100,
        "num_nodes": args.num_machines,
        "gpus": args.num_gpus,
        "num_sanity_val_steps": 0,
    }
    if cfg.SOLVER.AMP.ENABLED:
        trainer_params["precision"] = 16

    start_time = datetime.datetime.now().strftime("%m%d_%H%M%S")

    # EarlyStopping 및 ModelCheckpoint 콜백 설정
    early_stopping_callback = EarlyStopping(
        monitor="mAP75",  # 평가할 지표
        patience=cfg.SOLVER.EARLY_STOPING,  # 개선되지 않을 때 중지할 에포크 수
        verbose=True,
        mode="max"
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{base_dir}/output/{start_time}",
        monitor="mAP75",
        filename=f"best_model-{start_time}-{{epoch:02d}}-{{avg_val_loss:.2f}}",
        save_top_k=1,
        mode="max"
    )



    last_checkpoint = os.path.join(f"{base_dir}/output/{start_time}", f"model_final_fold{fold}.pth")
    module = TrainingModule(cfg)
    
    if args.resume:
        # resume training from checkpoint
        trainer_params["resume_from_checkpoint"] = last_checkpoint
        logger.info(f"Resuming training from checkpoint: {last_checkpoint}.")
    
    trainer = pl.Trainer(callbacks=[early_stopping_callback, checkpoint_callback],**trainer_params)
    # trainer = pl.Trainer(**trainer_params)

    logger.info(f"start to train with {args.num_machines} nodes and {args.num_gpus} GPUs")

    
    data_module = DataModule(cfg)

    if args.eval_only:
        logger.info("Running inference")

        
        pred = trainer.predict(module, data_module)
        pred_str_list = []
        file_name_list = [] 

        for pred_str,file_name in pred:

            pred_str_list.append(pred_str)
            image_id = os.path.join(*file_name.split(os.sep)[-2:])
            file_name_list.append(image_id)

        submission = pd.DataFrame()
        submission['PredictionString'] = pred_str_list
        submission['image_id'] = file_name_list
        submission.to_csv(os.path.join(cfg.OUTPUT_DIR, f'submission_det.csv'), index=None)

    else:
        logger.info("Running training")
        trainer.fit(module, data_module)



def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.defrost()
    
    default_setup(cfg, args)
    cfg.freeze()
    return cfg


def invoke_main() -> None:
    parser = default_argument_parser()
    args = parser.parse_args()
    logger.info("Command Line Args: %s", args)
    main(args)


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
