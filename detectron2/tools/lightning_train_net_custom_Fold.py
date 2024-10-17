#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# Lightning Trainer should be considered beta at this point
# We have confirmed that training and validation run correctly and produce correct results
# Depending on how you launch the trainer, there are issues with processes terminating correctly
# This module is still dependent on D2 logging, but could be transferred to use Lightning logging

import logging
import os
import time
import copy
import weakref
from collections import OrderedDict
from typing import Any, Dict, List
import pytorch_lightning as pl  # type: ignore
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pycocotools.coco import COCO

import torch 
import datetime
import wandb

import pandas as pd
import numpy as np
from ensemble_boxes import weighted_boxes_fusion

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
from detectron2.evaluation import print_csv_format,COCOEvaluator
from detectron2.evaluation.testing import flatten_results_dict
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage
from detectron2.utils.logger import setup_logger
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo
import detectron2.data.transforms as T

# from train_net import build_evaluator



def build_evaluator(cfg, dataset_name, output_folder=None):
    if output_folder is None:
        os.makedirs('./output_eval', exist_ok = True)
        output_folder = './output_eval'
        
    return COCOEvaluator(dataset_name, cfg, False, output_folder)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("detectron2")


class TrainingModule(LightningModule):
    def __init__(self, cfg, current_fold):
        super().__init__()
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        self.cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        self.storage: EventStorage = None
        self.model = build_model(self.cfg)

        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.current_fold = current_fold 

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

        # if self.storage.iter % self.cfg.SOLVER.CHECKPOINT_PERIOD == 0:
        #     start_time = datetime.datetime.now().strftime("%m%d_%H%M%S")
        #     self.checkpointer.save(f"{self.checkpoint_dir}/model_final_{start_time}")

        return training_step_outpus

    def training_epoch_end(self, training_step_outputs):
        self.iteration_timer.after_train()
        if comm.is_main_process():
            save_position = os.path.join(base_dir, 'output', start_time)
            checkpoint_name = f"{save_position}/model_final_Fold{self.current_fold}"
            self.checkpointer.save(checkpoint_name)
        for writer in self.writers:
            writer.write()
            writer.close()
        self.storage.__exit__(None, None, None)

    def _process_dataset_evaluation_results(self) -> OrderedDict:
        results = OrderedDict()
        for idx, dataset_name in enumerate(self.cfg.DATASETS.TEST):
            results[dataset_name] = self._evaluators[idx].evaluate()
            if comm.is_main_process():
                print_csv_format(results[dataset_name])

        if len(results) == 1:
            results = list(results.values())[0]
        return results

    def _reset_dataset_evaluators(self):
        self._evaluators = []
        for dataset_name in self.cfg.DATASETS.TEST:
            evaluator = build_evaluator(self.cfg, dataset_name)
            evaluator.reset()
            self._evaluators.append(evaluator)

    def on_validation_epoch_start(self,):
        self._reset_dataset_evaluators()

    def validation_epoch_end(self, _outputs):
        results = self._process_dataset_evaluation_results()
        flattened_results = flatten_results_dict(results)

        # Extract mAP, mAP50, and mAP75 and store them in instance variables
        if 'bbox/AP' in flattened_results:
            self.mAP = flattened_results['bbox/AP']
            self.log('mAP', self.mAP) 
        if 'bbox/AP50' in flattened_results:
            self.mAP50 = flattened_results['bbox/AP50']
            self.log('mAP50', self.mAP50) 
        if 'bbox/AP75' in flattened_results:
            self.mAP75 = flattened_results['bbox/AP75']
            self.log('mAP75', self.mAP75)  

        wandb.log({
            "mAP": self.mAP,
            "mAP50": self.mAP50,
            "mAP75": self.mAP75,
            "current_step": self.storage.iter,
            "learning_rate": self.optimizers().param_groups[0]["lr"]
        })



        flattened_results = flatten_results_dict(results)
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
        outputs = self.model(batch)
        self._evaluators[dataloader_idx].process(batch, outputs)

    def on_predict_epoch_start(self):
        # return super().on_predict_epoch_start()
        self.model.eval()

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

            if not outputs:
                print(f"No instances found in the out for file: {data_file_name}")
                prediction_results.append([prediction_string, data_file_name])
                continue

            # 예측 결과 추출
            targets = outputs.pred_classes.cpu().tolist()
            boxes = [i.cpu().detach().numpy() for i in outputs.pred_boxes]
            scores = outputs.scores.cpu().tolist()

            # 예측 문자열 생성
            for target, box, score in zip(targets, boxes, scores):
                prediction_string += (f"{target} {score} {box[0]} {box[1]} {box[2]} {box[3]} ")

            # 결과 저장
            prediction_results.append([prediction_string, data_file_name])
    
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
    
    dataset_dict['image'] = torch.as_tensor(image.transpose(2,0,1).astype('float32'))
    
    annos = [
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
    
    # dataset_dict['image'] = image
    
    dataset_dict['image'] = torch.as_tensor(image.transpose(2,0,1).astype('float32'))

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
        dataloaders = []
        for dataset_name in self.cfg.FOLD_VAL_NAME: #TODO?
            dataloaders.append(build_detection_test_loader(self.cfg, dataset_name,ValMapper))
        return dataloaders
    
    def predict_dataloader(self):
        return build_detection_test_loader(self.cfg, 'coco_trash_test', TestMapper) #TODO? 



def main(args):
    global start_time
    start_time = datetime.datetime.now().strftime("%m%d_%H%M%S")
    wandb.init(project='trash object detection',entity='tayoung1005-aitech',config=args,reinit=True)

    try:
        register_coco_instances('coco_trash_test', {}, '/home/minipin/cv-01/level2-objectdetection-cv-01/dataset/test.json', '/home/minipin/cv-01/level2-objectdetection-cv-01/dataset')
    except AssertionError:
        pass


    if args.eval_only:
        cfg = setup(args)
        train(cfg, args, fold=-1)

    else:
        global base_dir
        base_dir = '/home/minipin/cv-01/level2-objectdetection-cv-01/'
        checkpoint_dir = os.path.join(base_dir, 'output', start_time)
        os.makedirs(checkpoint_dir, exist_ok=True)

        num_fold, random_state = 5, 7

        for fold in range(1,num_fold+1):
            train_json_path = base_dir + f'data/{num_fold}splits_{random_state}/fold{fold}-train.json'
            val_json_path = base_dir + f'data/{num_fold}splits_{random_state}/fold{fold}-val.json'
        
            # Fold별로 데이터셋 등록
            fold_train_name = f'coco_trash_train_fold_{fold}'
            fold_val_name = f'coco_trash_val_fold_{fold}'

            try:
                register_coco_instances(fold_train_name, {}, train_json_path, base_dir + 'dataset')
            except AssertionError:
                pass
            
            try:
                register_coco_instances(fold_val_name, {}, val_json_path, base_dir + 'dataset')
            except AssertionError:
                pass

            
            MetadataCatalog.get(fold_train_name).thing_classes = ["General trash", "Paper", "Paper pack", "Metal", 
                                                                "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]
            MetadataCatalog.get(fold_val_name).thing_classes = ["General trash", "Paper", "Paper pack", "Metal", 
                                                                "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]

 
            cfg = setup(args)
            cfg.defrost()
            cfg.FOLD_VAL_NAME = [fold_val_name] 
            cfg.DATASETS.TRAIN = (fold_train_name,)
            cfg.DATASETS.TEST = ('coco_trash_test',)  
            cfg.FOLD_NUMBER = fold
            
            # 현재 fold에 대한 학습 수행
            train(cfg, args, fold)




def train(cfg, args, fold):

    base_dir = '/home/minipin/cv-01/level2-objectdetection-cv-01/'
    start_time = datetime.datetime.now().strftime("%m%d_%H%M%S")


    trainer_params = {
        # training loop is bounded by max steps, use a large max_epochs to make
        # sure max_steps is met first
        "max_epochs": 10**8,
        "max_steps": cfg.SOLVER.MAX_ITER,
        "val_check_interval": cfg.TEST.EVAL_PERIOD if cfg.TEST.EVAL_PERIOD > 0 else 10**8,
        # "val_check_interval": cfg.SOLVER.EVAL_PERIOD ,
        "num_nodes": args.num_machines,
        "gpus": args.num_gpus,
        "num_sanity_val_steps": 0,
    }
    if cfg.SOLVER.AMP.ENABLED:
        trainer_params["precision"] = 16


    checkpoint_dir = os.path.join(base_dir, 'output', start_time)
    # checkpoint_callback = ModelCheckpoint(
    #     dirpath=checkpoint_dir,
    #     monitor="mAP",
    #     filename=f"best_model-fold{fold}-{{epoch:02d}}-mAP{{mAP:.2f}}",
    #     save_top_k=1,
    #     mode="max",
    #     save_weights_only=True
    # )
    last_checkpoint = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")


    module = TrainingModule(cfg, current_fold = fold)
    
    if args.resume:
        trainer_params["resume_from_checkpoint"] = last_checkpoint
        logger.info(f"Resuming training from checkpoint: {last_checkpoint}.")
    
    

    # trainer = pl.Trainer(**trainer_params, callbacks=checkpoint_callback)
    trainer = pl.Trainer(**trainer_params)
    logger.info(f"start to train with {args.num_machines} nodes and {args.num_gpus} GPUs")

    
    data_module = DataModule(cfg)

    if args.eval_only:
        eval_dir = os.path.join(cfg.OUTPUT_DIR, 'test3_5split_333randomstate')
        checkpoints = [os.path.join(eval_dir, ckpt) for ckpt in os.listdir(eval_dir) if ckpt.endswith(".pth")]
        
        # Iterate through checkpoints and make predictions
        cnt = 1
        for checkpoint_path in checkpoints:
            logger.info(f"Evaluating checkpoint: {checkpoint_path}")

            cfg.defrost()
            # checkpoint_filename = os.path.basename(checkpoint_path)
            # temp_weights_path = os.path.join(eval_dir, f"{checkpoint_filename}")
            # extract_weights_only(checkpoint_path, temp_weights_path)

            cfg.MODEL.WEIGHTS = checkpoint_path
            module = TrainingModule(cfg=cfg)
            # plot_weights_from_checkpoint(cfg.MODEL.WEIGHTS)
            # return
            pred = trainer.predict(module, data_module)
            pred_str_list = []
            file_name_list = []
        
            for idx, pred_item in enumerate(pred):
                if pred_item:
                    pred_str, file_name = pred_item[0]
                    file_name = os.path.join(*file_name.split(os.sep)[-2:])
                    pred_str_list.append(pred_str)
                    file_name_list.append(file_name)

            submission = pd.DataFrame()
            submission['PredictionString'] = pred_str_list
            submission['image_id'] = file_name_list
            submission.to_csv(os.path.join(eval_dir,f'result_{cnt}.csv'), index=None)
            
            cnt += 1
            
    else:
        logger.info("Running training")
        trainer.fit(module, data_module)

def plot_weights_from_checkpoint(checkpoint_path):

    import torch
    import matplotlib.pyplot as plt
    import numpy as np
    # Checkpoint에서 가중치 로드
    checkpoint = torch.load(checkpoint_path)
    
    # state_dict에서 모델 가중치만 추출
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    # 가중치 산점도로 시각화
    for name, param in state_dict.items():
        if 'weight' in name:  # 가중치 파라미터만 선택
            weights = param.detach().cpu().numpy()  # 가중치를 numpy 배열로 변환
            flattened_weights = weights.flatten()

            plt.figure(figsize=(6, 4))
            plt.scatter(np.arange(len(flattened_weights)), flattened_weights, alpha=0.5, color='blue')
            plt.title(f'Weight Distribution of {name} (Scatter Plot)')
            plt.xlabel('Weight Index')
            plt.ylabel('Weight Value')
            plt.grid(True)
            plt.show()

def extract_weights_only(checkpoint_path, output_path):
    # Checkpoint에서 가중치만 불러오기
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    if 'state_dict' in checkpoint:
        new_state_dict = {}
        for layer_name, param_tensor in checkpoint['state_dict'].items():
            # 'model.'을 제거한 키를 사용
            new_key = layer_name.replace('model.', '', 1)
            new_state_dict[new_key] = param_tensor

        torch.save(new_state_dict, output_path)
    
    else:
        torch.save(checkpoint, output_path)
    
    return output_path
    
    

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def invoke_main() -> None:
    parser = default_argument_parser()
    args = parser.parse_args()
    logger.info("Command Line Args: {args}")
    main(args)


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
