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
import shutil
from collections import OrderedDict
from typing import Any, Dict, List
import pytorch_lightning as pl  # type: ignore
from pytorch_lightning import LightningDataModule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torchvision.ops import nms

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

from ensemble_boxes import weighted_boxes_fusion

def build_evaluator(cfg, dataset_name, output_folder=None):
    if output_folder is None:
        os.makedirs('./output_eval', exist_ok = True)
        output_folder = './output_eval'
        
    return COCOEvaluator(dataset_name, cfg, False, output_folder)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("detectron2")


class TrainingModule(pl.LightningModule):
    def __init__(self, cfg, eval_only=False):
        super().__init__()
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        self.cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        self.storage: EventStorage = None
        self.model = build_model(self.cfg)
        self.eval_only = eval_only
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
        # If eval_only, do not initialize storage
        if not self.eval_only:
            # Initialize storage if not in eval_only mode
            if self.storage is None:
                self.storage = EventStorage(self.start_iter)
            else:
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
        start_time = datetime.datetime.now().strftime("%m%d_%H%M%S")

        self.iteration_timer.after_train()
        if comm.is_main_process():
            self.checkpointer.save(f"model_final{start_time}")
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
    
    
    def predict_step(self, batch, batch_idx):
        prediction_results = []
        threshold = 0.5
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
                continue

            # 예측 결과 추출
            targets = outputs.pred_classes.cpu().tolist()
            boxes = [i.cpu().detach().numpy() for i in outputs.pred_boxes]
            scores = outputs.scores.cpu().tolist()

            # keep_indices = [i for i, score in enumerate(scores) if score >= threshold]
            targets = [targets[i] for i in keep_indices]
            boxes = [boxes[i] for i in keep_indices]
            scores = [scores[i] for i in keep_indices]

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

    start_time = datetime.datetime.now().strftime("%m%d_%H%M%S")
    checkpoint_dir = os.path.join(base_dir, 'output', start_time)
    os.makedirs(checkpoint_dir, exist_ok=True)

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


    if args.eval_only:
        cfg = setup(args)
        train(cfg, args, 0, checkpoint_dir, start_time)

    else:
        # StratifiedGroupKFold 적용
        num_folds, random_state = 5, 1999
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
            if fold_train_name in DatasetCatalog:
                DatasetCatalog.remove(fold_train_name)
            if fold_val_name in DatasetCatalog:
                DatasetCatalog.remove(fold_val_name)    

            # 새로운 데이터셋 등록
            DatasetCatalog.register(fold_train_name, lambda d=train_dataset: d)
            MetadataCatalog.get(fold_train_name).set(thing_classes=original_metadata.thing_classes)
            
            DatasetCatalog.register(fold_val_name, lambda d=val_dataset: d)
            MetadataCatalog.get(fold_val_name).set(thing_classes=original_metadata.thing_classes)

            cfg = setup(args)
            cfg.defrost()
            cfg.FOLD_VAL_NAME = [fold_val_name] 
            cfg.DATASETS.TRAIN = (fold_train_name,)
            cfg.DATASETS.TEST = ('coco_trash_test',)  
        
            # 현재 fold에 대한 학습 수행
            train(cfg, args, fold, checkpoint_dir, start_time)

            # Fold별로 Wandb 로그
            wandb.log({"fold_completed": fold})


def train(cfg, args, fold, checkpoint_dir, start_time):


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

    # EarlyStopping 및 ModelCheckpoint 콜백 설정
    # early_stopping_callback = EarlyStopping(
    #     monitor="mAP75",  # 평가할 지표
    #     patience=cfg.SOLVER.EARLY_STOPING,  # 개선되지 않을 때 중지할 에포크 수
    #     verbose=True,
    #     mode="max"
    # )

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor="mAP75",
        filename=f"best_model-fold{fold}-{{epoch:02d}}-mAP{{mAP:.2f}}",
        save_top_k=1,
        mode="max"
    )

    # last_checkpoint = os.path.join(f"{base_dir}/output/{start_time}", f"model_final_fold{fold}.pth")
    
    # if args.resume:
        # resume training from checkpoint
        # trainer_params["resume_from_checkpoint"] = last_checkpoint
        # logger.info(f"Resuming training from checkpoint: {last_checkpoint}.")
    
    trainer = pl.Trainer(callbacks=checkpoint_callback,**trainer_params)
    logger.info(f"start to train with {args.num_machines} nodes and {args.num_gpus} GPUs")

    data_module = DataModule(cfg)
    if args.eval_only:
        os.rmdir(checkpoint_dir)
        logger.info("Running inference")
        
        eval_dir = os.path.join(cfg.OUTPUT_DIR,"1007_111543")
        # Get the list of all checkpoints in the fold-specific folder
        checkpoints = [os.path.join(eval_dir, ckpt) for ckpt in os.listdir(eval_dir) if ckpt.endswith(".ckpt")]
        results = []

        # Iterate through checkpoints and make predictions
        for checkpoint_path in checkpoints:
            logger.info(f"Evaluating checkpoint: {checkpoint_path}")
            
            cfg.defrost()
            checkpoint_filename = os.path.basename(checkpoint_path)
            temp_weights_path = os.path.join(eval_dir, f"weight_only_{checkpoint_filename}")

            cfg.MODEL.WEIGHTS = extract_weights_only(checkpoint_path, temp_weights_path)
            # cfg.MODEL.WEIGHTS = checkpoint_path
            module = TrainingModule(cfg=cfg, eval_onl   y=True)

            # module = TrainingModule.load_from_checkpoint(checkpoint_path, cfg=cfg, eval_only=True)
            pred = trainer.predict(module, data_module)

            for pre in pred:
                for pred_str, file_name in pre:
                    image_id = os.path.join(*file_name.split(os.sep)[-2:])
                    results.append((image_id, pred_str))

        # Perform wbf to combine results from all checkpoints
        final_predictions = apply_weighted_boxes_fusion(results)

        # Save the final submission
        submission = pd.DataFrame(final_predictions, columns=['PredictionString','image_id'])
        submission.to_csv(os.path.join(eval_dir, f'submission_det.csv'), index=None)

        return

    else:
        module = TrainingModule(cfg)
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

def extract_weights_only(checkpoint_path, output_path):
    # Load the checkpoint and extract only the model weights
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print(f"Extracted state_dict keys: {state_dict.keys()}")  # 가중치 키 확인
    else:
        state_dict = checkpoint  # If not a PyTorch Lightning checkpoint
    # Save the extracted weights to a temporary file
    torch.save(state_dict, output_path)
    return output_path

def apply_weighted_boxes_fusion(predictions):
    voting_results = {}

    for image_id, pred_str in predictions:
        if image_id not in voting_results:
            voting_results[image_id] = []

        # 예측 문자열을 파싱하여 bounding box와 점수를 추출
        boxes, scores, labels = parse_prediction_string(pred_str)
        voting_results[image_id].append((boxes, scores, labels))

    # 각 이미지 ID별로 예측 결과를 합칩니다.
    final_results = []
    for image_id, preds in voting_results.items():
        combined_boxes, combined_scores, combined_labels = aggregate_wbf_predictions(preds)
        final_pred_str = create_prediction_string(combined_boxes, combined_scores, combined_labels)
        final_results.append((final_pred_str, image_id))

    return final_results

def parse_prediction_string(pred_str):
    elements = pred_str.strip().split()
    boxes = []
    scores = []
    labels = []

    for i in range(0, len(elements), 6):
        label = int(elements[i])
        score = float(elements[i + 1])
        x_min = float(elements[i + 2])
        y_min = float(elements[i + 3])
        width = float(elements[i + 4])
        height = float(elements[i + 5])
        
        # Convert COCO format (x_min, y_min, width, height) to Pascal VOC format (x_min, y_min, x_max, y_max)
        x_max = x_min + width
        y_max = y_min + height
        boxes.append([x_min, y_min, x_max, y_max])
        scores.append(score)
        labels.append(label)

    return torch.tensor(boxes), torch.tensor(scores), torch.tensor(labels)

def aggregate_wbf_predictions(preds):
    all_boxes = []
    all_scores = []
    all_labels = []
    image_width = 1024 
    image_height = 1024
    
    for boxes, scores, labels in preds:
        # Skip empty predictions
        if boxes.numel() == 0:
            continue

        # confidence score를 기준으로 내림차순 정렬
        sorted_indices = torch.argsort(scores, descending=True)
        boxes = boxes[sorted_indices]
        scores = scores[sorted_indices]
        labels = labels[sorted_indices]

        all_boxes.append(boxes.numpy().tolist())
        all_scores.append(scores.numpy().tolist())
        all_labels.append(labels.numpy().tolist())

    # Check if there are any predictions to aggregate
    if len(all_boxes) == 0:
        # Return empty tensors if there are no valid predictions
        return torch.empty((0, 4)), torch.empty((0,)), torch.empty((0,))
    elif len(all_boxes) == 1:
        return torch.tensor(all_boxes[0]), torch.tensor(all_scores[0]), torch.tensor(all_labels[0])
    
    # WBF를 적용하기 위해 bounding box 좌표를 정규화합니다.
    for i in range(len(all_boxes)):
        for j in range(len(all_boxes[i])):
            all_boxes[i][j][0] /= image_width  # x_min 정규화
            all_boxes[i][j][2] /= image_width  # x_max 정규화
            all_boxes[i][j][1] /= image_height  # y_min 정규화
            all_boxes[i][j][3] /= image_height  # y_max 정규화

    # Weighted Boxes Fusion 적용
    wbf_boxes, wbf_scores, wbf_labels = weighted_boxes_fusion(
        all_boxes, all_scores, all_labels, iou_thr=0.5, skip_box_thr=0.0001
    )

    # 정규화된 좌표를 이미지의 실제 크기로 되돌립니다.
    for i in range(len(wbf_boxes)):
        wbf_boxes[i][0] *= image_width  # x_min 역정규화
        wbf_boxes[i][2] *= image_width  # x_max 역정규화
        wbf_boxes[i][1] *= image_height  # y_min 역정규화
        wbf_boxes[i][3] *= image_height  # y_max 역정규화

    return torch.tensor(wbf_boxes), torch.tensor(wbf_scores), torch.tensor(wbf_labels)


def create_prediction_string(boxes, scores, labels):
    pred_str = ''
    for label, score, box in zip(labels, scores, boxes):
        x_min, y_min, x_max, y_max = box.tolist()
        pred_str += f"{int(label)} {score:.14f} {x_min:.5f} {y_min:.5f} {x_max:.5f} {y_max:.5f} "
    return pred_str.strip()

if __name__ == "__main__":
    invoke_main()  # pragma: no cover
