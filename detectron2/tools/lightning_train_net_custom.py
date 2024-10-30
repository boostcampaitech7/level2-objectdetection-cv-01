#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# Lightning Trainer should be considered beta at this point
# We have confirmed that training and validation run correctly and produce correct results
# Depending on how you launch the trainer, there are issues with processes terminating correctly
# This module is still dependent on D2 logging, but could be transferred to use Lightning logging

import copy
import logging
import os
import time
import weakref
from collections import OrderedDict
from typing import Any, Dict, List

import albumentations as A
import detectron2.data.transforms as T
import detectron2.utils.comm as comm
import pandas as pd
import pytorch_lightning as pl  # type: ignore
import torch
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
    detection_utils,
)
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import (
    DefaultTrainer,
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
)
from detectron2.evaluation import COCOEvaluator, print_csv_format
from detectron2.evaluation.testing import flatten_results_dict
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage
from detectron2.utils.logger import setup_logger
from pytorch_lightning import LightningDataModule, LightningModule

# from train_net import build_evaluator



def build_evaluator(cfg, dataset_name, output_folder=None):
    if output_folder is None:
        os.makedirs('./output_eval', exist_ok = True)
        output_folder = './output_eval'
        
    return COCOEvaluator(dataset_name, cfg, False, output_folder)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("detectron2")


class TrainingModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        self.cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        self.storage: EventStorage = None
        self.model = build_model(self.cfg)

        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER

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

        if self.storage.iter % self.cfg.SOLVER.CHECKPOINT_PERIOD == 0:
            self.checkpointer.save(f"model_{self.storage.iter}")

        return training_step_outpus

    def training_epoch_end(self, training_step_outputs):
        self.iteration_timer.after_train()
        if comm.is_main_process():
            self.checkpointer.save("model_final")
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
            # print("???????????????????????????????????????????????????",dataset_name)
            evaluator = build_evaluator(self.cfg, dataset_name)
            evaluator.reset()
            self._evaluators.append(evaluator)

    def on_validation_epoch_start(self,):
        self._reset_dataset_evaluators()

    def validation_epoch_end(self, _outputs):
        results = self._process_dataset_evaluation_results()

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

        # test dataloder의 batch_size는 1
        # TODO : batch size 변경 
        prediction_string = ''
        data = batch
        data = data[0]
        data_file_name,data = data['file_name'],data['image']
        height, width = data.shape[:2]
        
        #TODO? 
        # if self.input_format == "RGB":
        #     # whether the model expects BGR inputs or RGB
        #     original_image = original_image[:, :, ::-1]


        data = self.aug.get_transform(data).apply_image(data)
        data = torch.as_tensor(data.astype("float32").transpose(2, 0, 1))
        data.to(self.cfg.MODEL.DEVICE)

        inputs = {"image": data, "height": height, "width": width}

        outputs = self.model([inputs])[0]['instances'] #==self.model at eval  

        targets = outputs.pred_classes.cpu().tolist()
        boxes = [i.cpu().detach().numpy() for i in outputs.pred_boxes]
        scores = outputs.scores.cpu().tolist()

        for target, box, score in zip(targets,boxes,scores):
            prediction_string += (str(target) + ' ' + str(score) + ' ' + str(box[0]) + ' ' 
            + str(box[1]) + ' ' + str(box[2]) + ' ' + str(box[3]) + ' ')

        return prediction_string, data_file_name


    def configure_optimizers(self):
        optimizer = build_optimizer(self.cfg, self.model)
        self._best_param_group_id = hooks.LRScheduler.get_best_param_group_id(optimizer)
        scheduler = build_lr_scheduler(self.cfg, optimizer)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
    
def AlbumentationTransform(img,anno,mode):
    #img = 1024,1024,3 array
    #anno = list of dict [{'iscloud':0, 'bbox':[x0,y0,w,h], 'category_id':1, 'bbox_mode':1}]

    #트레인 테스트에 모두 적용되는 트랜스폼
    transform_both = [
        #A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1)
        ]

    if mode == 'train':
        #트레인 데이터에만 적용되는 트랜스폼
        rotation = A.OneOf([
                    A.NoOp(p=1),
                    A.Rotate(limit=(90, 90), p=1),
                    A.Rotate(limit=(180, 180), p=1),
                    A.Rotate(limit=(270, 270), p=1),
                ], p=1.0)
        flip = A.OneOf([
                A.NoOp(p=1),
                A.HorizontalFlip(p=1),
            ], p=1.0)
        transform_list = [rotation,flip]

        transform_list = transform_both+transform_list
        if transform_list:
            anno_df = pd.DataFrame(anno)
            transform = A.Compose(transform_list, bbox_params=A.BboxParams(format='coco', label_fields=['labels']))
            transformed = transform(image=img, bboxes=anno_df['bbox'].tolist(), labels=anno_df['category_id'].tolist())
            transformed_df = pd.DataFrame({'bbox':transformed['bboxes'], 'category_id':transformed['labels']})
            transformed_df['isclowd'] = 0
            transformed_df['bbox_mode'] = 1
            img = transformed['image']
            return img, transformed_df.to_dict(orient='records')
        return img, anno
    elif mode == 'test':
        # test time augmentation 테스트 에만 적용되는 트랜스폼
        tta = []
        tta = transform_both+tta
        if tta:
            transform = A.Compose(tta)
            transformed = transform(image=img)
            img = transformed['image']
        return img
    else: print('Unknown mode'); quit()
##### Mapper 
def TrainMapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    image = detection_utils.read_image(dataset_dict['file_name'], format='BGR')

    transform_list = [
       # T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
       # T.RandomBrightness(0.8, 1.8),
       # T.RandomContrast(0.6, 1.3)
    ]
    
    image, transforms = T.apply_transform_gens(transform_list, image)

    image, dataset_dict['annotations'] = AlbumentationTransform(image, dataset_dict['annotations'], 'train')

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
    
    image = AlbumentationTransform(image, 0, 'test')

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
        for dataset_name in self.cfg.DATASETS.TEST: #TODO?
            dataloaders.append(build_detection_test_loader(self.cfg, dataset_name,ValMapper))
        return dataloaders
    
    def predict_dataloader(self):
        return build_detection_test_loader(self.cfg, 'coco_trash_test', TestMapper) #TODO? 



def main(args):

    try:
        register_coco_instances('coco_trash_train', {}, '/data/ephemeral/home/level2-objectdetection-cv-01/dataset/train.json', '/data/ephemeral/home/level2-objectdetection-cv-01/dataset')
    except AssertionError:
        pass

    try:
        register_coco_instances('coco_trash_test', {}, '/data/ephemeral/home/level2-objectdetection-cv-01/dataset/test.json', '/data/ephemeral/home/level2-objectdetection-cv-01/dataset')
    except AssertionError:
        pass

    MetadataCatalog.get('coco_trash_train').thing_classes = ["General trash", "Paper", "Paper pack", "Metal", 
                                                            "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]


    cfg = setup(args)
    train(cfg, args)



def train(cfg, args):


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

    last_checkpoint = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")


    module = TrainingModule(cfg)
    
    if args.resume:
        # resume training from checkpoint
        # print('???????????????????????????????????')
        trainer_params["resume_from_checkpoint"] = last_checkpoint
        logger.info(f"Resuming training from checkpoint: {last_checkpoint}.")
    
    

    trainer = pl.Trainer(**trainer_params)
    logger.info(f"start to train with {args.num_machines} nodes and {args.num_gpus} GPUs")

    
    data_module = DataModule(cfg)

    if args.eval_only:
        logger.info("Running inference")

        
        pred = trainer.predict(module, data_module)
        pred_str_list = []
        file_name_list = [] 

        for pred_str,file_name in pred:

            pred_str_list.append(pred_str)
            file_name_list.append(file_name)

        submission = pd.DataFrame()
        submission['PredictionString'] = pred_str_list
        submission['image_id'] = file_name_list
        submission.to_csv(os.path.join(cfg.OUTPUT_DIR, f'submission_det_test.csv'), index=None)

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
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def invoke_main() -> None:
    parser = default_argument_parser()
    args = parser.parse_args()
    logger.info("Command Line Args:", args)
    main(args)


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
