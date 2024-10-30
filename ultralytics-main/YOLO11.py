import os
from datetime import datetime

from tqdm import tqdm
from ultralytics import RTDETR
from ultralytics import YOLO
import pandas as pd
import wandb

wandb.login(key="20ced4618a33e8061ca7264d38e0409df13c8daa")

from ultralytics.data.augment import Albumentations
from ultralytics.utils import LOGGER, colorstr


def __init__(self, p=1.0):
    self.p = p
    self.transform = None
    prefix = colorstr("albumentations: ")

    try:
        import albumentations as A

        # List of possible spatial transforms
        spatial_transforms = {
            "Affine",
            "BBoxSafeRandomCrop",
            "CenterCrop",
            "CoarseDropout",
            "Crop",
            "CropAndPad",
            "CropNonEmptyMaskIfExists",
            "D4",
            "ElasticTransform",
            "Flip",
            "GridDistortion",
            "GridDropout",
            "HorizontalFlip",
            "Lambda",
            "LongestMaxSize",
            "MaskDropout",
            "MixUp",
            "Morphological",
            "NoOp",
            "OpticalDistortion",
            "PadIfNeeded",
            "Perspective",
            "PiecewiseAffine",
            "PixelDropout",
            "RandomCrop",
            "RandomCropFromBorders",
            "RandomGridShuffle",
            "RandomResizedCrop",
            "RandomRotate90",
            "RandomScale",
            "RandomSizedBBoxSafeCrop",
            "RandomSizedCrop",
            "Resize",
            "Rotate",
            "SafeRotate",
            "ShiftScaleRotate",
            "SmallestMaxSize",
            "Transpose",
            "VerticalFlip",
            "XYMasking",
        }  # from https://albumentations.ai/docs/getting_started/transforms_and_targets/#spatial-level-transforms

        # Transforms
        T = [
            # A.OneOf([A.Blur(p=1.0), A.MedianBlur(p=1.0), A.GaussianBlur(p=0.01)], p=0.3),
            # A.ToGray(p=0.3),
            # A.CLAHE(p=0.3),
            # A.RandomBrightnessContrast(p=0.3),
            # A.RandomGamma(p=0.3),
            # A.HorizontalFlip(p=0.5),
        ]

        # Compose transforms
        self.contains_spatial = any(
            transform.__class__.__name__ in spatial_transforms for transform in T
        )
        self.transform = (
            A.Compose(
                T,
                bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
            )
            if self.contains_spatial
            else A.Compose(T)
        )
        LOGGER.info(
            prefix
            + ", ".join(f"{x}".replace("always_apply=False, ", "") for x in T if x.p)
        )
    except ImportError:  # package not installed, skip
        pass
    except Exception as e:
        LOGGER.info(f"{prefix}{e}")


Albumentations.__init__ = __init__

# COCO 사전 훈련된 RT-DETR-l 모델 로드
model = YOLO(
    "yolo11n.pt",
)

# 모델 정보 표시 (선택 사항)
model.info()

model.train(
    data="/home/taeyoung4060ti/바탕화면/level2-objectdetection-cv-01/ultralytics-main/dataset_yaml/ann_len_under40.yaml",
    epochs=20,
    imgsz=1024,
    batch=32,
    device=[0],
    cache="disk",
    project="Ultralytics",
    name="yolo11n",
    half=True,
    mosaic=1.0,
    close_mosaic=10,
    cos_lr=True,
    augment=True,
)

# results = model.predict(
#     source="ultralytics_dataset/test/images", agnostic_nms=True, augment=True
# )

# # 저장할 폴더 경로
# save_dir = "results"

# # 결과 저장 폴더가 없으면 생성
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)

# # 타임스탬프를 사용하여 고유한 파일명 생성
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# output_file = os.path.join(save_dir, f"predictions_coco_format_{timestamp}.csv")

# # COCO 형식으로 저장할 데이터를 담을 딕셔너리
# csv_data = {}

# # 각 결과 처리
# for result in tqdm(results):
#     image_id = (
#         result.path.split("/")[-3] + "/" + result.path.split("/")[-1]
#     )  # 파일 이름에서 image_id 추출
#     width, height = (
#         result.orig_shape[1],
#         result.orig_shape[0],
#     )  # 이미지의 원본 크기 (width, height)
#     prediction_string = ""
#     for box in result.boxes:
#         # YOLO 형식에서 COCO 형식으로 변환 (상대 좌표를 절대 좌표로 변환)
#         bbox = box.xywh.tolist()[
#             0
#         ]  # YOLO 형식에서 [x_center, y_center, width, height] 가져옴 (상대 좌표)
#         # 상대 좌표 (비율)를 절대 좌표 (픽셀 단위)로 변환
#         x_center, y_center, w, h = bbox
#         x_min = x_center - w / 2  # 중심 좌표에서 xmin 계산
#         y_min = y_center - h / 2  # 중심 좌표에서 ymin 계산
#         x_max = x_center + w / 2
#         y_max = y_center + h / 2

#         category_id = int(box.cls.item())
#         score = box.conf.item()

#         # COCO 형식으로 bbox 저장
#         bbox_str = f"{x_min} {y_min} {x_max} {y_max}"

#         # 같은 image_id의 예측 결과를 한 줄로 합침
#         prediction_string += f"{category_id} {score} {bbox_str} "

#     # image_id에 대해 기존 예측이 있으면 이어 붙임
#     if image_id in csv_data:
#         csv_data[image_id] += prediction_string
#     else:
#         csv_data[image_id] = prediction_string

# # 최종적으로 image_id와 PredictionString을 CSV로 저장
# df = pd.DataFrame(
#     [
#         {"PredictionString": pred_str, "image_id": image_id}
#         for image_id, pred_str in csv_data.items()
#     ]
# )

# df.to_csv(output_file, index=False)

# print(f"Predictions saved to {output_file}")
