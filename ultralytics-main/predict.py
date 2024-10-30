import os
from datetime import datetime

from tqdm import tqdm
from ultralytics import RTDETR
import pandas as pd

model = RTDETR(
    "/home/taeyoung4060ti/바탕화면/level2-objectdetection-cv-01/ultralytics-main/Ultralytics/RT-DETR-l50/weights/best.pt"
)

results = model.predict(
    source="ultralytics_dataset/test/images", augment=True, device=[0]
)

# 저장할 폴더 경로
save_dir = "results"

# 결과 저장 폴더가 없으면 생성
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 타임스탬프를 사용하여 고유한 파일명 생성
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = os.path.join(save_dir, f"predictions_coco_format_{timestamp}.csv")

# COCO 형식으로 저장할 데이터를 담을 딕셔너리
csv_data = {}

# 각 결과 처리
for result in tqdm(results):
    image_id = (
        result.path.split("/")[-3] + "/" + result.path.split("/")[-1]
    )  # 파일 이름에서 image_id 추출
    width, height = (
        result.orig_shape[1],
        result.orig_shape[0],
    )  # 이미지의 원본 크기 (width, height)
    prediction_string = ""
    for box in result.boxes:
        # YOLO 형식에서 COCO 형식으로 변환 (상대 좌표를 절대 좌표로 변환)
        bbox = box.xywh.tolist()[
            0
        ]  # YOLO 형식에서 [x_center, y_center, width, height] 가져옴 (상대 좌표)
        # 상대 좌표 (비율)를 절대 좌표 (픽셀 단위)로 변환
        x_center, y_center, w, h = bbox
        x_min = max(0, x_center - w / 2)
        y_min = max(0, y_center - h / 2)
        x_max = min(width, x_center + w / 2)
        y_max = min(height, y_center + h / 2)

        category_id = int(box.cls.item())
        score = box.conf.item()

        # COCO 형식으로 bbox 저장
        bbox_str = f"{x_min} {y_min} {x_max} {y_max}"

        # 같은 image_id의 예측 결과를 한 줄로 합침
        prediction_string += f"{category_id} {score} {bbox_str} "

    # image_id에 대해 기존 예측이 있으면 이어 붙임
    if image_id in csv_data:
        csv_data[image_id] += prediction_string
    else:
        csv_data[image_id] = prediction_string

# 최종적으로 image_id와 PredictionString을 CSV로 저장
df = pd.DataFrame(
    [
        {"PredictionString": pred_str, "image_id": image_id}
        for image_id, pred_str in csv_data.items()
    ]
)

df.to_csv(output_file, index=False)

print(f"Predictions saved to {output_file}")
