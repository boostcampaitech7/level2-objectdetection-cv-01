import pandas as pd
import numpy as np
from ensemble_boxes import nms
from pycocotools.coco import COCO

#submission_df 는 제출 가능한 csv를 pd로 읽은 df들의 리스트
def ensenble(submission_df):
    image_ids = submission_df[0]['image_id'].tolist()
    annotation = '/data/ephemeral/home/level2-objectdetection-cv-01/dataset/test.json'
    coco = COCO(annotation)

    prediction_strings = []
    file_names = []
    # ensemble 시 설정할 iou threshold 이 부분을 바꿔가며 대회 metric에 알맞게 적용해봐요!
    iou_thr = 0.4

    # 각 image id 별로 submission file에서 box좌표 추출
    for i, image_id in enumerate(image_ids):
        prediction_string = ''
        boxes_list = []
        scores_list = []
        labels_list = []
        image_info = coco.loadImgs(i)[0]
        # 각 submission file 별로 prediction box좌표 불러오기
        for df in submission_df:

            predict_string = df[df['image_id'] == image_id]['PredictionString']
            if predict_string.empty:
                print(image_id)
                continue
            predict_string = predict_string.tolist()[0]
            predict_list = str(predict_string).split()

            if len(predict_list)==0 or len(predict_list)==1:
                continue
    
            predict_list = np.reshape(predict_list, (-1, 6))
            box_list = []

            for box in predict_list[:, 2:6].tolist():
                # box의 각 좌표를 float형으로 변환한 후 image의 넓이와 높이로 각각 정규화
                image_width = image_info['width']
                image_height = image_info['height']
                box[0] = float(box[0]) / image_width
                box[1] = float(box[1]) / image_height
                box[2] = float(box[2]) / image_width
                box[3] = float(box[3]) / image_height
                box_list.append(box)

            boxes_list.append(box_list)
            scores_list.append(list(map(float, predict_list[:, 1].tolist())))
            labels_list.append(list(map(int, predict_list[:, 0].tolist())))

        # 예측 box가 있다면 이를 ensemble 수행
        if len(boxes_list):
            # ensemble_boxes에서 import한 nms()를 사용하여 NMS 계산 수행
            # 👉 위의 코드에서 만든 정보들을 함수에 간단하게 적용해보세요!
            # nms에 필요한 인자: [NMS할 box의 lists, confidence score의 lists, label의 list, iou에 사용할 threshold]
            boxes, scores, labels = nms(boxes_list, scores_list, labels_list, 0.5)
            for box, score, label in zip(boxes, scores, labels):
                prediction_string += str(label) + ' ' + str(score) + ' ' + str(box[0] * image_info['width']) + ' ' + str(box[1] * image_info['height']) + ' ' + str(box[2] * image_info['width']) + ' ' + str(box[3] * image_info['height']) + ' '

        prediction_strings.append(prediction_string)
        file_names.append(image_id)
    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    return submission