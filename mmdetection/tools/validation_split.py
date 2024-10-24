import json
import numpy as np
import os
from sklearn.model_selection import StratifiedGroupKFold

def split_and_save(n_splits, random_state):
    folder = f'{n_splits}-fold_seed-{random_state}'
    dir = f'/data/ephemeral/home/level2-objectdetection-cv-01/dataset/{folder}'
    try:
        os.makedirs(dir)
    except:
        print('data already exists')
        return folder
    
    annotation = '/data/ephemeral/home/level2-objectdetection-cv-01/dataset/train.json'

    with open(annotation) as f: data = json.load(f)
    
    var = [(ann['image_id'], ann['category_id']) for ann in data['annotations']]
    X = np.ones((len(data['annotations']),1))
    y = np.array([v[1] for v in var])
    groups = np.array([v[0] for v in var])
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    train_idx_list = []
    val_idx_list = []
    for train_idx, val_idx in cv.split(X, y, groups):
        train_idx_list.append(train_idx)
        val_idx_list.append(val_idx)

    for idx, (train_idx, val_idx) in enumerate(zip(train_idx_list, val_idx_list)):

        train_image_ids = set(groups[train_idx])
        val_image_ids = set(groups[val_idx])

        # 학습/검증 데이터에 포함된 주석과 이미지를 필터링
        train_annotations = [ann for ann in data['annotations'] if ann['image_id'] in train_image_ids]
        val_annotations = [ann for ann in data['annotations'] if ann['image_id'] in val_image_ids]

        # COCO 형식에 맞게 학습 및 검증 데이터 재구성
        train_coco = {
            "images": [img for img in data['images'] if img['id'] in train_image_ids],
            "annotations": train_annotations,
            "categories": data['categories']
        }
        val_coco = {
            "images": [img for img in data['images'] if img['id'] in val_image_ids],
            "annotations": val_annotations,
            "categories": data['categories']
        }

        # 각 파일 저장
        with open(f"{dir}/train_fold-{idx}.json", "w") as f_train:
            json.dump(train_coco, f_train)
            
        with open(f"{dir}/val_fold-{idx}.json", "w") as f_val:
            json.dump(val_coco, f_val)

    print(f"데이터셋이 StratifiedGroupKFold를 사용해 {n_splits-1}:{1}로 나누어졌습니다.")
    return folder