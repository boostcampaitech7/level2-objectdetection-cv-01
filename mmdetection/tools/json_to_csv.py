import pandas as pd
import json

def create_prediction_dataframe(path,name_prefix=''):
    with open(path+'.bbox.json') as f:
        json_data = json.load(f)

    df = pd.DataFrame(json_data)

    def coco_to_pascal(bbox):
        x_min, y_min, width, height = bbox
        x_max = x_min + width
        y_max = y_min + height
        return [x_min, y_min, x_max, y_max]

    df['image_id'] = df['image_id'].apply(lambda x: f"test/{x:04d}.jpg")
    df['bbox'] = df['bbox'].apply(coco_to_pascal)  # bbox 변환

    grouped = df.groupby('image_id')

    prediction_strings = []
    image_ids = []

    for image_id, group in grouped:
        predictions = [
            f"{row['category_id']} {row['score']} " + ' '.join(map(str, row['bbox']))
            for _, row in group.iterrows()
        ]
        prediction_strings.append(' '.join(predictions))
        image_ids.append(image_id)

    result = pd.DataFrame({
        'PredictionString': prediction_strings,
        'image_id': image_ids
    })

    result[['PredictionString','image_id']].to_csv(path+name_prefix+'output.csv', index=False)
    return result[['PredictionString', 'image_id']]