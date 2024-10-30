# BoostCamp AI Tech 7th CV-01 Project

## Team Members
| <img src="https://github.com/user-attachments/assets/7d2414bd-dadf-48f0-a315-a5921d3d52fa" width="100"/> | <img src="https://github.com/user-attachments/assets/7f9949f9-703a-4e42-bd1b-5a5683ed24d2" width="100"/> | <img src="https://github.com/user-attachments/assets/b94ec0ce-873d-4fe0-b9d3-7fe6a3b37163" width="100"/> | <img src="https://github.com/user-attachments/assets/fd845c5d-08a1-4d92-950e-c54d3812662d" width="100"/> | <img src="https://github.com/user-attachments/assets/be384c7a-b7ea-4ce0-b1a1-7da4e8a9b6e5" width="100"/> | <img src="https://github.com/user-attachments/assets/ac1ddf74-ccdb-4170-869e-74177ca00633" width="100"/> |
|:----------------------------------------------------------:|:----------------------------------------------------------:|:----------------------------------------------------------:|:----------------------------------------------------------:|:----------------------------------------------------------:|:----------------------------------------------------------:|
| **동준**                                                     | **경윤**                                                      | **영석**                                                      | **태영**                                                      | **태성**                                                      | **세린**                                                      |

## Contribute

| Member | Roles |
|--------|-------|
| **동준** | EDA, Detectron2 k-fold, 빠른 테스트 위한 RT-DETR 선정, Class 세부 분리 실험, 결과 시각화 |
| **영석** | EDA, Streamlit으로 데이터 시각화 및 양상블, detectron2 Albumentation, mmdetection k-fold |
| **경윤** | EDA, 라이브러리 구조파악, Model 선택, 가설 설정 및 실험 |
| **태영** | EDA, 라이브러리 구조파악, Data 정제, Model 선택, 가설 설정 및 augmentation 실험 |
| **세린** | EDA, Detectron2 k-fold |
| **태성** | Model 선택, 템플릿 코드 작성, 가설 설정 및 실험, 결과 시각화 |


## Overview

This project is part of BoostCamp AI Tech and focuses on developing an object detection model specifically designed to identify and classify various types of trash. The primary goal is to enhance waste sorting efficiency by accurately detecting different categories of waste materials in images.

## Dataset

This project utilizes a dataset specifically designed for trash classification. The dataset contains labeled images of various types of waste, divided into training and test sets.

- **Training Images**: 4883
- **Test Images**: 4871
- **Classes**: 10 categories of trash items, with counts as follows:
  - **General trash**: 3966
  - **Paper**: 6352
  - **Paper pack**: 897
  - **Metal**: 936
  - **Glass**: 982
  - **Plastic**: 2943
  - **Styrofoam**: 1263
  - **Plastic bag**: 5178
  - **Battery**: 159
  - **Clothing**: 468

Each image in the dataset is annotated with bounding boxes for these categories to facilitate the training of an object detection model.

## Development Environment

| **Category**       | **Details**                        | **Category**       | **Details**            |
|--------------------|------------------------------------|--------------------|------------------------|
| **Hardware**       | GPU: V100 32GB × 4                | **Python**         | 3.10                   |
| **CUDA**           | 11.6                              | **PyTorch**        | 1.12                   |
| **PyTorch Lightning** | 1.8                           | **Libraries**      | MMDetection(3.3.0), Detectron2(0.6), Ultralytics(8.3.23) |
| **Collaboration Tools** | Notion, WandB               |                   |                        |


## Results

<div style="display: flex; flex-direction: column; align-items: center;">
    <div style="text-align: center; margin-bottom: 10px;">
        <img src="https://github.com/user-attachments/assets/431726a1-bf6c-4ff1-8096-3753a49469cd" width="500" />
    </div>
    <div style="text-align: center; margin-bottom: 10px;">
        <img src="https://github.com/user-attachments/assets/559bd795-3ff7-4d8a-afe8-53588c244b63" width="500" />
    </div>
    <div style="text-align: center; margin-bottom: 10px;">
        <img src="https://github.com/user-attachments/assets/edd341ae-72bc-486c-83ef-50697e075214" width="500" />
    </div>
    <div style="text-align: center; margin-bottom: 10px;">
        <img src="https://github.com/user-attachments/assets/6cf9891d-5646-4f3f-9852-542199bf8338" width="500" />
    </div>
    <div style="text-align: center; margin-bottom: 10px;">
        <img src="https://github.com/user-attachments/assets/5cf5c523-3fd2-4add-bab9-cca201e0b3a9" width="500" />
    </div>
    <div style="text-align: center; margin-bottom: 10px;">
        <img src="https://github.com/user-attachments/assets/32548ac5-55f4-4d99-a9e0-88bf0c38687d" width="500" />
    </div>
</div>



## Usage

<details>
  <summary id="mmdetection">MMDetection</summary>

  ### Data Preprocessing and Model Architecture
 - please refer to the configuration file in mmDetection/configs for details.
  ### Train & Test

  - To train the model, run the following command:
      ```bash
      python tools/train.py configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py
      ```
  - To test the model, run the following command:
    ```bash
      python tools/test.py --work-dir "save/dir" "path/pth"
    ```
  ### Additional Settings

  - see`mmdetection/configs/` to adjust various training and model parameters
</details>

<details>
  <summary id="detectron2">Detectron2</summary>

  ### Data Preprocessing

  - please refer to the configuration file in detectron2/configs for details.

  ### Model Architecture

  - Detectron2 provides pre-defined model architectures. Customize by modifying `config.yaml` and adding models to the `backbone` folder.

  ### Train & Test

  - Run the following command to train and test the model:
      ```bash
      python lightning_train_net_custom.py --config-file ../configs/COCO-Detection/faster_rcnn_R_101_FPN_3x_TRASH.yaml --num-gpus 1
      ```

</details>

<details>
  <summary id="ultralytics">Ultralytics</summary>

  ### Data Preprocessing

  - **Data Augmentation**  
    Use OpenCV-based augmentation methods and specify additional transformations in `select_transforms.py`.

  ### Model Architecture

  - Ultralytics models (like YOLOv5/YOLOv8) are configured using `ultralytics/config.yaml`. You can adjust the architecture by changing the model type and parameters in this file.

  ### Train & Test

  - Use the following command to start training and testing:
      ```bash
      python train.py --data data.yaml --cfg cfg/yolov5.yaml --weights weights/yolov5s.pt
      ```

  ### Additional Settings

  - Customize `ultralytics.yaml` for batch size, epochs, learning rate, and other parameters.
  
      ```yaml
      batch_size: 64
      epochs: 50
      data: ./data.yaml
      weights: yolov5s.pt
      imgsz: 640
      device: 0
      project: yolov5
      name: exp
      cache: True
      optimizer: SGD
      ```
      
</details>



## Tree Structure
![Structure](https://github.com/user-attachments/assets/4239ee19-a39b-4f92-a8f3-cc6cf9bccc02)

