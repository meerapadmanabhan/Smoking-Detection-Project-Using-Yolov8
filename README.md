# Smoking-Detection-Project-Using-Yolov8
The smoking detection project was an excellent example of how new technologies can be harnessed to address public health issues.
<br/>
![abc001](https://github.com/meerapadmanabhan/Smoking-Detection-Project-Using-Yolov8/assets/94631005/ec6e516a-1cb5-4d53-bb77-1eead67a3b31) <br/>

## Table of Contents
1. Introduction
2. Data Source
3. Data Annotation
4. Model Building
5. Results
6. Inference
## Introduction
The smoking detection project was an excellent example of how new technologies can be harnessed to address public health issues. The primary goal was to create a robust system that could monitor public spaces and identify instances of smoking to enforce smoking bans and promote healthier environments.
## Data Source
The smoker detection dataset obtained from Mendeley consists of 1120 images. You can find more information about it [here](https://data.mendeley.com/datasets/j45dj8bgfc/1).
## Data Annotation
The data annotation for the Smoking class was performed by [Roboflow](https://roboflow.com/). The dataset is divided into training and validation sets with an 80:20 ratio.
## Model Building
To create a model in Google Colab that detects instances of the 'Smoking' class using the YOLOv8 architecture for object segmentation, we need to
### Creating Virtual Environement:
First, change the runtime to T4 GPU and connect it. Then, check to ensure that the GPU is properly configured and available for use.
```
!nvidia-smi
```
The following code is used to mount Google Drive in a Google Colab environment:

```
from google.colab import drive
drive.mount('/content/gdrive')
```

The following commands are used to navigate directories in Google Colab environment:

```
%cd yolov8
%cd custom_dataset
```

In the custom_dataset folder, we have training and validation datasets along with a data.yaml file. The data.yaml file typically contains information about the dataset, such as the number of classes, class names, and paths to the training and validation data.
![image](https://github.com/meerapadmanabhan/Smoking-Detection-Project-Using-Yolov8/assets/94631005/f54f6dfc-a1c3-4ee1-af6c-52d9f2de84e0) <br/>
<br/>
By navigating to this directory, we are setting up the environment to access and use these files for training the YOLOv8 model.
### Set Up Yolov8 Repository
The following command installs the ultralytics package:
```
!pip install ultralytics
```
### Train Yolov8 Model
The following command is used to train a YOLOv8 segmentation model:

```
!yolo task=segment mode=train epochs=100 data=data.yaml model=yolov8m-seg.pt imgsz=640 batch=16
```

This command initiates the training of a YOLOv8 segmentation model using the specified pre-trained model (yolov8m-seg.pt) on THE custom dataset defined in data.yaml, with the images resized to 640x640 pixels, and a batch size of 16 for 100 epochs.

After the training process, the best-performing model weights are saved as **best.pt** in the directory runs/segment/train/weights/. This best.pt file represents the model with the highest validation performance during training. To utilize this trained model for future inference or further training, it is common practice to copy best.pt and paste it into the custom_data folder, renaming it to yolov8m-seg-custom.pt. This ensures that the most optimized version of the model is readily accessible and identified within the custom dataset directory. in the directory runs/segment/train/weights/. This best.pt file represents the model with the highest validation performance during training. To utilize this trained model for future inference or further training, it is common practice to copy best.pt and paste it into the custom_data folder, renaming it to **yolov8m-seg-custom.pt**. This ensures that the most optimized version of the model is readily accessible and identified within the custom dataset directory.

### Testing on Images and Videos
![image](https://github.com/meerapadmanabhan/Smoking-Detection-Project-Using-Yolov8/assets/94631005/fef87f5b-6c77-4262-bf0f-a483229abe2a) <br/>
<br/>
The output will be saved in the 'exp' folder within the 'detect' directory under 'runs'.
### Results
Utilized evaluation metrics such as Confusion Matrix and F1 Curve.Achieved accuracy of detecting smoking as 70 % using Instance Segmentation.
![confusion_matrix_normalized](https://github.com/meerapadmanabhan/Smoking-Detection-Project-Using-Yolov8/assets/94631005/f12cc5ff-5162-4c35-ae97-6b9b3f569050)
![MaskF1_curve](https://github.com/meerapadmanabhan/Smoking-Detection-Project-Using-Yolov8/assets/94631005/b6a60f48-cb60-4555-bc29-7891efaee616)
### Inference
The confusion matrix results indicated a 70% accuracy. We need to improve this performance.

