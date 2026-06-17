# Yolo Training Data Project

**Author:** Hieu Vu  
**Roboflow Project Link:** [hieus-yolodemo on Roboflow Universe](https://universe.roboflow.com/hieus-yolodemo/yolo_traindata-eoiiu/model/3)

> ⚠️ **Important Project Note: Aspect Ratio & Accuracy Impact**  
> In the initial version of this project, the raw source video frames were uploaded and annotated before resizing. Applying the 640x640 **"Stretch to"** preprocessing step *after* completing the annotation process distorted the images. This stretching altered the original aspect ratios of the street objects, which negatively affected the model's overall detection accuracy.  
>  
> **Future Updates:** The next dataset version will resolve this issue. The source frames will be properly scaled to preserve their original aspect ratios before generating the YOLO training versions.

---

## About the Project
This project demonstrates how to prepare, augment, and train custom object detection models using both **YOLOv8** and **YOLOv11** architectures. 

The source dataset was built from video footage captured at my neighborhood street. The raw video frames were uploaded, annotated, and processed using the Roboflow platform. To meet the specific architectural requirements of the YOLO models and improve detection accuracy, the dataset was optimized with the following steps:
* **Resizing:** Scaled all images to a uniform 640x640 resolution.
* **Augmentations:** Applied brightness adjustments and horizontal flips to artificially expand the dataset and improve model robustness against diverse lighting and angles.

---

## Model Variations
The repository includes three distinct trained models used for benchmarking and performance comparisons:

1. **`best.pt`**  
   * **Environment:** Google Colab (Cloud GPU)
   * **Description:** The optimal weights file generated during cloud-based training sessions.
2. **`yolov8n.pt`**  
   * **Environment:** Local Computer
   * **Model Scale:** YOLOv8 Nano (`n`)
   * **Description:** A lightweight, fast-inference model trained locally for real-time tracking efficiency.
3. **`yolo11s.pt`**  
   * **Environment:** Local Computer
   * **Model Scale:** YOLOv11 Small (`s`)
   * **Description:** A highly accurate model leveraging the updated YOLOv11 architecture trained locally for superior feature extraction.

---

## File Structure & Usage

### 1. Model Training (`train_yolov8.ipynb`)
This Jupyter Notebook contains the complete step-by-step pipeline for model development. It covers:
* Authenticating and downloading the augmented dataset directly from Roboflow.
* Configuring the Ultralytics training environment.
* Executing the training loops for both YOLOv8 and YOLOv11 algorithms.
* Evaluating performance metrics (Loss curves, mAP50, and mAP50-95).

### 2. Inference & Video Processing (`video_object_detect.py`)
This standalone Python script demonstrates how to deploy your trained weights to a practical use case. 
* **Input:** A raw `.mp4` video file of the neighborhood street.
* **Process:** Loads the specified custom model (`best.pt`, `yolov8n.pt`, or `yolo11s.pt`) and runs frame-by-frame object detection.
* **Output:** Generates a real-time visual playback showing bounding boxes, class labels, and confidence percentages around detected street objects.
