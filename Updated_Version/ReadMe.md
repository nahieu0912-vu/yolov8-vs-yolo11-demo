# YOLO Training Data Project - Update Version

This repository contains the updated YOLO object detection project, featuring optimized datasets and multiple training configurations to improve model accuracy.

## Project Information
* **Author:** Hieu Vu
* **Roboflow Project:** [hieus-yolodemo on Roboflow Universe](https://roboflow.com)

## What I Upgraded

### 1. Fixed Training Dataset
* Used original source video frames.
* Standardized image size to 640x640 pixels.

### 2. Data Augmentation
* Applied robust data augmentation techniques.
* Enhanced model generalization to reduce overfitting.

### 3. Increased Epochs
* Extended training cycles.
* Allowed the model more time to converge.

### 4. Multiple YOLO Versions
* Trained models using both pre-trained weights and from scratch.
* Evaluated performance variations across different model sizes.

## Model Weights Directory Guide

All trained model weights were uploaded from the `runs/detect/` directory structure:

* **`train/weights/`**
  * Model: YOLOv8n (Nano)
  * Configuration: Trained from **pre-trained** weights
* **`train-2/weights/`**
  * Model: YOLOv8n (Nano)
  * Configuration: Trained from **scratch**
* **`train-3/weights/`**
  * Model: YOLOv8m (Medium)
  * Configuration: Trained from **scratch**
