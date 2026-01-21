# Drone Localization Using Thermal Camera
Capstone Project Repository

Authors:
- Marcin Kaczmarek
- Szymon Mela
- Mateusz Mrowicki

Date:
January 2026

---

## 1. Project Overview

This repository contains the complete implementation of a drone localization pipeline based on thermal imagery, developed as part of an academic Capstone project.

The main objective of the project is to detect, track, and predict the motion of drones observed using a thermal camera, both on real-world datasets and within a simulated environment.

The project integrates the following components:
- YOLOv5-based drone detection
- Dataset preparation and format conversion
- Model retraining and evaluation
- DeepSORT multi-object tracking
- AirSim-based simulation and data acquisition
- Extended Kalman Filter (EKF) for trajectory prediction

A detailed, step-by-step operational manual is provided in the document:

CAPSTONE-MODULE-USER-GUIDE

This README describes the architecture and structure of the repository, while the User Guide explains how to use each module in practice.

---

## 2. Repository Organization

Main directories included in the repository:

- 1 - Visualize result of final model  
  Pretrained YOLOv5 weights used for inference and visualization.

- 4 - Converting matlab to yolo format  
  MATLAB scripts converting thermal videos and annotation files into YOLO format.

- 5 - Retraining dataset (university dataset with labels)  
  Curated label sets used for fine-tuning the detection model.

- 6 - Deepsort tracker  
  YOLOv5 and DeepSORT based multi-object tracking pipeline.

- 7 - AirSim configuration  
  Configuration files and scripts for Microsoft AirSim simulation.

- 8 - Extended Kalman Filter  
  EKF-based drone trajectory estimation and prediction.

- AirSim  
  Local AirSim-related resources.

- ALL TASKS  
  Weekly milestones, experiments, datasets, and evaluation results.

- CAPSTONE-MODULE-USER-GUIDE  
  Complete operational manual for the project.

---

## 3. Module Description

### Visualize Result of Final Model

Contains pretrained YOLOv5 weights (best.pt) used for inference only.
Allows fast visualization of detection results without retraining.

### Dataset Conversion (MATLAB to YOLO)

MATLAB scripts convert thermal video files and MATLAB annotation files into YOLO-compatible image frames and label files.
Required for training YOLOv5 using the Drone-detection-dataset (v1.0.0).

### Dataset Retraining and Labeling

Refined label sets prepared for fine-tuning the detection model on a university dataset.
Different folders correspond to specific experimental scenarios.

### Multi-Object Tracking (DeepSORT)

Detection and tracking pipeline combining YOLOv5 with DeepSORT.
Provides persistent object IDs, handles temporary detection loss, and supports CSV logging.

### AirSim Simulation Configuration

Scripts and configuration files for Microsoft AirSim.
Enable dual-drone simulation, thermal-style video recording, and telemetry logging.

### Extended Kalman Filter (EKF)

EKF-based tracker estimating drone motion from YOLO detections.
Predicts future positions and visualizes uncertainty using covariance ellipses.

### Coursework and Experimental Archive

Weekly milestones, experiments, intermediate datasets, and evaluation results created during the Capstone course.
Documents the full research and development process.

---

## 4. Usage Instructions

Detailed instructions for:
- Environment setup
- Dataset download
- Model training and retraining
- Detection and tracking
- AirSim simulation
- Extended Kalman Filter prediction

are provided in:

CAPSTONE-MODULE-USER-GUIDE.docx

This document is the primary operational reference.

---

## 5. Technologies Used

- Python 3.10 / 3.11
- YOLOv5 (Ultralytics)
- PyTorch
- OpenCV
- DeepSORT
- MATLAB
- Microsoft AirSim
- NumPy

CUDA-enabled GPUs are supported but not required.

---

## 6. Academic Context and Reproducibility

The project was designed with:
- Modular architecture
- Explicit dataset handling
- Deterministic experiments using fixed random seeds
- Clear separation between detection, tracking, simulation, and prediction

The repository is suitable for academic evaluation, research extensions, and comparative tracking studies.

---

## 7. License

MIT License

Copyright (c) 2026  
Marcin Kaczmarek, Szymon Mela, Mateusz Mrowicki

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the Software), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED AS IS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## 8. Contact

For questions regarding this project, please refer to the authors listed above or consult the CAPSTONE-MODULE-USER-GUIDE.
