# Lab 05: HOG-Based Industrial Defect Detection and Classification

**Author:** Atif Ali Shah  
**Student ID:** FA23-BAI-004  
**Department:** Artificial Intelligence & Data Science  
**Course:** Computer Vision (Spring 2026)  
**Parent Repository:** [ComputerVision-LABS](https://github.com/AtifAliShah/ComputerVision-LABS)

---

## 📌 Executive Overview
This laboratory implements an **Automated Optical Inspection (AOI)** framework for real-time industrial surface defect detection on hot-rolled steel strips from the **NEU Surface Defect Database**. Using the **Histogram of Oriented Gradients (HOG)** descriptor paired with supervised learning classifiers, the system achieves **97.14% classification accuracy** and **2.50 ms inference latency**, supporting over 300 inspections per second on industrial conveyor belts.

---

## 📊 Experimental Results

### 1. Classifier Performance Comparison (Stratified 70/30 Split)
| Classifier | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | Inference Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear SVM (Optimal)** | **97.14** | **98.41** | **98.41** | **98.41** | **2.50 ms** |
| Random Forest (100 Trees) | 90.00 | 90.48 | 98.41 | 93.94 | 8.80 ms |
| K-Nearest Neighbors ($k=5$) | 72.86 | 96.30 | 72.22 | 80.89 | 3.10 ms |

### 2. HOG Hyperparameter Sweep
| Cell Size | Orientations | Feature Dim | Test Accuracy (%) | Extraction Latency (ms) |
| :---: | :---: | :---: | :---: | :---: |
| $4 \times 4$ | 9 | 34,596 | 97.14% | 14.80 ms |
| $8 \times 8$ | 9 | 8,100 | 97.14% | 5.20 ms |
| **$16 \times 16$ (Pareto Optimal)** | **9** | **1,764** | **100.00%** | **2.50 ms** |
| $8 \times 8$ | 6 | 5,400 | 95.71% | 4.60 ms |
| $8 \times 8$ | 12 | 10,800 | 97.14% | 6.10 ms |

### 3. Factory Environmental Perturbation Robustness
| Perturbation Condition | Accuracy (%) | $\Delta$ Accuracy | Robustness Verdict |
| :--- | :---: | :---: | :--- |
| **Pristine Baseline** | **97.14%** | **0.00%** | Reference Benchmark |
| Brightness ($1.5\times$ Overexposure) | 97.14% | 0.00% | **Completely Invariant** (L2-Hys Norm) |
| Conveyor Skew ($\pm 15^\circ$ Rotation) | 97.14% | 0.00% | **Completely Invariant** |
| Sensor Gaussian Noise ($\sigma = 20$) | 88.57% | -8.57% | Moderate Degradation |
| Optical Defocus Blur ($5\times 5, \sigma=1.5$) | 64.29% | -32.86% | **Severe Degradation** (Gradient Washout) |

---

## 🖼️ Visualizations & Artifacts
All figures are saved in the [`figures/`](figures/) directory:
- `task1_surface_defect_samples.png`: NEU surface defect morphology vs control.
- `task3_hog_feature_visualizations.png`: HOG gradient vector fields across defects.
- `task4_classifier_confusion_matrices.png`: Confusion matrices comparing SVM, RF, and KNN.
- `task5_hog_parameter_comparison.png`: Parameter trade-off plots (accuracy vs feature length).
- `task6_robustness_perturbation_tests.png`: Robustness degradation curves under stress.
- `task7_qc_decision_telemetry.png`: Live quality control decision telemetry HUD.

---

## 🚀 How to Run

### 1. Run Complete Laboratory Pipeline
```bash
python run_lab05.py
```

### 2. Run Real-Time Conveyor Video Inspection Simulation
```bash
python webcam_defect_detector.py
```

### 3. Open Jupyter Notebook
```bash
jupyter notebook Lab_Task_05_FA23_BAI_004.ipynb
```

---

## 📄 Academic Research Paper
See [`RESEARCH_PAPER.md`](RESEARCH_PAPER.md) for the full 12-section IEEE-format academic paper:
*"Histogram of Oriented Gradients, Classical Classifiers, and Environmental Perturbations for Industrial Surface Defect Detection: An Efficiency-Aware Comparative Benchmark on the NEU Surface Defect Database"*.
