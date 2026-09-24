# Lab 03: Edge Detection Techniques and Their Impact on Classification Performance

**Student ID:** FA23-BAI-004  
**Course:** Computer Vision  
**GitHub Profile:** [AtifAliShah](https://github.com/AtifAliShah)  
**Repository:** [AtifAliShah / ComputerVision-LABS](https://github.com/AtifAliShah/ComputerVision-LABS)  

---

## 📌 Overview
This directory contains the complete source code, experimental pipelines, visualizations, and laboratory report for **Lab 03: Edge Detection Techniques and Their Impact on Classification Performance**.

The experiment compares:
- **First-Order Detectors:** Sobel ($G_x, G_y$, Magnitude), Prewitt
- **Second-Order Detectors:** Laplacian, Laplacian of Gaussian (LoG)
- **Multi-Stage Detector:** Canny Edge Detection
- **Noise & Preprocessing Sensitivity:** Gaussian & Salt-and-Pepper noise with Gaussian & Median filtering
- **Canny Parameter Tuning:** Threshold and kernel size analysis
- **Cross-Lab Evaluation:** Classification on **Set A (Raw)**, **Set B (Filtered)**, and **Set C (Edge)** across SVM, Random Forest, KNN, Simple CNN, and Transfer Learning CNN (ResNet-18)

---

## 📂 Directory Structure
```
Edge_Detection_Skin-Lesion/
├── data/
│   ├── raw/           # Original dermoscopy images (nv, mel, bkl)
│   ├── filtered/      # Gaussian filtered images (Lab 02 best filter)
│   └── edge/          # Canny edge maps (Lab 03 best detector)
├── figures/
│   ├── task1_comparative_edge_detection.png
│   ├── task1_first_order_components.png
│   ├── task2_noise_and_filtering_effects.png
│   ├── task3_canny_parameter_analysis.png
│   ├── task5_classifier_performance_comparison.png
│   ├── task6_confusion_matrices.png
│   └── task6_metric_comparison_barchart.png
├── Lab_Task_03_FA23_BAI_004.ipynb    # Complete runnable Jupyter Notebook (Colab compatible)
├── run_lab03.py                     # Autonomous execution script for Tasks 1-6
├── build_notebook.py                # Notebook generator utility
├── results.md                       # Comprehensive Technical Lab Report & Discussion Answers
├── table_1.json                     # Table 1: Effect of Noise and Preprocessing
├── table_2.json                     # Table 2: Canny Parameter Analysis
└── table_3.json                     # Table 3: Cross-Lab Classification Performance
```

---

## 🚀 How to Run

### Option 1: Run via Python CLI
```bash
python run_lab03.py
```
This executes all 6 tasks, trains the machine learning models, generates Tables 1, 2, and 3, and saves all high-resolution figures into the `figures/` directory.

### Option 2: Run via Jupyter Notebook
Launch Jupyter Notebook or Jupyter Lab:
```bash
jupyter notebook Lab_Task_03_FA23_BAI_004.ipynb
```
Or open in **Google Colab** by uploading `Lab_Task_03_FA23_BAI_004.ipynb`.

---

## 📊 Summary of Results

### Cross-Lab Classification Performance (Table 3)
| Model | Accuracy Raw (Lab 1) | Accuracy Filtered (Lab 2) | Accuracy Edge (Lab 3) | Precision | Recall | F1-Score | Training Time (s) | Inference Time (ms) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **SVM (RBF)** | 0.1538 | 0.1538 | 0.2308 | 0.0839 | 0.2308 | 0.1231 | 0.015 | 0.56 |
| **Random Forest** | 0.0769 | 0.0769 | 0.0769 | 0.0641 | 0.0769 | 0.0699 | 0.443 | 0.55 |
| **KNN (k=3)** | 0.1538 | 0.1538 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.003 | 1.85 |
| **CNN Model 1** | 0.2308 | 0.3077 | 0.2308 | 0.0769 | 0.2308 | 0.1154 | 0.947 | 2.17 |
| **CNN Model 2 (ResNet)** | 0.2308 | 0.1538 | 0.3846 | 0.1479 | 0.3846 | 0.2137 | 30.293 | 16.56 |

For complete discussion answers and detailed methodology, see [results.md](results.md).
