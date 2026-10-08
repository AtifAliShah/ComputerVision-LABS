# Lab 05: HOG-Based Industrial Defect Detection and Classification (NEU-DET, Colab Run)

**Course:** Computer Vision  
**Student ID:** FA23-BAI-004  
**GitHub Profile:** [AtifAliShah](https://github.com/AtifAliShah)  
**Repository:** [AtifAliShah / ComputerVision-LABS](https://github.com/AtifAliShah/ComputerVision-LABS)  

---

## 📌 Overview
This directory contains the Google Colab notebook, its executed outputs, and the generated figures for **Lab 05: HOG-Based Industrial Defect Detection and Classification**. All numbers below are taken directly from the executed notebook [`Lab_Task_05_HOG_NEU-DET_FA23_BAI_004.ipynb`](./Lab_Task_05_HOG_NEU-DET_FA23_BAI_004.ipynb).

Pipeline: `Image → Preprocessing (Gaussian blur + CLAHE) → HOG Feature Extraction → SVM / Random Forest → Defective / Non-Defective Decision`

The experiment covers:
- **Dataset loading and inspection:** NEU-DET, 1800 grayscale images, 6 classes × 300 images
- **Preprocessing:** grayscale, resize to 128×128, 3×3 Gaussian blur, CLAHE
- **HOG parameter study:** cell sizes 4×4, 8×8, 16×16 × orientations 6, 9, 12
- **Classifiers:** HOG + SVM and HOG + Random Forest (binary and 6-class)
- **Robustness:** brightness, Gaussian noise, rotation, blur
- **QC decision module:** prediction, confidence, ACCEPT / REJECT action

> **Note on the dataset:** NEU-DET contains **only defective samples** (no defect-free class). For the binary task, the `crazing` class was used as a proxy for "Non-Defective" (Class 0) and the other five classes as "Defective" (Class 1), giving 300 vs 1500 images. This is a limitation of the dataset, not a true normal-vs-defect setting. Both Kaggle slugs from the lab handout returned `403`, so the notebook automatically fell back to `kaustubhdikshit/neu-surface-defect-database` (same 1800 NEU-DET images).

---

## 📂 Directory Structure
```
HOG_NEU-DET_Colab_Results/
├── figures/
│   ├── task1_dataset_samples.png
│   ├── task2_hog_visualization.png
│   ├── task3_binary_confusion_matrices.png
│   ├── task3_multiclass_svm_confusion_matrix.png
│   ├── task7_qc_decision_demo_1.png
│   ├── task7_qc_decision_demo_2.png
│   └── task7_qc_decision_demo_3.png
├── Lab_Task_05_HOG_NEU-DET_FA23_BAI_004.ipynb   # Colab notebook with executed outputs
└── README.md
```

---

## 🚀 How to Run
1. Open [Google Colab](https://colab.research.google.com) and upload `Lab_Task_05_HOG_NEU-DET_FA23_BAI_004.ipynb`.
2. Use a CPU runtime (no GPU needed).
3. Run cells top to bottom (Cell 1 downloads the dataset; Cell 4 is the slowest at a few minutes).

**Setup:** 1800 images → 128×128 grayscale; stratified 80/20 split (1440 train / 360 test, `random_state=42`).

---

## 📊 Summary of Results

### 1. HOG Parameter Study (Binary, test set of 360 images)
| Model | Cell | Orientations | Feature Dim | Accuracy | Precision | Recall | F1-Score |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| SVM (Linear) | 4×4 | 6 | 23,064 | 0.8472 | 0.8769 | 0.9500 | 0.9120 |
| Random Forest | 4×4 | 6 | 23,064 | 0.8333 | 0.8333 | 1.0000 | 0.9091 |
| SVM (Linear) | 4×4 | 9 | 34,596 | 0.8500 | 0.8618 | 0.9767 | 0.9156 |
| Random Forest | 4×4 | 9 | 34,596 | 0.8333 | 0.8333 | 1.0000 | 0.9091 |
| SVM (Linear) | 4×4 | 12 | 46,128 | 0.8833 | 0.8886 | 0.9833 | 0.9335 |
| Random Forest | 4×4 | 12 | 46,128 | 0.8333 | 0.8333 | 1.0000 | 0.9091 |
| SVM (Linear) | 8×8 | 6 | 5,400 | 0.8556 | 0.9161 | 0.9100 | 0.9130 |
| Random Forest | 8×8 | 6 | 5,400 | 0.8611 | 0.8634 | 0.9900 | 0.9224 |
| SVM (Linear) | 8×8 | 9 | 8,100 | 0.8611 | 0.9032 | 0.9333 | 0.9180 |
| Random Forest | 8×8 | 9 | 8,100 | 0.8278 | 0.8362 | 0.9867 | 0.9052 |
| SVM (Linear) | 8×8 | 12 | 10,800 | 0.8944 | 0.9253 | 0.9500 | 0.9375 |
| Random Forest | 8×8 | 12 | 10,800 | 0.8361 | 0.8394 | 0.9933 | 0.9099 |
| SVM (Linear) | 16×16 | 6 | 1,176 | 0.8194 | 0.9181 | 0.8600 | 0.8881 |
| Random Forest | 16×16 | 6 | 1,176 | 0.9194 | 0.9274 | 0.9800 | 0.9530 |
| SVM (Linear) | 16×16 | 9 | 1,764 | 0.8333 | 0.9082 | 0.8900 | 0.8990 |
| Random Forest | 16×16 | 9 | 1,764 | 0.9111 | 0.9136 | 0.9867 | 0.9487 |
| SVM (Linear) | 16×16 | 12 | 2,352 | 0.8972 | 0.9369 | 0.9400 | 0.9384 |
| **Random Forest** | **16×16** | **12** | **2,352** | **0.9417** | **0.9401** | **0.9933** | **0.9660** |

- Higher orientation count (12) gave the best result for both cell sizes where SVM was strongest (4×4, 8×8, 16×16).
- The notebook selected **16×16 cells with 12 orientations** as the best configuration (best SVM F1 of 0.9384 on the sweep, also the best overall row). It is also the cheapest to compute (about 12 s vs 158 s for 4×4 / 12 orientations in the sweep) with only 2,352 features.

### 2. Final Classifier Comparison (16×16 cells, 12 orientations, Binary)
| Classifier | Accuracy | Precision | Recall | F1-Score |
|---|:---:|:---:|:---:|:---:|
| **HOG + SVM (RBF, C=10)** | **0.9833** | **0.9900** | **0.9900** | **0.9900** |
| HOG + Random Forest (200 trees) | 0.9389 | 0.9344 | 0.9967 | 0.9645 |

Per-class report (test set, 60 Non-Defective / 300 Defective):

| Classifier | Class | Precision | Recall | F1 |
|---|---|:---:|:---:|:---:|
| HOG + SVM | Non-Defective | 0.95 | 0.95 | 0.95 |
| HOG + SVM | Defective | 0.99 | 0.99 | 0.99 |
| HOG + Random Forest | Non-Defective | 0.97 | 0.65 | 0.78 |
| HOG + Random Forest | Defective | 0.93 | 1.00 | 0.96 |

The Random Forest looks good on overall accuracy but misses about 35% of the Non-Defective samples (recall 0.65), so it would reject many good products. The SVM is balanced across both classes.

### 3. Multi-Class (6 defect types, macro-averaged)
| Classifier | Accuracy | Precision | Recall | F1-Score |
|---|:---:|:---:|:---:|:---:|
| **HOG + SVM (RBF)** | **0.9222** | 0.9221 | 0.9222 | 0.9220 |
| HOG + Random Forest | 0.8750 | 0.8805 | 0.8750 | 0.8727 |

### 4. Robustness Analysis (binary, change vs the original test set)
| Condition | SVM Acc | SVM F1 | ΔAcc | ΔF1 | RF Acc | RF F1 | ΔAcc | ΔF1 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Original | 0.9833 | 0.9900 | 0.0000 | 0.0000 | 0.9389 | 0.9645 | 0.0000 | 0.0000 |
| Brightness ×0.6 | 0.9694 | 0.9819 | -0.0139 | -0.0081 | 0.9056 | 0.9464 | -0.0333 | -0.0181 |
| Brightness ×1.4 | 0.9417 | 0.9660 | -0.0417 | -0.0240 | 0.9111 | 0.9494 | -0.0278 | -0.0151 |
| Gaussian noise σ=15 | 0.4833 | 0.5507 | -0.5000 | -0.4393 | 0.5833 | 0.6739 | -0.3556 | -0.2906 |
| Gaussian noise σ=30 | 0.2222 | 0.1250 | -0.7611 | -0.8650 | 0.3278 | 0.3240 | -0.6111 | -0.6405 |
| Rotation 10° | 0.9667 | 0.9803 | -0.0167 | -0.0097 | 0.9083 | 0.9479 | -0.0306 | -0.0166 |
| Rotation 20° | 0.9444 | 0.9672 | -0.0389 | -0.0228 | 0.9083 | 0.9479 | -0.0306 | -0.0166 |
| Blur k=5 | 0.8472 | 0.9160 | -0.1361 | -0.0740 | 0.8333 | 0.9091 | -0.1056 | -0.0554 |
| Blur k=9 | 0.8333 | 0.9091 | -0.1500 | -0.0809 | 0.8333 | 0.9091 | -0.1056 | -0.0554 |

Key observations:
- **Brightness and rotation:** only a small drop (about 1–4 points), because CLAHE and HOG block normalization absorb most lighting changes.
- **Gaussian noise:** the most damaging condition. HOG is gradient-based, so random noise creates strong false gradients and accuracy falls to chance level or below. Denoising (Gaussian/median filter) before HOG would be needed in a real deployment.
- **Blur:** moderate to large drop (about 13–15 points for SVM) because blur removes the fine edges that HOG relies on.

### 5. QC Decision Module (sample output)
| Actual class | Prediction | Confidence | Action |
|---|---|:---:|---|
| patches | DEFECTIVE | 100.0% | REJECT PRODUCT |
| inclusion | DEFECTIVE | 100.0% | REJECT PRODUCT |
| crazing (proxy normal) | NON-DEFECTIVE | 91.0% | ACCEPT PRODUCT |

Example console output:
```
PRODUCT INSPECTION RESULT
Prediction: NON-DEFECTIVE
Confidence: 91.0%
Action: ACCEPT PRODUCT
```

---

## 🖼️ Figures
- `task1_dataset_samples.png`: one sample from each of the 6 NEU-DET classes.
- `task2_hog_visualization.png`: preprocessed images (top) and their HOG representation (bottom).
- `task3_binary_confusion_matrices.png`: confusion matrices for SVM and Random Forest.
- `task3_multiclass_svm_confusion_matrix.png`: 6-class SVM confusion matrix.
- `task7_qc_decision_demo_1.png` to `task7_qc_decision_demo_3.png`: images used in the QC decision demo.

---

## ⚠️ Limitations
- NEU-DET has no defect-free images, so "Non-Defective" is a proxy (`crazing`) and the binary split is imbalanced (300 vs 1500). Results should not be read as true normal-vs-defect performance.
- Single random 80/20 split with one seed; no cross-validation.
- The HOG parameter sweep used a Linear SVM for speed, while the final comparison used an RBF SVM.
- The system is not robust to image noise; denoising must be added before deployment.
- The webcam / real-time bonus challenge was not implemented.
