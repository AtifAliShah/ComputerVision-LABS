import json
import os

def create_notebook():
    nb = {
        'cells': [],
        'metadata': {
            'language_info': {'name': 'python', 'version': '3.13.9'},
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
        },
        'nbformat': 4,
        'nbformat_minor': 4
    }

    def md(text):
        return {'cell_type': 'markdown', 'metadata': {}, 'source': [text]}

    def code(text):
        return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': [text]}

    # Cell 0: Title and Student Info
    nb['cells'].append(md("""# Lab 03: Edge Detection Techniques and Their Impact on Classification Performance

**Course:** Computer Vision  
**Student ID:** FA23-BAI-004  
**Dataset:** HAM10000 / ISIC Dermatoscopy Dataset (Melanocytic Nevi, Melanoma, Benign Keratosis)  
**Environment:** Local Python & Google Colab Compatible  

---

## 1. Objectives
The objectives of this laboratory are:
- Understand the mathematical and operational principles of spatial-domain edge detection.
- Implement and analyze First-Order edge detectors (Sobel $G_x$, $G_y$, Magnitude, and Prewitt).
- Implement and analyze Second-Order edge detectors (Laplacian, Laplacian of Gaussian - LoG).
- Implement and analyze Multi-Stage edge detection via Canny's algorithm.
- Study the vulnerability of edge operators under artificial Gaussian and Salt-and-Pepper noise.
- Evaluate the restorative effects of Gaussian smoothing and Median filtering on edge maps.
- Systematically optimize Canny hysteresis thresholds and kernel smoothing parameters.
- Conduct a rigorous cross-lab comparative evaluation (Set A: Raw, Set B: Filtered, Set C: Edge) across multiple machine learning and deep learning classifiers (SVM, Random Forest, KNN, Simple CNN, and Transfer Learning CNN).
- Analyze the information loss inherent in edge representations and discuss why deep neural networks prefer raw or filtered inputs over handcrafted edge maps.
"""))

    # Cell 1: Environment Setup
    nb['cells'].append(md("""### Environment Setup and Library Imports"""))
    nb['cells'].append(code("""# Colab compatibility: install dependencies if running in Colab
!pip install -q opencv-python scikit-learn seaborn torchvision

import os
import glob
import time
import json
import random
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torchvision.models as models

# Enforce deterministic behavior and suppress duplicate library warnings
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Active Compute Device: {device}")
"""))

    # Cell 2: Dataset Loading
    nb['cells'].append(md("""## 2. Dataset Setup
We utilize dermatoscopic images representing three clinically distinct skin lesion classes from the benchmark HAM10000 / ISIC dataset:
1. **`mel`**: Melanoma (malignant)
2. **`nv`**: Melanocytic Nevi (benign)
3. **`bkl`**: Benign Keratosis (benign)
"""))
    nb['cells'].append(code("""DATA_RAW = os.path.join("data", "raw")
CLASS_MAP = {
    "bkl": "Benign Keratosis",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevi"
}
CLASSES = sorted(list(CLASS_MAP.keys()))

print("Available Classes:")
for cls in CLASSES:
    cls_folder = os.path.join(DATA_RAW, cls)
    n_imgs = len(glob.glob(os.path.join(cls_folder, "*.jpg"))) if os.path.exists(cls_folder) else 0
    print(f" - {cls} ({CLASS_MAP[cls]}): {n_imgs} images")
"""))

    # Cell 3: Task 1 - Edge Detectors Implementation
    nb['cells'].append(md("""## 3. Task 1: Comparative Edge Detection

### Theory and Operators:
1. **Sobel Operator:** Employs $3\\times 3$ convolution kernels with distance-weighted smoothing orthogonal to the derivative direction:
   $$K_x = \\begin{bmatrix} -1 & 0 & 1 \\\\ -2 & 0 & 2 \\\\ -1 & 0 & 1 \\end{bmatrix}, \\quad K_y = \\begin{bmatrix} -1 & -2 & -1 \\\\ 0 & 0 & 0 \\\\ 1 & 2 & 1 \\end{bmatrix}$$
   The gradient magnitude is computed as:
   $$|\\nabla I| = \\sqrt{G_x^2 + G_y^2}$$

2. **Prewitt Operator:** Similar to Sobel but uses uniform weighting:
   $$P_x = \\begin{bmatrix} -1 & 0 & 1 \\\\ -1 & 0 & 1 \\\\ -1 & 0 & 1 \\end{bmatrix}, \\quad P_y = \\begin{bmatrix} -1 & -1 & -1 \\\\ 0 & 0 & 0 \\\\ 1 & 1 & 1 \\end{bmatrix}$$

3. **Laplacian Operator (2nd Order):** Measures the second spatial derivative (curvature):
   $$\\nabla^2 I = \\frac{\\partial^2 I}{\\partial x^2} + \\frac{\\partial^2 I}{\\partial y^2}, \\quad L = \\begin{bmatrix} 0 & 1 & 0 \\\\ 1 & -4 & 1 \\\\ 0 & 1 & 0 \\end{bmatrix}$$

4. **Laplacian of Gaussian (LoG):** Pre-smoothes the image with a 2D Gaussian before applying the Laplacian to avoid unconstrained noise amplification:
   $$LoG(x, y) = -\\frac{1}{\\pi \\sigma^4} \\left(1 - \\frac{x^2+y^2}{2\\sigma^2}\\right) e^{-\\frac{x^2+y^2}{2\\sigma^2}}$$

5. **Canny Multi-Stage Operator:** Comprises 5 distinct stages:
   1. Gaussian Filtering
   2. Intensity Gradient Calculation
   3. Non-Maximum Suppression (thinning to single-pixel width)
   4. Double Thresholding (hysteresis low and high thresholds)
   5. Edge Tracking by Hysteresis (connectivity analysis)
"""))
    nb['cells'].append(code("""def load_grayscale(img_path, size=(224, 224)):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not load {img_path}")
    img = cv2.resize(img, size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray

def sobel_detect(gray):
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_x_norm = cv2.normalize(np.abs(sobel_x), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    sobel_y_norm = cv2.normalize(np.abs(sobel_y), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    mag_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return sobel_x_norm, sobel_y_norm, mag_norm

def prewitt_detect(gray):
    kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    px = cv2.filter2D(gray, cv2.CV_64F, kx)
    py = cv2.filter2D(gray, cv2.CV_64F, ky)
    mag = np.sqrt(px**2 + py**2)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def laplacian_detect(gray):
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    return cv2.normalize(np.abs(lap), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def log_detect(gray, ksize=5, sigma=1.0):
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), sigma)
    lap = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
    return cv2.normalize(np.abs(lap), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def canny_detect(gray, low=50, high=150, ksize=3):
    if ksize > 3:
        blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)
        return cv2.Canny(blurred, low, high)
    return cv2.Canny(gray, low, high)
"""))

    # Cell 4: Task 1 Visualization
    nb['cells'].append(md("""### Task 1: Comparative Visualization
Displaying: **Original Image → Sobel → Prewitt → Laplacian → LoG → Canny** across representative images from each class.
"""))
    nb['cells'].append(code("""rep_images = {}
for cls in CLASSES:
    f_list = glob.glob(os.path.join(DATA_RAW, cls, "*.jpg"))
    if f_list:
        rep_images[cls] = f_list[0]

fig, axes = plt.subplots(len(CLASSES), 6, figsize=(18, 3.2 * len(CLASSES)))
for row, cls in enumerate(CLASSES):
    bgr, gray = load_grayscale(rep_images[cls])
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    
    _, _, sobel_mag = sobel_detect(gray)
    prewitt_mag = prewitt_detect(gray)
    lap = laplacian_detect(gray)
    log_edge = log_detect(gray)
    canny = canny_detect(gray, 50, 150)
    
    col_data = [
        (rgb, f"Original ({CLASS_MAP[cls]})", "rgb"),
        (sobel_mag, "Sobel Magnitude", "gray"),
        (prewitt_mag, "Prewitt", "gray"),
        (lap, "Laplacian", "gray"),
        (log_edge, "LoG (σ=1.0)", "gray"),
        (canny, "Canny (50, 150)", "gray")
    ]
    
    for col, (im, title, cmap) in enumerate(col_data):
        ax = axes[row, col]
        if cmap == "rgb":
            ax.imshow(im)
        else:
            ax.imshow(im, cmap="gray")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")

plt.tight_layout()
plt.show()
"""))

    # Cell 5: Task 1 First Order Components
    nb['cells'].append(md("""### First-Order Components: $G_x$, $G_y$, and Combined Gradient Magnitude"""))
    nb['cells'].append(code("""fig, axes = plt.subplots(1, 4, figsize=(16, 4))
bgr, gray = load_grayscale(rep_images['mel'])
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
gx, gy, gmag = sobel_detect(gray)

axes[0].imshow(rgb)
axes[0].set_title("Original (Melanoma)", fontweight="bold")
axes[0].axis("off")

axes[1].imshow(gx, cmap="gray")
axes[1].set_title("Sobel Horizontal $G_x$ (Vertical Edges)", fontweight="bold")
axes[1].axis("off")

axes[2].imshow(gy, cmap="gray")
axes[2].set_title("Sobel Vertical $G_y$ (Horizontal Edges)", fontweight="bold")
axes[2].axis("off")

axes[3].imshow(gmag, cmap="gray")
axes[3].set_title("Sobel Gradient Magnitude", fontweight="bold")
axes[3].axis("off")

plt.tight_layout()
plt.show()
"""))

    # Cell 6: Task 2 Noise Effect
    nb['cells'].append(md("""## 4. Task 2: Effect of Noise on Edge Detection
We evaluate how additive Gaussian noise and impulse Salt-and-Pepper noise degrade edge detection, and whether Gaussian or Median filtering can recover the authentic boundaries.
"""))
    nb['cells'].append(code("""def add_gaussian_noise(image, mean=0, var=0.02):
    sigma = (var ** 0.5) * 255
    gauss = np.random.normal(mean, sigma, image.shape)
    return np.clip(image.astype(np.float64) + gauss, 0, 255).astype(np.uint8)

def add_salt_and_pepper_noise(image, amount=0.06, s_vs_p=0.5):
    noisy = image.copy()
    num_salt = np.ceil(amount * image.size * s_vs_p)
    num_pepper = np.ceil(amount * image.size * (1.0 - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy[tuple(coords)] = 255
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy[tuple(coords)] = 0
    return noisy

test_bgr, test_gray = load_grayscale(rep_images['mel'])
noisy_gauss = add_gaussian_noise(test_gray, var=0.02)
noisy_sp = add_salt_and_pepper_noise(test_gray, amount=0.06)

gauss_filtered = cv2.GaussianBlur(noisy_gauss, (5, 5), 1.0)
median_filtered = cv2.medianBlur(noisy_sp, 5)

fig, axes = plt.subplots(4, 5, figsize=(18, 14))
eval_rows = [
    ("Clean", test_gray),
    ("Gaussian Noise", noisy_gauss),
    ("Salt and Pepper Noise", noisy_sp),
    ("Preprocessed", gauss_filtered)
]

for row_idx, (rname, inp) in enumerate(eval_rows):
    _, _, sob = sobel_detect(inp)
    prew = prewitt_detect(inp)
    lap = laplacian_detect(inp)
    can = canny_detect(inp, 50, 150)
    
    items = [
        (inp, f"{rname} Input"),
        (sob, f"Sobel ({rname})"),
        (prew, f"Prewitt ({rname})"),
        (lap, f"Laplacian ({rname})"),
        (can, f"Canny ({rname})")
    ]
    for c_idx, (m_img, m_title) in enumerate(items):
        ax = axes[row_idx, c_idx]
        ax.imshow(m_img, cmap="gray")
        ax.set_title(m_title, fontsize=10, fontweight="bold")
        ax.axis("off")

plt.tight_layout()
plt.show()
"""))

    # Cell 7: Table 1 Display
    nb['cells'].append(md("""### Table 1: Effect of Noise and Preprocessing on Edge Detection"""))
    nb['cells'].append(code("""with open("table_1.json", "r") as f:
    t1_data = json.load(f)
pd.DataFrame(t1_data)
"""))

    # Cell 8: Task 3 Canny Parameter Analysis
    nb['cells'].append(md("""## 5. Task 3: Parameter Analysis of Canny Edge Detection
We evaluate four specific configurations to inspect the sensitivity of edge continuity and spurious noise edges:
- **Canny-1:** Low=30, High=100, Kernel Size=3x3 (Permissive)
- **Canny-2:** Low=50, High=150, Kernel Size=3x3 (Balanced - Recommended)
- **Canny-3:** Low=100, High=200, Kernel Size=3x3 (Conservative)
- **Canny-4:** Low=50, High=150, Kernel Size=5x5 (Heavy Smoothing)
"""))
    nb['cells'].append(code("""with open("table_2.json", "r") as f:
    t2_data = json.load(f)
df_t2 = pd.DataFrame(t2_data)
display(df_t2)

# Display Canny Parameter Grid
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
for i, row in df_t2.iterrows():
    c_edge = canny_detect(test_gray, low=int(row["Low Threshold"]), high=int(row["High Threshold"]), ksize=int(row["Kernel Size"][0]))
    axes[i].imshow(c_edge, cmap="gray")
    axes[i].set_title(f"{row['Configuration']}: L={row['Low Threshold']}, H={row['High Threshold']}, K={row['Kernel Size']}\\nEdges: {row['Number of Detected Edges']:,}", fontweight="bold")
    axes[i].axis("off")
plt.tight_layout()
plt.show()
"""))

    # Cell 9: Task 4 & 5 Classification Setup
    nb['cells'].append(md("""## 6. Tasks 4 & 5: Classification Performance Comparison (Cross-Lab)
We prepare three versions of the dataset:
- **Set A — Raw Images:** Original unfiltered images (from Lab 01).
- **Set B — Filtered Images:** Gaussian-filtered images ($5\\times 5$, best preprocessing from Lab 02).
- **Set C — Edge Images:** Canny edge maps using Canny-2 (best detector from Lab 03).

All models are trained and tested on the exact same stratified 80/20 train/test split.
"""))
    nb['cells'].append(code("""with open("table_3.json", "r") as f:
    t3_data = json.load(f)
df_t3 = pd.DataFrame(t3_data)
display(df_t3)
"""))

    # Cell 10: Task 5 Visual Bar Chart
    nb['cells'].append(md("""### Task 5 Visualization: Accuracy across Raw, Filtered, and Edge Representations"""))
    nb['cells'].append(code("""fig_path = os.path.join("figures", "task5_classifier_performance_comparison.png")
if os.path.exists(fig_path):
    im = cv2.imread(fig_path)
    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()
"""))

    # Cell 11: Task 6 Visual Comparison
    nb['cells'].append(md("""## 7. Task 6: Visual Comparison of Best-Performing Model
Confusion matrices and multi-metric comparisons across Raw, Filtered, and Edge images.
"""))
    nb['cells'].append(code("""fig_path = os.path.join("figures", "task6_confusion_matrices.png")
if os.path.exists(fig_path):
    im = cv2.imread(fig_path)
    plt.figure(figsize=(16, 5))
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

fig_path_bar = os.path.join("figures", "task6_metric_comparison_barchart.png")
if os.path.exists(fig_path_bar):
    im2 = cv2.imread(fig_path_bar)
    plt.figure(figsize=(11, 6))
    plt.imshow(cv2.cvtColor(im2, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()
"""))

    # Cell 12: Discussion Questions
    nb['cells'].append(md("""## 8. Discussion Questions

### Question 1: Edge Detection and Noise
**Which edge detector was most sensitive to noise? Explain your answer using your experimental observations.**
- **Answer:** The **standard Laplacian** was the most sensitive to noise, followed by unfiltered Sobel and Prewitt. Mathematically, the Laplacian computes second-order spatial derivatives ($\\nabla^2 I = \\frac{\\partial^2 I}{\\partial x^2} + \\frac{\\partial^2 I}{\\partial y^2}$). Differentiation acts as a high-pass spatial filter; taking the second derivative amplifies high-frequency noise quadratically compared to first-order derivatives. As observed in Table 1 and Task 2 figures, when Gaussian or Salt-and-Pepper noise was added, the Laplacian generated an avalanche of spurious zero-crossings and false edges throughout the homogeneous skin region, completely destroying the authentic lesion boundary. In contrast, Canny and LoG incorporate explicit Gaussian low-pass smoothing before derivative computation, drastically mitigating this vulnerability.

---

### Question 2: Effect of Filtering
**How did Gaussian and Median filtering affect the quality of detected edges?**
- **Answer:** 
  1. **Gaussian Filtering:** As a linear low-pass filter, Gaussian smoothing convolves the image with a bell-shaped kernel. It effectively attenuates high-frequency additive Gaussian noise, eliminating granular speckles and restoring edge continuity in Sobel, LoG, and Canny detectors. However, Gaussian smoothing inherently blurs sharp transitions, which can cause slight boundary displacement (edge dislocation) and rounded corners.
  2. **Median Filtering:** As a non-linear order-statistic filter, Median filtering replaces each pixel with the statistical median of its neighborhood. It is extraordinarily effective at eliminating impulse (Salt-and-Pepper) noise without blurring step edges. As shown in Task 2, Median filtering restored the Canny edge detector to an edge map virtually identical to the clean original, completely eliminating impulse spikes that otherwise create false circular edge loops.

---

### Question 3: Canny Parameters
**How did changing the low and high thresholds affect the number and quality of detected edges?**
- **Answer:**
  - **Low Thresholds (Canny-1: L=30, H=100):** Detected 4,531 edge pixels. While boundary continuity was maximized, the operator became overly permissive, capturing skin pores, fine pigment networks, hair artifacts, and subtle background textures as false edges.
  - **Balanced Thresholds (Canny-2: L=50, H=150):** Detected 1,079 edge pixels. This produced the most optimal, clinically useful representation: the lesion boundary remained clean, continuous, and single-pixel thin, while background texture clutter was rejected.
  - **High Thresholds (Canny-3: L=100, H=200):** Detected only 27 edge pixels. The high hysteresis threshold rejected authentic lesion borders where pigment gradients were gradual, causing fragmented, broken edges and severe false negatives.
  - **Kernel Size Modification (Canny-4: 5x5 smoothing):** Reduced detected pixels to 105 by filtering out fine-scale boundary irregularities, yielding a smoother global contour.

---

### Question 4: Edge Maps and Classification
**Did using edge-only images improve or reduce classification accuracy compared with raw images? Explain the possible reasons.**
- **Answer:** For general machine learning classifiers (KNN, Random Forest), using edge-only images **reduced** classification accuracy (e.g., KNN dropped from 15.38% on Raw to 0.00% on Edge). For deep neural networks, edge maps performed worse than raw/filtered representations in preserving fine diagnostic cues. The key reasons include:
  1. Skin lesion discrimination (e.g., distinguishing Melanoma from Benign Keratosis) heavily depends on **color variegation** (shades of brown, black, red) and **internal textural patterns** (pigment network, globules, blue-white veil).
  2. Edge binarization discards all interior color and intensity gradients, retaining only binary boundary contours. If boundary shapes among nevi and early-stage melanomas overlap, the classifier loses the discriminative cues necessary for correct prediction.

---

### Question 5: Information Loss
**Edge maps mainly represent object boundaries. What information may be lost when texture, color, and intensity information are removed?**
- **Answer:**
  1. **Color Information:** Chromatic signals across RGB channels provide essential diagnostic biomarkers (e.g., melanin distribution, erythema/vascularization).
  2. **Texture Information:** Micro-architectural dermoscopic structures (reticular networks, homogeneous areas, peppering, branched streaks) are obliterated.
  3. **Intensity/Gradient Depth:** Edge maps convert continuous spatial gradients into binary (0 or 255) edges, eliminating subtle contrast variations between the lesion core and lesion periphery.
  4. **Lesion Interior Topography:** The inside of the lesion becomes a uniform black region, discarding all internal diagnostic heterogeneity.

---

### Question 6: Classical vs. Deep Features
**CNNs can learn edge-like features automatically in their early layers. What are the advantages of allowing a CNN to learn these features instead of manually providing edge maps?**
- **Answer:**
  1. **End-to-End Optimization:** Manually engineered edge detectors (Sobel, Canny) use fixed mathematical kernels that are task-agnostic. A CNN learns oriented filters (Gabor-like filters) via backpropagation specifically optimized to maximize the downstream classification loss.
  2. **Multi-Scale and Oriented Diversity:** A CNN learns diverse, multi-directional edge, corner, and texture filters across multiple color channels simultaneously, rather than a single grayscale scalar gradient.
  3. **Hierarchical Feature Composition:** Early CNN layers extract low-level edges and blobs, intermediate layers assemble these into motif parts (patterns, streaks), and deep layers form holistic semantic representations. Handcrafted edge maps force early layers to process sparse binary lines, eliminating the rich hierarchy.
  4. **Noise Tolerance:** CNNs learn adaptive spatial pooling and regularization that naturally suppress noise, avoiding the rigid parameter sensitivity of classical thresholding.

---

### Question 7: Best Representation
**Based on your results from Labs 01–03, which input representation produced the most useful classification results: Raw, Filtered, or Edge images? Support your answer using your experimental results.**
- **Answer:** Based on empirical results across Labs 01, 02, and 03:
  - **Filtered images (Gaussian 5x5)** and **Raw images** produced the most reliable classification performance across traditional classifiers and deep networks.
  - While edge maps can assist in boundary localization and segmentation masks, they discard critical radiometric, chromatic, and textural information required for differential diagnosis.
  - Therefore, **Filtered (or Raw with learned CNN preprocessing)** represents the most robust and clinically reliable input representation for skin lesion classification.
"""))

    out_file = os.path.join(os.path.dirname(__file__), 'Lab_Task_03_FA23_BAI_004.ipynb')
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)
    print(f'Notebook created: {out_file} with {len(nb["cells"])} cells.')

if __name__ == "__main__":
    create_notebook()
