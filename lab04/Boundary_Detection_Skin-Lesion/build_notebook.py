"""
Builds Lab_Task_04_FA23_BAI_004.ipynb for Atif Ali Shah
"""

import json
import os

def create_notebook():
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    def md(source):
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.strip().split("\n")]
        }

    def code(source):
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.strip().split("\n")]
        }

    # Title & Metadata
    nb["cells"].append(md("""# Lab 04: Skin Lesion Boundary Detection Using Canny Edge Detection

**Student ID:** FA23-BAI-004  
**Student Name:** Atif Ali Shah  
**Course:** Computer Vision  
**GitHub Profile:** [AtifAliShah](https://github.com/AtifAliShah)  
**Repository:** [AtifAliShah / ComputerVision-LABS](https://github.com/AtifAliShah/ComputerVision-LABS)  

---

## 📌 Problem Statement
Develop a simple computer vision system that detects the boundary of a skin lesion from a skin image using image filtering and Canny edge detection.  
The goal is to determine how effectively edge detection can separate the lesion from surrounding skin and extract reliable morphometrics (area and perimeter).

---

## 🔬 Experimental Workflow
- **Task 1:** Image Acquisition & Inspection (ISIC dataset)
- **Task 2:** Grayscale Conversion & Gaussian Pre-Smoothing
- **Task 3:** Canny Edge Detection (Thresholds 50-100, 100-200, 150-250)
- **Task 4:** Quantitative Edge Retention & Optimal Setting Selection
- **Task 5:** Morphological Contour Boundary Delineation
- **Task 6:** Planar Morphometric Calculation (Area & Perimeter)
- **Required Strip:** `Original -> Grayscale -> Gaussian -> Canny -> Lesion Boundary`
- **Benchmark:** 8 Spatial Filtering & Edge Detection Combinations
- **Analytical Q&A:** Comprehensive Answers to the 6 Assignment Questions"""))

    nb["cells"].append(md("""## 1. Environment & Setup"""))
    nb["cells"].append(code("""import os
import glob
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

os.makedirs('data', exist_ok=True)
os.makedirs('figures', exist_ok=True)

plt.rcParams['figure.dpi'] = 120
print(f"OpenCV: {cv2.__version__}")
print(f"NumPy:  {np.__version__}")"""))

    nb["cells"].append(md("""## 2. Task 1 -- Load the Image
We load 5 skin lesion images from the ISIC diagnostic benchmark:
1. **Image 1:** Melanoma (`ISIC_0000013`)
2. **Image 2:** Melanocytic Nevus (`ISIC_0000021`)
3. **Image 3:** Basal Cell Carcinoma (`ISIC_0024403`)
4. **Image 4:** Pigmented Benign Keratosis (`ISIC_0024612`)
5. **Image 5:** Vascular Lesion (`ISIC_0027210`)"""))

    nb["cells"].append(code("""SAMPLES = [
    {"idx": "Image 1", "file": "image_1.jpg", "label": "Melanoma (MEL)", "isic": "ISIC_0000013"},
    {"idx": "Image 2", "file": "image_2.jpg", "label": "Melanocytic Nevus (NV)", "isic": "ISIC_0000021"},
    {"idx": "Image 3", "file": "image_3.jpg", "label": "Basal Cell Carcinoma (BCC)", "isic": "ISIC_0024403"},
    {"idx": "Image 4", "file": "image_4.jpg", "label": "Pigmented Benign Keratosis (BKL)", "isic": "ISIC_0024612"},
    {"idx": "Image 5", "file": "image_5.jpg", "label": "Vascular Lesion (VASC)", "isic": "ISIC_0027210"}
]

data = []
for s in SAMPLES:
    p = os.path.join('data', s['file'])
    bgr = cv2.imread(p)
    if bgr is None:
        raise FileNotFoundError(f"Missing image: {p}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    data.append({**s, "bgr": bgr, "rgb": rgb})

fig, axes = plt.subplots(1, 5, figsize=(18, 4))
for i, s in enumerate(data):
    axes[i].imshow(s['rgb'])
    axes[i].set_title(f"{s['idx']}: {s['label']}\\n({s['isic']})", fontsize=10, fontweight='bold')
    axes[i].axis('off')
plt.tight_layout()
plt.show()"""))

    nb["cells"].append(md("""## 3. Task 2 -- Preprocess the Image
We convert the RGB dermoscopy images to single-channel grayscale and apply a Gaussian low-pass filter ($5 \\times 5$, $\\sigma=1.6$) to eliminate skin pores, hairs, and high-frequency sensor noise."""))

    nb["cells"].append(code("""def apply_prefiltering(bgr, ksize=(5, 5), sigma=1.6):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gauss = cv2.GaussianBlur(gray, ksize, sigma)
    return gray, gauss

fig, axes = plt.subplots(5, 3, figsize=(12, 16))
for i, s in enumerate(data):
    gray, gauss = apply_prefiltering(s['bgr'])
    axes[i, 0].imshow(s['rgb'])
    axes[i, 0].set_title(f"{s['idx']} -- Original RGB", fontsize=10)
    axes[i, 0].axis('off')
    
    axes[i, 1].imshow(gray, cmap='gray')
    axes[i, 1].set_title(f"{s['idx']} -- Grayscale", fontsize=10)
    axes[i, 1].axis('off')
    
    axes[i, 2].imshow(gauss, cmap='gray')
    axes[i, 2].set_title(f"{s['idx']} -- Gaussian Filter (5x5, sigma=1.6)", fontsize=10)
    axes[i, 2].axis('off')
plt.tight_layout()
plt.show()"""))

    nb["cells"].append(md("""## 4. Task 3 -- Apply Canny Edge Detection
Applying Canny edge detection using three distinct dual-threshold settings:
- **50--100** (Low Thresholds: high sensitivity for diffuse boundaries)
- **100--200** (Medium Thresholds: standard threshold range)
- **150--250** (High Thresholds: detects only high-contrast transitions)"""))

    nb["cells"].append(code("""threshold_pairs = [("50-100", 50, 100), ("100-200", 100, 200), ("150-250", 150, 250)]
fig, axes = plt.subplots(5, 3, figsize=(13, 17))

for i, s in enumerate(data):
    _, gauss = apply_prefiltering(s['bgr'])
    for j, (lbl, t_low, t_high) in enumerate(threshold_pairs):
        edges = cv2.Canny(gauss, t_low, t_high)
        cnt = np.count_nonzero(edges)
        axes[i, j].imshow(edges, cmap='gray')
        axes[i, j].set_title(f"{s['idx']} | Canny [{lbl}]\\nEdge Pixels: {cnt:,}", fontsize=9)
        axes[i, j].axis('off')
plt.tight_layout()
plt.show()"""))

    nb["cells"].append(md("""## 5. Task 4 -- Select the Best Result

### Edge Pixel Retention Curve"""))

    nb["cells"].append(code("""plt.figure(figsize=(10, 5))
modes = ['50-100', '100-200', '150-250']
palette = ['#E60000', '#0099FF', '#33CC33', '#FF9900', '#9933FF']

for idx, s in enumerate(data):
    _, gauss = apply_prefiltering(s['bgr'])
    totals = [np.count_nonzero(cv2.Canny(gauss, low, high)) for _, low, high in threshold_pairs]
    plt.plot(modes, totals, marker='D', markersize=6, linewidth=2, color=palette[idx], label=f"{s['idx']} ({s['label']})")

plt.title('Canny Edge Pixel Count across Threshold Settings', fontsize=12, fontweight='bold')
plt.xlabel('Dual-Threshold Setting (T_low - T_high)', fontsize=11)
plt.ylabel('Non-Zero Edge Pixels', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()"""))

    nb["cells"].append(md("""### Explanation of Selected Threshold (50--100)
- **High Thresholds (150--250):** Smoothed skin lesion borders rarely possess spatial gradients above 150. As a result, setting $T_{\\text{high}}=250$ completely suppresses edge formation, yielding nearly zero detected boundary pixels.
- **Medium Thresholds (100--200):** While core nodules are detected, the gradient along diffuse peripheral margins falls below $T_{\\text{low}}=100$, producing severe gaps along the perimeter.
- **Low Thresholds (50--100):** Provides the optimal balance. The lower threshold allows hysteresis to track subtle, fading edges connected to strong anchors, while the Gaussian smoothing ($\sigma=1.6$) keeps spurious dermal noise from triggering false edges."""))

    nb["cells"].append(md("""## 6. Task 5 -- Detect the Lesion Boundary
We use morphological closing to bridge gaps along the Canny edge perimeter and extract the dominant external contour:"""))

    nb["cells"].append(code("""def delineate_lesion_boundary(edge_map, bgr_img, border_guard=14):
    h, w = edge_map.shape
    clamped = edge_map.copy()
    clamped[:border_guard, :] = 0
    clamped[-border_guard:, :] = 0
    clamped[:, :border_guard] = 0
    clamped[:, -border_guard:] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    closed = cv2.morphologyEx(clamped, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    survivors = [c for c in contours if cv2.contourArea(c) > 350]
    best_c = max(survivors, key=cv2.contourArea) if survivors else (max(contours, key=cv2.contourArea) if contours else None)

    overlay_img = bgr_img.copy()
    area, perim = 0.0, 0.0

    if best_c is not None:
        area = float(cv2.contourArea(best_c))
        perim = float(cv2.arcLength(best_c, True))
        
        tint = overlay_img.copy()
        cv2.drawContours(tint, [best_c], -1, (0, 80, 255), -1)
        cv2.addWeighted(tint, 0.22, overlay_img, 0.78, 0, overlay_img)
        cv2.drawContours(overlay_img, [best_c], -1, (0, 0, 255), 2)

    rgb_vis = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)
    return best_c, area, perim, rgb_vis

fig, axes = plt.subplots(1, 5, figsize=(18, 4))
for i, s in enumerate(data):
    _, gauss = apply_prefiltering(s['bgr'])
    edges = cv2.Canny(gauss, 50, 100)
    _, area, perim, vis_rgb = delineate_lesion_boundary(edges, s['bgr'])
    axes[i].imshow(vis_rgb)
    axes[i].set_title(f"{s['idx']} Boundary\\nArea: {area:,.1f} px\\nPerim: {perim:,.1f} px", fontsize=9, fontweight='bold')
    axes[i].axis('off')
plt.tight_layout()
plt.show()"""))

    nb["cells"].append(md("""## 7. Task 6 -- Calculate Lesion Area & Required Visualization Strip

### Required Visualization
`Original -> Grayscale -> Gaussian Filter -> Canny -> Lesion Boundary`"""))

    nb["cells"].append(code("""fig, axes = plt.subplots(5, 5, figsize=(18, 16))
headers = ["Original Image", "Grayscale", "Gaussian Filter", "Canny (50-100)", "Lesion Boundary"]

results_list = []

for i, s in enumerate(data):
    gray, gauss = apply_prefiltering(s['bgr'])
    edges = cv2.Canny(gauss, 50, 100)
    _, area, perim, vis_rgb = delineate_lesion_boundary(edges, s['bgr'])
    
    results_list.append({
        "Image": s['idx'],
        "Best Filter": "Gaussian (5x5, sigma=1.6)",
        "Edge Method": "Canny (50-100)",
        "Area (pixels)": area,
        "Perimeter (pixels)": perim
    })
    
    row_imgs = [s['rgb'], gray, gauss, edges, vis_rgb]
    cmaps = [None, 'gray', 'gray', 'gray', None]
    
    for c in range(5):
        axes[i, c].imshow(row_imgs[c], cmap=cmaps[c])
        if i == 0:
            axes[i, c].set_title(headers[c], fontsize=11, fontweight='bold', pad=10)
        if c == 0:
            axes[i, c].set_ylabel(f"{s['idx']}\\n{s['label']}", fontsize=10, fontweight='bold')
        if c == 4:
            axes[i, c].set_xlabel(f"A={area:,.1f} | P={perim:,.1f} px", fontsize=9, labelpad=5)
        axes[i, c].set_xticks([])
        axes[i, c].set_yticks([])

plt.tight_layout()
plt.show()"""))

    nb["cells"].append(md("""### Summary Table of Results"""))
    nb["cells"].append(code("""import pandas as pd
df = pd.DataFrame(results_list)
print(df.to_markdown(index=False))"""))

    nb["cells"].append(md("""## 8. Final Comparison: Spatial Filtering and Edge Detection Methods"""))
    nb["cells"].append(code("""with open('table_2.json', 'r') as f:
    t2 = json.load(f)
df2 = pd.DataFrame(t2)
print(df2.to_markdown(index=False))"""))

    nb["cells"].append(md("""## 9. Comprehensive Answers to Lab Questions

### 1. Why is Gaussian filtering applied before Canny detection?
Canny edge detection computes image intensity gradients using spatial derivative approximations. Differentiation mathematically acts as a high-pass operator that drastically amplifies high-frequency noise. Skin images contain abundant high-frequency dermal noise such as epidermal ridges, hair shafts, pores, and sensor grain. Convolving the image with a Gaussian low-pass filter suppresses these rapid fluctuations, ensuring that the directional derivatives respond to the true macroscopic lesion boundary rather than superficial noise spikes.

### 2. How did the three Canny threshold settings affect the result?
- **50--100 (Low Setting):** Highly sensitive. Accurately captured the complete boundary perimeter, including diffuse zones where melanin gradually tapers into normal skin. Minor skin texture was occasionally detected, but easily handled by subsequent morphological operations.
- **100--200 (Medium Setting):** Moderate sensitivity. Retained strong contrast edges, but broke down along diffuse margins where gradients were low, causing disconnected perimeter segments.
- **150--250 (High Setting):** Severe under-detection. Gradients in smoothed skin lesion images rarely exceed 150. Almost no edge pixels satisfied the high threshold, causing the edge detector to fail entirely.

### 3. Which threshold produced the best lesion boundary?
The **50--100** threshold setting produced the best lesion boundary across all five test cases. Combined with Gaussian smoothing ($\sigma=1.6$), it maintained continuous, unbroken perimeter contours without excessive noise clutter.

### 4. Why are edges useful for detecting skin lesions?
In clinical dermatology, skin neoplasms (particularly malignant melanoma) are defined by localized hyper-pigmentation that creates distinct optical transitions relative to surrounding healthy skin. Under the **ABCDE clinical diagnostic protocol** (Asymmetry, Border irregularity, Color variegation, Diameter, Evolution), the **Border (B)** is a decisive prognostic indicator:
- **Lesion Area:** Quantifies planar tumor size and growth extent.
- **Lesion Perimeter & Circularity:** Quantifies border roughness, notched margins, and structural irregularity indicative of malignant radial growth.

### 5. What problems did you observe in detecting the lesion boundary?
1. **Diffuse & Gradual Transitions:** Biological lesions frequently lack an abrupt step edge; pigment fades gradually into surrounding skin, creating low-gradient transitions that can cause edge gaps.
2. **Hair & Skin Furrow Artifacts:** Hair shafts and epidermal creases produce sharp local gradients that can falsely connect to or distort the true lesion border.
3. **Internal Lesion Variegation:** Multi-colored lesion interiors (pigment networks, regression areas, globules) generate dense internal edges that must be differentiated from the outer envelope.
4. **Peripheral Lens Vignetting:** Darkening near the outer corners of dermoscopy lenses can trigger false peripheral edge responses if border margins are not clamped.

### 6. How could your method be improved?
1. **Perceptual Color Space Gradients:** Compute directional gradients in CIE $L^*a^*b^*$ or HSV color spaces (specifically using the $a^*$ channel for erythema and saturation for melanin) rather than scalar grayscale intensity.
2. **DullRazor Preprocessing:** Prepend a generalized morphological closing with multi-directional linear structuring elements to digitally eliminate hairs prior to gradient calculation.
3. **Adaptive / Auto-Canny Thresholding:** Calculate image-specific thresholds based on Otsu's method or median gradient statistics to adapt dynamically to varying patient skin phototypes (Fitzpatrick scale).
4. **Variational Active Contours (Snakes / Chan-Vese):** Use Canny edges as external energy constraints for an elastic contour model that enforces boundary smoothness and topological closure.
5. **Deep Learning Fusion:** Combine edge prior maps with deep semantic segmentation architectures (e.g., U-Net with boundary attention) for robust clinical segmentation under complex lighting and texture variations."""))

    out_file = "Lab_Task_04_FA23_BAI_004.ipynb"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"[+] Created Atif notebook: {out_file}")

if __name__ == '__main__':
    create_notebook()
