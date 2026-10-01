"""
Lab 04: Skin Lesion Boundary Detection Using Canny Edge Detection
Student ID: FA23-BAI-004
Course: Computer Vision
GitHub: AtifAliShah (https://github.com/AtifAliShah)
Repository: ComputerVision-LABS/lab04/Boundary_Detection_Skin-Lesion
"""

import os
import json
import csv
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('figures', exist_ok=True)
os.makedirs('data', exist_ok=True)

IMAGE_RECORDS = [
    {
        "idx": "Image 1",
        "file": "image_1.jpg",
        "label": "Melanoma (MEL)",
        "isic": "ISIC_0000013",
        "pathology": "Malignant Melanoma"
    },
    {
        "idx": "Image 2",
        "file": "image_2.jpg",
        "label": "Melanocytic Nevus (NV)",
        "isic": "ISIC_0000021",
        "pathology": "Benign Melanocytic Nevus"
    },
    {
        "idx": "Image 3",
        "file": "image_3.jpg",
        "label": "Basal Cell Carcinoma (BCC)",
        "isic": "ISIC_0024403",
        "pathology": "Basal Cell Epithelioma"
    },
    {
        "idx": "Image 4",
        "file": "image_4.jpg",
        "label": "Pigmented Benign Keratosis (BKL)",
        "isic": "ISIC_0024612",
        "pathology": "Seborrheic / Lichenoid Keratosis"
    },
    {
        "idx": "Image 5",
        "file": "image_5.jpg",
        "label": "Vascular Lesion (VASC)",
        "isic": "ISIC_0027210",
        "pathology": "Hemangioma / Angiokeratoma"
    }
]

def load_skin_samples():
    loaded = []
    for item in IMAGE_RECORDS:
        p = os.path.join('data', item['file'])
        if not os.path.exists(p):
            raise FileNotFoundError(f"Sample missing: {p}")
        bgr = cv2.imread(p)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        loaded.append({**item, "bgr": bgr, "rgb": rgb})
    return loaded

def apply_prefiltering(bgr, ksize=(5, 5), sigma=1.6):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gauss = cv2.GaussianBlur(gray, ksize, sigma)
    return gray, gauss

def compute_canny_edges(gauss_img):
    return {
        "50-100": cv2.Canny(gauss_img, 50, 100),
        "100-200": cv2.Canny(gauss_img, 100, 200),
        "150-250": cv2.Canny(gauss_img, 150, 250)
    }

def delineate_lesion_boundary(edge_map, bgr_img, border_guard=14):
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
        
        # Coral / Crimson border
        tint = overlay_img.copy()
        cv2.drawContours(tint, [best_c], -1, (0, 80, 255), -1)
        cv2.addWeighted(tint, 0.22, overlay_img, 0.78, 0, overlay_img)
        cv2.drawContours(overlay_img, [best_c], -1, (0, 0, 255), 2)

    rgb_vis = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)
    return best_c, area, perim, rgb_vis

def run_all_visualizations(samples):
    # Task 1
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    for i, s in enumerate(samples):
        axes[i].imshow(s['rgb'])
        axes[i].set_title(f"{s['idx']}: {s['label']}\n({s['isic']})", fontsize=10, fontweight='bold')
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig('figures/task1_original_images.png', dpi=200, bbox_inches='tight')
    plt.close()

    # Task 2
    fig, axes = plt.subplots(5, 3, figsize=(12, 16))
    for i, s in enumerate(samples):
        gray, gauss = apply_prefiltering(s['bgr'])
        axes[i, 0].imshow(s['rgb'])
        axes[i, 0].set_title(f"{s['idx']} -- Original RGB", fontsize=10)
        axes[i, 0].axis('off')
        axes[i, 1].imshow(gray, cmap='gray')
        axes[i, 1].set_title(f"{s['idx']} -- Grayscale", fontsize=10)
        axes[i, 1].axis('off')
        axes[i, 2].imshow(gauss, cmap='gray')
        axes[i, 2].set_title(f"{s['idx']} -- Gaussian (5x5, sigma=1.6)", fontsize=10)
        axes[i, 2].axis('off')
    plt.tight_layout()
    plt.savefig('figures/task2_preprocessing.png', dpi=200, bbox_inches='tight')
    plt.close()

    # Task 3
    fig, axes = plt.subplots(5, 3, figsize=(13, 17))
    for i, s in enumerate(samples):
        _, gauss = apply_prefiltering(s['bgr'])
        regimes = compute_canny_edges(gauss)
        for j, (lbl, edge_img) in enumerate(regimes.items()):
            cnt = np.count_nonzero(edge_img)
            axes[i, j].imshow(edge_img, cmap='gray')
            axes[i, j].set_title(f"{s['idx']} | Canny [{lbl}]\nEdge Pixels: {cnt:,}", fontsize=9)
            axes[i, j].axis('off')
    plt.tight_layout()
    plt.savefig('figures/task3_canny_thresholds.png', dpi=200, bbox_inches='tight')
    plt.close()

    # Task 4
    plt.figure(figsize=(10, 5))
    modes = ['50-100', '100-200', '150-250']
    palette = ['#E60000', '#0099FF', '#33CC33', '#FF9900', '#9933FF']
    for idx, s in enumerate(samples):
        _, gauss = apply_prefiltering(s['bgr'])
        regimes = compute_canny_edges(gauss)
        totals = [np.count_nonzero(regimes[m]) for m in modes]
        plt.plot(modes, totals, marker='D', markersize=6, linewidth=2, color=palette[idx], label=f"{s['idx']} ({s['label']})")
    plt.title('Canny Edge Pixel Count across Threshold Settings', fontsize=12, fontweight='bold')
    plt.xlabel('Dual-Threshold Setting (T_low - T_high)', fontsize=11)
    plt.ylabel('Non-Zero Edge Pixels', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/task4_best_canny_selection.png', dpi=200, bbox_inches='tight')
    plt.close()

    # Task 5
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    for i, s in enumerate(samples):
        _, gauss = apply_prefiltering(s['bgr'])
        edges = cv2.Canny(gauss, 50, 100)
        _, area, perim, vis_rgb = delineate_lesion_boundary(edges, s['bgr'])
        axes[i].imshow(vis_rgb)
        axes[i].set_title(f"{s['idx']} Boundary\nArea: {area:,.1f} px\nPerim: {perim:,.1f} px", fontsize=9, fontweight='bold')
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig('figures/task5_lesion_boundary.png', dpi=200, bbox_inches='tight')
    plt.close()

    # Required Visualization Pipeline Strip
    fig, axes = plt.subplots(5, 5, figsize=(18, 16))
    headers = ["Original Image", "Grayscale", "Gaussian Filter", "Canny (50-100)", "Lesion Boundary"]
    for i, s in enumerate(samples):
        gray, gauss = apply_prefiltering(s['bgr'])
        edges = cv2.Canny(gauss, 50, 100)
        _, area, perim, vis_rgb = delineate_lesion_boundary(edges, s['bgr'])
        row_imgs = [s['rgb'], gray, gauss, edges, vis_rgb]
        cmaps = [None, 'gray', 'gray', 'gray', None]
        for c in range(5):
            axes[i, c].imshow(row_imgs[c], cmap=cmaps[c])
            if i == 0:
                axes[i, c].set_title(headers[c], fontsize=11, fontweight='bold', pad=10)
            if c == 0:
                axes[i, c].set_ylabel(f"{s['idx']}\n{s['label']}", fontsize=10, fontweight='bold')
            if c == 4:
                axes[i, c].set_xlabel(f"A={area:,.1f} | P={perim:,.1f} px", fontsize=9, labelpad=5)
            axes[i, c].set_xticks([])
            axes[i, c].set_yticks([])
    plt.tight_layout()
    plt.savefig('figures/required_visualization_pipeline.png', dpi=200, bbox_inches='tight')
    plt.close()

    # 8-Method Comparison Grid
    s0 = samples[2]  # BCC image
    gray = cv2.cvtColor(s0['bgr'], cv2.COLOR_BGR2GRAY)
    filtered_variants = {
        "Original": gray,
        "Average": cv2.blur(gray, (5, 5)),
        "Gaussian": cv2.GaussianBlur(gray, (5, 5), 1.6),
        "Median": cv2.medianBlur(gray, 5)
    }
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    grid = [
        ("Original + Sobel", filtered_variants["Original"], "sobel", (0, 0)),
        ("Original + Canny", filtered_variants["Original"], "canny", (0, 1)),
        ("Average + Sobel", filtered_variants["Average"], "sobel", (0, 2)),
        ("Average + Canny", filtered_variants["Average"], "canny", (0, 3)),
        ("Gaussian + Sobel", filtered_variants["Gaussian"], "sobel", (1, 0)),
        ("Gaussian + Canny", filtered_variants["Gaussian"], "canny", (1, 1)),
        ("Median + Sobel", filtered_variants["Median"], "sobel", (1, 2)),
        ("Median + Canny", filtered_variants["Median"], "canny", (1, 3))
    ]
    for title, src, kind, (r, c) in grid:
        if kind == 'sobel':
            gx = cv2.Sobel(src, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(src, cv2.CV_64F, 0, 1, ksize=3)
            mag = cv2.magnitude(gx, gy)
            mag = np.uint8(np.clip(mag / (mag.max() + 1e-5) * 255.0, 0, 255))
            _, edge_map = cv2.threshold(mag, 45, 255, cv2.THRESH_BINARY)
        else:
            edge_map = cv2.Canny(src, 50, 100)
        axes[r, c].imshow(edge_map, cmap='gray')
        axes[r, c].set_title(title, fontsize=11, fontweight='bold')
        axes[r, c].axis('off')
    plt.suptitle(f"Comparison of 8 Filter and Edge Detection Methods ({s0['label']})", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('figures/final_comparison_filters_edges.png', dpi=200, bbox_inches='tight')
    plt.close()

def compute_tables(samples):
    table_1 = []
    print("\n" + "="*80)
    print("TABLE 1: LESION AREA AND PERIMETER MEASUREMENTS")
    print("="*80)
    print(f"{'Image':<10} | {'Best Filter':<24} | {'Edge Method':<14} | {'Area (pixels)':<15} | {'Perimeter (pixels)':<18}")
    print("-" * 88)
    for s in samples:
        _, gauss = apply_prefiltering(s['bgr'])
        edges = cv2.Canny(gauss, 50, 100)
        _, area, perim, _ = delineate_lesion_boundary(edges, s['bgr'])
        row = {
            "Image": s['idx'],
            "Diagnosis": s['label'],
            "ISIC_ID": s['isic'],
            "Best Filter": "Gaussian (5x5, sigma=1.6)",
            "Edge Method": "Canny (50-100)",
            "Area (pixels)": round(area, 1),
            "Perimeter (pixels)": round(perim, 1)
        }
        table_1.append(row)
        print(f"{row['Image']:<10} | {row['Best Filter']:<24} | {row['Edge Method']:<14} | {row['Area (pixels)']:<15,.1f} | {row['Perimeter (pixels)']:<18,.1f}")

    with open('table_1.json', 'w', encoding='utf-8') as f:
        json.dump(table_1, f, indent=2)

    table_2 = [
        {
            "Method": "Original + Sobel",
            "Noise Handling": "Poor (Pores, hairs, and fine surface texture generate heavy noise)",
            "Edge Quality": "Thick, diffuse edges with severe noise clutter",
            "Boundary Detection": "Inadequate; lesion perimeter is obscured by dermal texture noise",
            "Overall Performance": "Poor"
        },
        {
            "Method": "Original + Canny",
            "Noise Handling": "Fair (Hysteresis suppresses isolated noise, but preserves micro-texture)",
            "Edge Quality": "Thin 1-pixel edges, but heavily cluttered across epidermal furrows",
            "Boundary Detection": "Fragmented; texture edges interfere with main perimeter closure",
            "Overall Performance": "Moderate"
        },
        {
            "Method": "Average + Sobel",
            "Noise Handling": "Moderate (Box filter reduces fine noise uniformly)",
            "Edge Quality": "Over-smoothed edges with significant spatial displacement",
            "Boundary Detection": "Imprecise; subtle pigment margins are blurred away",
            "Overall Performance": "Fair"
        },
        {
            "Method": "Average + Canny",
            "Noise Handling": "Good (Box blur suppresses high-frequency dermal noise)",
            "Edge Quality": "Sharp edges, minor boundary position drift",
            "Boundary Detection": "Good; detects general contour with occasional perimeter gaps",
            "Overall Performance": "Good"
        },
        {
            "Method": "Gaussian + Sobel",
            "Noise Handling": "Good (Gaussian weighting suppresses noise while preserving edges)",
            "Edge Quality": "Continuous gradient response, but lacks non-maximum suppression",
            "Boundary Detection": "Fair to Good; requires empirical threshold calibration",
            "Overall Performance": "Good"
        },
        {
            "Method": "Gaussian + Canny",
            "Noise Handling": "Optimal (Gaussian filtering + dual-threshold hysteresis tracking)",
            "Edge Quality": "Superior; strictly thin, non-maximum suppressed sharp edges",
            "Boundary Detection": "Superior; forms coherent, well-localized closed boundary",
            "Overall Performance": "Best (Benchmark)"
        },
        {
            "Method": "Median + Sobel",
            "Noise Handling": "Very Good (Non-linear filter removes hair and specular noise)",
            "Edge Quality": "Sharp step-edge preservation without gradient smearing",
            "Boundary Detection": "Good; robust against hair artifacts and specular reflections",
            "Overall Performance": "Very Good"
        },
        {
            "Method": "Median + Canny",
            "Noise Handling": "Excellent (Exceptional hair and specular artifact rejection)",
            "Edge Quality": "Thin, crisp edges with minimal false positive responses",
            "Boundary Detection": "Excellent; highly accurate boundary delineation",
            "Overall Performance": "Excellent (Alternative)"
        }
    ]

    with open('table_2.json', 'w', encoding='utf-8') as f:
        json.dump(table_2, f, indent=2)

    print("\n" + "="*110)
    print("TABLE 2: FINAL COMPARISON ACROSS FILTER AND EDGE DETECTION METHODS")
    print("="*110)
    print(f"{'Method':<18} | {'Noise Handling':<25} | {'Edge Quality':<25} | {'Boundary Detection':<25} | {'Overall'}")
    print("-" * 110)
    for r in table_2:
        print(f"{r['Method']:<18} | {r['Noise Handling'][:24]:<25} | {r['Edge Quality'][:24]:<25} | {r['Boundary Detection'][:24]:<25} | {r['Overall Performance']}")

def main():
    print('[*] Loading samples for Atif Ali Shah (FA23-BAI-004)...')
    samples = load_skin_samples()
    print(f'[+] Loaded {len(samples)} dermoscopic images.')
    print('[*] Generating task visualizations and comparison grids...')
    run_all_visualizations(samples)
    print('[*] Calculating tables and metrics...')
    compute_tables(samples)
    print('[+] Complete.')

if __name__ == '__main__':
    main()
