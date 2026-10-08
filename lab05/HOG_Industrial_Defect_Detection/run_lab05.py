"""
Lab 05: HOG-Based Industrial Defect Detection and Classification
Student ID: FA23-BAI-004
Course: Computer Vision
Repository: AtifAliShah / ComputerVision-LABS

This script executes the complete end-to-end experimental workflow:
  - Task 1: Ingest NEU Industrial Defect Dataset
  - Task 2: Standardized Resolution Normalization (128x128 Grayscale)
  - Task 3: HOG Feature Extraction & Orientation Field Visualization
  - Task 4: Classifier Induction (Support Vector Machine vs Random Forest)
  - Task 5: 5-Fold Stratified Cross-Validation & Metric Comparison
  - Task 6: Parameter Grid Study (Cell Sizes 4x4, 8x8, 16x16 & Orientations 6, 9, 12)
  - Task 7: Industrial Stress & Perturbation Testing (Exposure, Noise, Skew, Blur)
  - Task 8: Automated Factory Quality Gate (Accept / Reject Logic)
"""

import os
import glob
import json
import time
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.feature import hog
from skimage import exposure
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import StratifiedKFold

os.makedirs("figures", exist_ok=True)
os.makedirs("data", exist_ok=True)

CLASS_NAMES = ['normal', 'crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
LABEL_MAP = {c: i for i, c in enumerate(CLASS_NAMES)}

def load_data():
    files = sorted(glob.glob("data/*.jpg"))
    if not files:
        raise FileNotFoundError("Dataset missing in data/ directory.")
    
    imgs, y_bin, y_mul, tags = [], [], [], []
    for f in files:
        cat = os.path.basename(f).rsplit('_', 1)[0]
        if cat not in LABEL_MAP:
            continue
        gray = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        resized = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
        imgs.append(resized)
        tags.append(cat)
        y_mul.append(LABEL_MAP[cat])
        y_bin.append(0 if cat == 'normal' else 1)
    
    print(f"Loaded {len(imgs)} samples (Normal: {y_bin.count(0)}, Defective: {y_bin.count(1)})")
    return np.array(imgs), np.array(y_bin), np.array(y_mul), tags

def plot_task1_gallery(images, tags):
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    fig.patch.set_facecolor('#1e293b')

    rep_map = {}
    for i, t in enumerate(tags):
        if t not in rep_map:
            rep_map[t] = i

    for idx, c in enumerate(CLASS_NAMES):
        ax = axes[idx // 4, idx % 4]
        ax.set_facecolor('#0f172a')
        s_idx = rep_map[c]
        ax.imshow(images[s_idx], cmap='gray')
        col = '#22c55e' if c == 'normal' else '#ef4444'
        ax.set_title(f"{c.upper().replace('_', ' ')}\n{'[PASS]' if c == 'normal' else '[DEFECT]'} ",
                     color=col, fontsize=10, fontweight='bold', pad=8)
        ax.axis('off')

    # Telemetry slot
    ax_tele = axes[1, 3]
    ax_tele.set_facecolor('#0f172a')
    ax_tele.axis('off')
    tele_text = (
        "METADATA SUMMARY\n"
        "=========================\n"
        f"Total Inspected: {len(images)}\n"
        "Input Res: 128x128 Grayscale\n"
        "Defect Categories: 6 Classes\n"
        "Baseline Pass Rate: 14.3%\n"
        "Defect Incidence: 85.7%\n"
        "=========================\n"
        "Student ID: FA23-BAI-004"
    )
    ax_tele.text(0.5, 0.5, tele_text, color='#94a3b8', fontsize=9.5, family='monospace',
                 ha='center', va='center', bbox=dict(boxstyle='square,pad=0.8', facecolor='#1e293b', edgecolor='#475569'))

    plt.suptitle("Figure 1: Industrial Surface Defect Gallery (Student ID: FA23-BAI-004)",
                 fontsize=14, fontweight='bold', color='#f8fafc', y=0.98)
    plt.tight_layout()
    plt.savefig("figures/task1_surface_defect_samples.png", dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()

def plot_task3_hog(images, tags):
    subset = ['normal', 'crazing', 'pitted_surface', 'scratches']
    fig, axes = plt.subplots(4, 3, figsize=(13, 15))
    fig.patch.set_facecolor('#1e293b')

    for r, c in enumerate(subset):
        idx = next(i for i, t in enumerate(tags) if t == c)
        img = images[idx]

        _, h_vis = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                       block_norm='L2-Hys', visualize=True)
        h_scaled = exposure.rescale_intensity(h_vis, in_range=(0, 10))

        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        axes[r, 0].imshow(img, cmap='gray')
        axes[r, 0].set_title(f"{c.upper()}\n[Original Surface]", color='#f8fafc', fontsize=10, fontweight='bold')
        axes[r, 0].axis('off')

        axes[r, 1].imshow(mag_norm, cmap='cividis')
        axes[r, 1].set_title("Sobel Gradient Magnitude", color='#38bdf8', fontsize=10, fontweight='bold')
        axes[r, 1].axis('off')

        axes[r, 2].imshow(h_scaled, cmap='magma')
        axes[r, 2].set_title("HOG Orientation Mesh (8x8, 9 Bins)", color='#f59e0b', fontsize=10, fontweight='bold')
        axes[r, 2].axis('off')

    plt.suptitle("Figure 2: HOG Orientation Signatures & Gradient Profiles",
                 fontsize=14, fontweight='bold', color='#f8fafc', y=0.99)
    plt.tight_layout()
    plt.savefig("figures/task3_hog_feature_visualizations.png", dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()

def run_classifiers(X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = {
        'Linear SVM': SVC(kernel='linear', C=1.0, probability=True, random_state=42),
        'Random Forest (100 Trees)': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    }

    metrics = {}
    cms = {}
    for name, clf in models.items():
        t0 = time.time()
        accs, precs, recs, f1s = [], [], [], []
        preds_all, trues_all = [], []

        for tr_i, te_i in skf.split(X, y):
            clf.fit(X[tr_i], y[tr_i])
            p = clf.predict(X[te_i])
            accs.append(accuracy_score(y[te_i], p))
            precs.append(precision_score(y[te_i], p, zero_division=0))
            recs.append(recall_score(y[te_i], p, zero_division=0))
            f1s.append(f1_score(y[te_i], p, zero_division=0))
            preds_all.extend(p)
            trues_all.extend(y[te_i])

        dt = time.time() - t0
        metrics[name] = {
            'accuracy': float(np.mean(accs)),
            'precision': float(np.mean(precs)),
            'recall': float(np.mean(recs)),
            'f1_score': float(np.mean(f1s)),
            'latency_ms': float((dt / len(X)) * 1000)
        }
        cms[name] = confusion_matrix(trues_all, preds_all)

    # Plot Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor('#1e293b')
    cmaps = ['Blues', 'Reds']

    for i, (name, cm) in enumerate(cms.items()):
        ax = axes[i]
        ax.set_facecolor('#0f172a')
        cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cmn, annot=True, fmt='.1%', cmap=cmaps[i],
                    xticklabels=['Pass', 'Defect'], yticklabels=['Pass', 'Defect'],
                    cbar=False, ax=ax, annot_kws={"size": 12, "weight": "bold"})
        ax.set_title(f"{name}\nAcc: {metrics[name]['accuracy']*100:.1f}% | F1: {metrics[name]['f1_score']*100:.1f}%",
                     color='#f8fafc', fontsize=11, fontweight='bold', pad=10)
        ax.set_xlabel("Predicted Action", color='#94a3b8')
        ax.set_ylabel("True Surface State", color='#94a3b8')
        ax.tick_params(colors='#cbd5e1')

    plt.suptitle("Figure 3: Classifier Confusion Matrices (5-Fold Stratified CV)",
                 fontsize=14, fontweight='bold', color='#f8fafc', y=1.02)
    plt.tight_layout()
    plt.savefig("figures/task4_classifier_confusion_matrices.png", dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()

    return metrics

def run_parameter_ablation(images, y):
    cell_sizes = [(4, 4), (8, 8), (16, 16)]
    orientations = [6, 9, 12]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = []
    for cell in cell_sizes:
        for ori in orientations:
            t0 = time.time()
            feats = []
            for img in images:
                f = hog(img, orientations=ori, pixels_per_cell=cell, cells_per_block=(2, 2),
                        block_norm='L2-Hys', visualize=False)
                feats.append(f)
            X_arr = np.array(feats)
            ext_time = (time.time() - t0) / len(images)

            clf = SVC(kernel='linear', C=1.0, random_state=42)
            accs, f1s = [], []
            for tr_i, te_i in skf.split(X_arr, y):
                clf.fit(X_arr[tr_i], y[tr_i])
                preds = clf.predict(X_arr[te_i])
                accs.append(accuracy_score(y[te_i], preds))
                f1s.append(f1_score(y[te_i], preds, zero_division=0))

            results.append({
                'cell_size': f"{cell[0]}x{cell[1]}",
                'orientations': ori,
                'feature_dim': int(X_arr.shape[1]),
                'extraction_ms': float(ext_time * 1000),
                'accuracy': float(np.mean(accs)),
                'f1_score': float(np.mean(f1s))
            })

    with open("table_1_hog_parameters.json", "w") as f:
        json.dump(results, f, indent=4)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.patch.set_facecolor('#1e293b')

    fixed_ori_9 = [r for r in results if r['orientations'] == 9]
    c_labels = [r['cell_size'] for r in fixed_ori_9]
    accs = [r['accuracy'] * 100 for r in fixed_ori_9]
    f1s = [r['f1_score'] * 100 for r in fixed_ori_9]
    dims = [r['feature_dim'] for r in fixed_ori_9]

    idx = np.arange(len(c_labels))
    w = 0.35
    ax1.set_facecolor('#0f172a')
    ax1.bar(idx - w/2, accs, w, label='Accuracy (%)', color='#38bdf8', edgecolor='#0284c7')
    ax1.bar(idx + w/2, f1s, w, label='F1-Score (%)', color='#a855f7', edgecolor='#7e22ce')
    ax1.set_xticks(idx)
    ax1.set_xticklabels([f"{c}\n(Dim: {d})" for c, d in zip(c_labels, dims)], color='#cbd5e1')
    ax1.set_title("HOG Cell Size Analysis (Fixed 9 Orientations)", color='#f8fafc', fontsize=11, fontweight='bold')
    ax1.set_ylabel("Performance (%)", color='#94a3b8')
    ax1.set_ylim(70, 105)
    ax1.legend(facecolor='#1e293b', edgecolor='#475569', labelcolor='#f8fafc')
    ax1.grid(color='#334155', linestyle='--', alpha=0.5)

    fixed_cell_8 = [r for r in results if r['cell_size'] == '8x8']
    b_labels = [f"{r['orientations']} Bins" for r in fixed_cell_8]
    b_accs = [r['accuracy'] * 100 for r in fixed_cell_8]
    b_f1s = [r['f1_score'] * 100 for r in fixed_cell_8]
    tms = [r['extraction_ms'] for r in fixed_cell_8]

    ax2.set_facecolor('#0f172a')
    ax2.bar(idx - w/2, b_accs, w, label='Accuracy (%)', color='#10b981', edgecolor='#059669')
    ax2.bar(idx + w/2, b_f1s, w, label='F1-Score (%)', color='#f59e0b', edgecolor='#d97706')
    ax2.set_xticks(idx)
    ax2.set_xticklabels([f"{l}\n({t:.1f}ms)" for l, t in zip(b_labels, tms)], color='#cbd5e1')
    ax2.set_title("HOG Orientations Analysis (Fixed 8x8 Cell Size)", color='#f8fafc', fontsize=11, fontweight='bold')
    ax2.set_ylabel("Performance (%)", color='#94a3b8')
    ax2.set_ylim(70, 105)
    ax2.legend(facecolor='#1e293b', edgecolor='#475569', labelcolor='#f8fafc')
    ax2.grid(color='#334155', linestyle='--', alpha=0.5)

    plt.suptitle("Figure 4: HOG Parameter Ablation Grid Study", fontsize=14, fontweight='bold', color='#f8fafc', y=1.02)
    plt.tight_layout()
    plt.savefig("figures/task5_hog_parameter_comparison.png", dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    return results

def run_robustness(images, y):
    np.random.seed(42)
    clean_feats = [hog(im, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys') for im in images]
    X_clean = np.array(clean_feats)

    clf = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
    clf.fit(X_clean, y)
    base_acc = accuracy_score(y, clf.predict(X_clean))
    base_f1 = f1_score(y, clf.predict(X_clean))

    def corrupt(img, mode):
        h, w = img.shape
        if mode == 'underexposure':
            return np.clip(img.astype(np.float32) * 0.50, 0, 255).astype(np.uint8)
        elif mode == 'overexposure':
            return np.clip(img.astype(np.float32) * 1.50, 0, 255).astype(np.uint8)
        elif mode == 'noise':
            n = np.random.normal(0, 20, img.shape)
            return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)
        elif mode == 'skew':
            M = cv2.getRotationMatrix2D((w // 2, h // 2), 15, 1.0)
            return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        elif mode == 'blur':
            return cv2.GaussianBlur(img, (7, 7), 2.5)
        return img

    tests = [
        ('Baseline (Clean)', 'clean'),
        ('Low Brightness (0.5x)', 'underexposure'),
        ('High Brightness (1.5x)', 'overexposure'),
        ('Gaussian Noise (sigma=20)', 'noise'),
        ('Conveyor Rotation (+15 deg)', 'skew'),
        ('Optical Blur (sigma=2.5)', 'blur')
    ]

    out = []
    samples = []
    for label, mode in tests:
        if mode == 'clean':
            out.append({
                'condition': label,
                'accuracy': float(base_acc),
                'f1_score': float(base_f1),
                'accuracy_drop': 0.0,
                'f1_drop': 0.0
            })
            samples.append(images[0])
            continue

        p_imgs = [corrupt(im, mode) for im in images]
        samples.append(p_imgs[0])
        X_p = np.array([hog(im, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys') for im in p_imgs])
        preds = clf.predict(X_p)
        acc = accuracy_score(y, preds)
        f1 = f1_score(y, preds)

        out.append({
            'condition': label,
            'accuracy': float(acc),
            'f1_score': float(f1),
            'accuracy_drop': float(base_acc - acc),
            'f1_drop': float(base_f1 - f1)
        })

    with open("table_2_robustness.json", "w") as f:
        json.dump(out, f, indent=4)

    # Plot
    fig = plt.figure(figsize=(16, 9.5))
    fig.patch.set_facecolor('#1e293b')
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 1.25])

    for i, ((name, _), smp) in enumerate(zip(tests, samples)):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor('#0f172a')
        ax.imshow(smp, cmap='gray')
        ax.set_title(name.split(' (')[0], color='#38bdf8', fontsize=9.5, fontweight='bold', pad=6)
        ax.axis('off')

    ax_b = fig.add_subplot(gs[1, :])
    ax_b.set_facecolor('#0f172a')
    names = [r['condition'] for r in out]
    accs = [r['accuracy'] * 100 for r in out]
    f1s = [r['f1_score'] * 100 for r in out]
    idx = np.arange(len(names))
    w = 0.35

    ax_b.bar(idx - w/2, accs, w, label='Accuracy (%)', color='#38bdf8', edgecolor='#0284c7')
    ax_b.bar(idx + w/2, f1s, w, label='F1-Score (%)', color='#ef4444', edgecolor='#dc2626')
    ax_b.set_xticks(idx)
    ax_b.set_xticklabels(names, rotation=15, ha='right', color='#cbd5e1', fontsize=9.5)
    ax_b.set_ylabel('Performance (%)', color='#94a3b8', fontsize=11)
    ax_b.set_ylim(50, 105)
    ax_b.legend(facecolor='#1e293b', edgecolor='#475569', labelcolor='#f8fafc', loc='lower left')
    ax_b.grid(color='#334155', linestyle='--', alpha=0.5)

    for i in range(1, len(out)):
        drop_val = out[i]['accuracy_drop'] * 100
        ax_b.annotate(f"-{drop_val:.1f}%", xy=(idx[i] - w/2, accs[i]), xytext=(0, 4),
                      textcoords="offset points", ha='center', color='#facc15', fontsize=9, fontweight='bold')

    plt.suptitle("Figure 5: Environmental Robustness Analysis (Atif Ali Shah - FA23-BAI-004)",
                 fontsize=14, fontweight='bold', color='#f8fafc', y=0.98)
    plt.tight_layout()
    plt.savefig("figures/task6_robustness_perturbation_tests.png", dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    return out

def plot_qc_gate(clf, images):
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    fig.patch.set_facecolor('#1e293b')

    for i in range(8):
        ax = axes[i // 4, i % 4]
        ax.set_facecolor('#0f172a')
        img = images[i]
        feat = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys').reshape(1, -1)
        prob = clf.predict_proba(feat)[0]
        is_def = prob[1] >= 0.50
        conf = prob[1] if is_def else prob[0]
        action = "REJECT PRODUCT" if is_def else "ACCEPT PRODUCT"
        pred = "DEFECTIVE" if is_def else "NON-DEFECTIVE"
        col = '#ef4444' if is_def else '#22c55e'

        ax.imshow(img, cmap='gray')
        ax.set_title(f"{action}\n{pred} ({conf*100:.1f}%)", color=col, fontsize=9.5, fontweight='bold', pad=6)
        ax.axis('off')
        rect = plt.Rectangle((0, 0), img.shape[1], img.shape[0], linewidth=3, edgecolor=col, facecolor='none')
        ax.add_patch(rect)

    plt.suptitle("Figure 6: Automated Factory QC Telemetry & Decision Output",
                 fontsize=14, fontweight='bold', color='#f8fafc', y=0.98)
    plt.tight_layout()
    plt.savefig("figures/task7_qc_decision_telemetry.png", dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()

def main():
    print("Executing Lab 05 for Atif Ali Shah (FA23-BAI-004)...")
    imgs, y_bin, y_mul, tags = load_data()
    plot_task1_gallery(imgs, tags)

    X_hog = np.array([hog(im, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys') for im in imgs])
    plot_task3_hog(imgs, tags)

    m = run_classifiers(X_hog, y_bin)
    print("Classifiers evaluated successfully.")

    run_parameter_ablation(imgs, y_bin)
    print("Parameter ablation finished.")

    run_robustness(imgs, y_bin)
    print("Robustness evaluation finished.")

    clf_final = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
    clf_final.fit(X_hog, y_bin)

    plot_qc_gate(clf_final, imgs)
    print("Lab 05 pipeline executed successfully for Atif Ali Shah!")

if __name__ == '__main__':
    main()
