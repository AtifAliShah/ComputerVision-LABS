"""
Industrial Automated Optical Inspection (AOI) - Live Conveyor Simulation
Student: Atif Ali Shah (FA23-BAI-004)
Course: Computer Vision Lab 05
"""

import cv2
import numpy as np
import time
import os
import glob
from skimage.feature import hog
from sklearn.svm import LinearSVC
import pickle

def simulate_aoi_inspector():
    print("=" * 65)
    print("   AUTOMATED OPTICAL INSPECTION (AOI) - LIVE CONVEYOR STREAM   ")
    print("   Student: Atif Ali Shah | Reg: FA23-BAI-004                  ")
    print("=" * 65)
    
    # 1. Train quick linear classifier on available images
    data_dir = "data"
    image_paths = glob.glob(os.path.join(data_dir, "*", "*.jpg")) + glob.glob(os.path.join(data_dir, "*", "*.png"))
    
    if not image_paths:
        print("[ERROR] No images found in data directory!")
        return

    print(f"[*] Training baseline HOG-SVM detector from {len(image_paths)} surface samples...")
    X, y = [], []
    for p in image_paths:
        cls_name = os.path.basename(os.path.dirname(p))
        is_defective = 0 if cls_name == "normal" else 1
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img_res = cv2.resize(img, (128, 128))
        feat = hog(img_res, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')
        X.append(feat)
        y.append(is_defective)

    clf = LinearSVC(C=1.0, random_state=42)
    clf.fit(X, y)
    print("[+] Model calibrated. Launching simulated conveyor belt inspection window...")
    print("    Press 'q' to stop conveyor feed, 'SPACE' to pause inspection.")

    # 2. Simulate conveyor frames
    test_samples = image_paths.copy()
    np.random.seed(42)
    np.random.shuffle(test_samples)

    idx = 0
    fps_time = time.time()
    frame_count = 0
    fps = 0.0

    while True:
        img_path = test_samples[idx % len(test_samples)]
        raw_img = cv2.imread(img_path)
        cls_label = os.path.basename(os.path.dirname(img_path))
        idx += 1

        if raw_img is None:
            continue

        start_t = time.perf_counter()
        # Processing
        gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (128, 128))
        feat, hog_vis = hog(resized, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                            block_norm='L2-Hys', visualize=True)
        
        # Classification
        pred = clf.predict([feat])[0]
        dist = clf.decision_function([feat])[0]
        # Calibrate pseudo-prob
        conf = 1.0 / (1.0 + np.exp(-dist)) if pred == 1 else 1.0 / (1.0 + np.exp(dist))
        conf_pct = min(max(conf * 100.0, 50.0), 99.9)

        infer_time = (time.perf_counter() - start_t) * 1000.0

        # Construct visual frame
        display_img = cv2.resize(raw_img, (320, 320))
        hog_vis_img = cv2.resize((hog_vis * 255).astype(np.uint8), (320, 320))
        hog_vis_bgr = cv2.cvtColor(hog_vis_img, cv2.COLOR_GRAY2BGR)

        # Composite HUD
        canvas = np.zeros((460, 680, 3), dtype=np.uint8)
        canvas[:320, :320] = display_img
        canvas[:320, 340:660] = hog_vis_bgr

        # Separator line
        cv2.line(canvas, (330, 0), (330, 320), (50, 50, 50), 2)
        cv2.line(canvas, (0, 330), (680, 330), (70, 70, 70), 2)

        # FPS calculation
        frame_count += 1
        if time.time() - fps_time >= 1.0:
            fps = frame_count / (time.time() - fps_time)
            frame_count = 0
            fps_time = time.time()

        # Telemetry Text
        status_color = (40, 40, 230) if pred == 1 else (60, 200, 60)
        status_text = "REJECT [DEFECT]" if pred == 1 else "ACCEPT [NORMAL]"

        cv2.putText(canvas, f"SURFACE STREAM: {cls_label.upper()}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(canvas, "HOG GRADIENT ENERGY MAP", (355, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Bottom HUD
        cv2.putText(canvas, f"STATUS: {status_text}", (20, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
        cv2.putText(canvas, f"CONFIDENCE: {conf_pct:.1f}%", (20, 405), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1)
        cv2.putText(canvas, f"LATENCY: {infer_time:.2f} ms | FPS: {fps:.1f}", (20, 435), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        cv2.putText(canvas, "QC STATION: LINE-04 (FA23-BAI-004)", (370, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 200, 255), 1)
        cv2.putText(canvas, f"HOG PARAMS: 8x8 cell, 9 bins", (370, 405), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 200, 255), 1)
        cv2.putText(canvas, "PRESS 'Q' TO QUIT", (370, 435), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 1)

        # Show frame
        cv2.imshow("Industrial Defect Detection HUD - Conveyor Stream", canvas)
        key = cv2.waitKey(400) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            cv2.waitKey(0)

    cv2.destroyAllWindows()
    print("[+] Conveyor stream halted successfully.")

if __name__ == "__main__":
    simulate_aoi_inspector()
