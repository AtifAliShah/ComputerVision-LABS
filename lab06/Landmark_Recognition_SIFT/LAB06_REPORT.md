# Academic Lab Report: Landmark Recognition Using the SIFT Algorithm

**Course:** Computer Vision  
**Student Name:** Atif Ali Shah  
**Student ID:** FA23-BAI-004  
**Repository:** [ComputerVision-LABS](https://github.com/AtifAliShah/ComputerVision-LABS)  
**Module:** `lab06/Landmark_Recognition_SIFT`  

---

## Abstract

Automated landmark recognition plays a crucial role in modern intelligent tourism systems. In this laboratory, we implement an end-to-end computer vision system using the Scale-Invariant Feature Transform (SIFT) algorithm, Brute-Force feature matching with Lowe's ratio test ($k=2$), and RANSAC homography estimation for geometric verification and localization. The system is benchmarked across six distinct environmental scenarios, demonstrating high accuracy under rotation, scale variation, and illumination changes, while rejecting non-database queries with 100% precision.

---

## 1. Theoretical Background & Mathematical Derivations

### 1.1 Scale-Space Extrema Detection (DoG)
To achieve scale invariance, the image is convolved with Gaussian kernels at octave intervals:
$$L(x, y, \sigma) = G(x, y, \sigma) * I(x, y)$$
The Difference-of-Gaussians (DoG) approximates the Laplacian of Gaussian:
$$D(x, y, \sigma) = (G(x, y, k\sigma) - G(x, y, \sigma)) * I(x, y) = L(x, y, k\sigma) - L(x, y, \sigma)$$
Extrema are identified by comparing each sample point against its 26 neighbors in a $3 \times 3 \times 3$ scale-space volume.

### 1.2 Keypoint Localization & Hessian Elimination
Low-contrast candidate points are discarded, and edge responses are eliminated using the 2D Hessian matrix:
$$H = \begin{bmatrix} D_{xx} & D_{xy} \\ D_{xy} & D_{yy} \end{bmatrix}$$
Keypoints are retained only if:
$$\frac{\text{Tr}(H)^2}{\text{Det}(H)} < \frac{(r+1)^2}{r}, \quad r = 10$$

### 1.3 Orientation Assignment
Gradient magnitude and orientation:
$$m(x, y) = \sqrt{(L(x+1, y) - L(x-1, y))^2 + (L(x, y+1) - L(x, y-1))^2}$$
$$\theta(x, y) = \text{atan2}(L(x, y+1) - L(x, y-1), L(x+1, y) - L(x-1, y))$$

### 1.4 Descriptor Construction
A $16 \times 16$ neighborhood around each keypoint is divided into $4 \times 4$ subregions, each contributing an 8-bin orientation histogram, resulting in a 128-dimensional feature vector.

### 1.5 Lowe's Ratio Test
Ambiguous matches are filtered by evaluating the distance ratio between the first and second nearest neighbors:
$$\text{Condition: } \frac{d_1}{d_2} < 0.75$$

### 1.6 RANSAC Planar Homography
Given 2D point correspondences $(x_i, y_i) \leftrightarrow (x'_i, y'_i)$, the planar projective transform is computed via:
$$s \begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

---

## 2. Experimental Results & Performance Table

| Condition | Ground Truth | Predicted Landmark | Query KPs | Matches | Inliers | Decision | Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Original Landmark Image** | `eiffel_tower` | `eiffel_tower` | `2,946` | `2,946` | `2,946` | **RECOGNIZED** | `986.8 ms` |
| **Rotated Landmark Image (45 deg)** | `eiffel_tower` | `eiffel_tower` | `2,517` | `1,234` | `1,227` | **RECOGNIZED** | `801.7 ms` |
| **Resized Landmark Image (0.5x Scale)** | `eiffel_tower` | `eiffel_tower` | `672` | `509` | `508` | **RECOGNIZED** | `236.6 ms` |
| **Reduced Brightness Image (0.35x)** | `eiffel_tower` | `eiffel_tower` | `760` | `718` | `717` | **RECOGNIZED** | `312.0 ms` |
| **Different Viewpoint Image** | `eiffel_tower` | `eiffel_tower` | `1,020` | `1,020` | `1,020` | **RECOGNIZED** | `377.4 ms` |
| **Non-Database Landmark (Big Ben)** | `unknown` | `Unknown / Rejected` | `1,984` | `22` | `0` | **REJECTED** | `666.9 ms` |

### Visual Documentation
- **Reference Database Gallery:** `figures/task1_reference_database.png`
- **SIFT Feature Detection:** `figures/task2_sift_features.png`
- **Feature Matching Pairs:** `figures/task4_feature_matching_comparison.png`
- **Environmental Stress Evaluation:** `figures/task6_environmental_evaluation.png`
- **Performance Telemetry:** `figures/task6_benchmark_telemetry.png`

---

## 3. Analytical Discussion

1. **Invariance Characteristics:** The system exhibited high stability across 45° rotation (1,234 matches, 1,227 inliers) and 0.5x scaling (509 matches, 508 inliers).
2. **Photometric Resistance:** Low illumination (0.35x brightness) had minimal impact on gradient ratios, retaining 718 matches.
3. **Rejection Gate Efficiency:** The non-database query (Big Ben) yielded only 22 spurious matches and 0 inliers, reliably triggering rejection.

---

## 4. Conclusion

The SIFT-based landmark recognition pipeline satisfies all assignment tasks with full functional verification, geometric boundary mapping, and automated rejection.
