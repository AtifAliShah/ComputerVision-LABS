# Lab 06: Landmark Recognition Using SIFT Algorithm

**Student:** Atif Ali Shah  
**Student ID:** FA23-BAI-004  
**Course:** Computer Vision  
**Parent Repository:** [ComputerVision-LABS](https://github.com/AtifAliShah/ComputerVision-LABS)  

---

## Overview

This module implements a complete landmark recognition and geometric verification pipeline using SIFT feature extraction, Brute-Force KNN matching ($k=2$), Lowe's ratio test ($0.75$), and RANSAC homography estimation.

### Benchmark Summary
| Condition | Ground Truth | Predicted | Matches | Inliers | Decision | Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Original Landmark Image** | `eiffel_tower` | `eiffel_tower` | `2,946` | `2,946` | `2,946` | **RECOGNIZED** | `986.8 ms` |
| **Rotated Landmark Image (45 deg)** | `eiffel_tower` | `eiffel_tower` | `2,517` | `1,234` | `1,227` | **RECOGNIZED** | `801.7 ms` |
| **Resized Landmark Image (0.5x Scale)** | `eiffel_tower` | `eiffel_tower` | `672` | `509` | `508` | **RECOGNIZED** | `236.6 ms` |
| **Reduced Brightness Image (0.35x)** | `eiffel_tower` | `eiffel_tower` | `760` | `718` | `717` | **RECOGNIZED** | `312.0 ms` |
| **Different Viewpoint Image** | `eiffel_tower` | `eiffel_tower` | `1,020` | `1,020` | `1,020` | **RECOGNIZED** | `377.4 ms` |
| **Non-Database Landmark (Big Ben)** | `unknown` | `Unknown / Rejected` | `1,984` | `22` | `0` | **REJECTED** | `666.9 ms` |

### Execution
```bash
python run_lab06.py
```
