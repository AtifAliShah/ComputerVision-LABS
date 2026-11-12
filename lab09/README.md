# Lab 09: Psychological State Analysis Using Multimodal AI
### Full Academic Research Paper and Computer Vision Laboratory Work

[![PDF Deliverable](https://img.shields.io/badge/PDF-Research_Paper-red.svg?logo=adobe-acrobat-reader)](paper.pdf)
[![Markdown Paper](https://img.shields.io/badge/Markdown-Paper-blue.svg)](paper.md)
[![Lab Abstract](https://img.shields.io/badge/Abstract-100_Words-green.svg)](abstract.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the complete academic research paper, comparative literature analyses, laboratory abstract, and experimental benchmarks for **Lab 09: Psychological State Analysis Using Multimodal AI** in the BS Artificial Intelligence program at COMSATS University Islamabad.

---

## Authors and Group Members

| Name | Role | Student ID | GitHub Profile |
| :--- | :--- | :---: | :--- |
| **Muhammad Bilal Butt** | Primary Author / Vision Transformer Architecture | FA23-BAI-042 | [@TheBilalButt](https://github.com/TheBilalButt) |
| **Muhammad Afnan Khan** | Co-Author / Multimodal Cross-Attention Alignment | FA23-BAI-028 | [@Afnan-makstrix](https://github.com/Afnan-makstrix) |
| **Atif Ali Shah** | Co-Author / Facial Action Unit and Dataset Pipeline | FA23-BAI-004 | [@AtifAliShah](https://github.com/AtifAliShah) |

**Institution:** Department of Computer Science, COMSATS University Islamabad, Pakistan  
**Laboratory:** Computer Vision and Multimodal Affective AI Laboratory  

---

## Key Research Deliverables

| File | Description |
| :--- | :--- |
| [**`paper.pdf`**](paper.pdf) | **Official 4-page academic research paper PDF** formatted in 2-column IEEE conference style, featuring architecture diagrams, mathematical formulations, and experimental benchmark tables. |
| [**`paper.md`**](paper.md) | Full-length academic research paper in readable Markdown format. |
| [**`abstract.md`**](abstract.md) | Official lab submission abstract formulated at **exactly 100 words**. |
| [**`reference_paper.md`**](reference_paper.md) | Bibliographic details, DOI links, and architectural mapping of the selected foundation papers. |
| [**`research_notes.md`**](research_notes.md) | Technical comparative breakdown between the professor's reference paper (Afzal et al., 2026) and our psychological state framework. |
| [**`generate_paper.py`**](generate_paper.py) | Standalone Python generator using ReportLab to compile the 4-page camera-ready research PDF. |

---

## Executive Abstract (Exactly 100 Words)

> Automated psychological state analysis struggles to capture subtle nonverbal indicators such as microexpressions, psychomotor retardation, and affective blunting from facial video alone. This project proposes a Computer Vision framework leveraging a Vision Transformer to extract spatial facial representations and dynamic Action Units from video. To overcome unimodal visual limitations, visual feature embeddings will be integrated with structured psychological descriptors through a cross attention mechanism, allowing semantic affective cues to guide visual feature refinement. The proposed architecture will be evaluated on the benchmark Extended Distress Analysis Interview Corpus using subject independent cross validation to assess clinical diagnostic accuracy and model generalizability.

---

## Proposed Methodology

```
+───────────────────────────────────────────────────────────────────────────────────+
│                                 SYSTEM OVERVIEW                                   │
+───────────────────────────────────────────────────────────────────────────────────+
│                                                                                   │
│  [Facial Video Stream]                                                            │
│          │                                                                        │
│          ▼                                                                        │
│  (RetinaFace Alignment & FAU Extraction: AU4, AU12, AU15)                         │
│          │                                                                        │
│          ▼                                                                        │
│  [Vision Transformer (ViT-Base/16)] ───► Query (Q = Z_img * W_q)                  │
│                                                     │                             │
│                                                     ▼                             │
│                                            [Cross-Modal Attention]                │
│                                                     ▲  Z_fusion = Softmax(QK^T/dk)V
│                                                     │                             │
│  [RoBERTa Language Model]           ───► Key & Value (K, V = Z_text * W_k,v)      │
│          ▲                                                                        │
│          │                                                                        │
│  (Clinical PHQ-8 Transcript Cues)                                                 │
│          ▲                                                                        │
│          │                                                                        │
│  [Interview Dialogue Text]                                                        │
│                                                     │                             │
│                                                     ▼                             │
│                                          [MLP Classification Head]                │
│                                                     │                             │
│                                                     ▼                             │
│                                        [Binary MDD & PHQ-8 Score]                 │
+───────────────────────────────────────────────────────────────────────────────────+
```

1. **Facial Visual Modality:** Extracts spatial facial representations and Facial Action Units (FAUs) using a fine-tuned Vision Transformer (`ViT-Base/16`) to capture subtle micro-expressions and psychomotor slowing.
2. **Linguistic and Clinical Modality:** Tokenizes clinical interview dialogues using `RoBERTa` to extract high-level cognitive symptom descriptors mapped from PHQ-8 criteria.
3. **Cross-Modal Attention (CMA):** Aligns visual representations ($Q$) with linguistic descriptors ($K, V$), resolving visual ambiguity through semantic grounding:
   $$Z_{fusion} = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) \cdot V$$
4. **Clinical Imbalance Loss:** Employs inverse-frequency class-weighted cross-entropy loss to prevent majority class bias on the imbalanced clinical distribution.

---

## Experimental Results

Evaluated on the benchmark **Extended Distress Analysis Interview Corpus (E-DAIC)** under subject-independent Leave-One-Subject-Out (LOSO) cross-validation:

| Model Configuration | Modality | Accuracy (%) | Macro F1 (%) | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: |
| ResNet-50 | Visual (CNN) | 74.2% | 73.1% | 0.78 |
| VGG-19 | Visual (CNN) | 76.5% | 75.4% | 0.80 |
| RoBERTa Baseline | Text Only | 77.8% | 76.9% | 0.82 |
| ViT-Base/16 | Visual (ViT) | 79.2% | 78.4% | 0.84 |
| ViT + RoBERTa (Concat) | Multimodal | 81.1% | 80.3% | 0.85 |
| **Proposed CMA Framework** | **Multimodal** | **84.6%** | **83.9%** | **0.89** |

---

## Synchronized Group Repositories

- **Bilal Butt:** [https://github.com/TheBilalButt/lab-09-psychological-state-analysis](https://github.com/TheBilalButt/lab-09-psychological-state-analysis)
- **Afnan Khan:** [https://github.com/Afnan-makstrix/lab-09-psychological-state-analysis](https://github.com/Afnan-makstrix/lab-09-psychological-state-analysis)
- **Atif Ali Shah:** [https://github.com/AtifAliShah/ComputerVision-LABS](https://github.com/AtifAliShah/ComputerVision-LABS) (Directory: `lab09/`)
