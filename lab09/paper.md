# MULTIMODAL PSYCHOLOGICAL STATE ANALYSIS VIA FACIAL VISION TRANSFORMER AND CROSS-MODAL FEATURE ALIGNMENT

**Muhammad Bilal Butt<sup>1</sup>, Muhammad Afnan Khan<sup>1</sup>, Atif Ali Shah<sup>1</sup>**  
*<sup>1</sup>Department of Computer Science, COMSATS University Islamabad, Pakistan*  
*Email: {primebilalbutt, afnan.makstrix, atif.shah}@gmail.com*  
*Computer Vision & Multimodal Affective AI Laboratory*  

---

## ABSTRACT

Automated psychological state analysis struggles to capture subtle nonverbal indicators such as microexpressions, psychomotor retardation, and affective blunting from facial video alone. This project proposes a Computer Vision framework leveraging a Vision Transformer to extract spatial facial representations and dynamic Action Units from video. To overcome unimodal visual limitations, visual feature embeddings will be integrated with structured psychological descriptors through a cross attention mechanism, allowing semantic affective cues to guide visual feature refinement. The proposed architecture will be evaluated on the benchmark Extended Distress Analysis Interview Corpus using subject independent cross validation to assess clinical diagnostic accuracy and model generalizability.

**Index Terms**: Affective computing, Computer Vision, facial expression recognition, multimodal artificial intelligence, Vision Transformer, cross-modal attention, depression detection.

---

## 1. INTRODUCTION

Psychological disorders such as Major Depressive Disorder (MDD), anxiety, and persistent mood dysregulation affect over 300 million people worldwide [1]. Clinical assessment still depends heavily on self-report questionnaires (such as PHQ-8, PHQ-9, and the Beck Depression Inventory BDI-II) and structured psychiatric interviews. While useful, these traditional screening tools suffer from recall bias, social stigma, and subjective clinician judgment, making continuous, objective monitoring difficult to achieve [2].

Clinical research tracks psychological conditions across three primary behavioral domains:
1. **Affective Domain:** Persistent sadness, flat affect, anhedonia, and emotional blunting.
2. **Cognitive Domain:** Feelings of worthlessness, slowing of thought, hopelessness, and negative attribution.
3. **Psychomotor Domain:** Observable motor retardation, reduced head dynamics, and downward gaze fixations [3].

To make psychological assessment objective, computer vision researchers have applied facial expression recognition (FER) using Convolutional Neural Networks (CNNs) [4]. However, standalone facial vision systems face major challenges. Standard CNNs rely on local receptive fields, which make it difficult to capture long-range spatial correlations across distant parts of the face, such as synchronous brow lowering and mouth drooping [5]. Furthermore, visual data alone carries contextual ambiguity: a neutral face might indicate focused concentration rather than affective blunting. Additionally, clinical video datasets have limited sample sizes and significant demographic variation [6].

On the other hand, natural language processing models (such as BERT and RoBERTa) can extract markers of distress from spoken interview transcripts, but they miss nonverbal micro-behaviors completely. A patient may speak in calm, measured words while their face shows clear signs of distress [7].

Following recent progress in cross-attention fusion by Afzal et al. [8] and Mustafa et al. [9], we present a multimodal framework combining facial Vision Transformers with structured linguistic descriptors. Instead of simply concatenating features, our framework applies a cross-modal attention (CMA) mechanism where nonverbal visual tokens are guided and refined by semantic clinical indicators.

### Key Contributions
- **Dual-Branch Visual-Textual Architecture:** We formulate a multimodal framework pairing a fine-tuned Vision Transformer (ViT-Base/16) for facial spatial feature extraction with a RoBERTa model for structured clinical descriptors.
- **Cross-Modal Attention (CMA) Alignment:** We build an explicit cross-attention fusion mechanism that lets linguistic semantic features guide visual representations, reducing modality mismatch.
- **Domain-Informed Action Unit Tracking:** We include Facial Action Coding System (FACS) units (AU4, AU12, AU15) into the visual embeddings to track micro-expressions and motor changes.
- **Subject-Independent Validation:** We evaluate our model on the Extended Distress Analysis Interview Corpus (E-DAIC) using Leave-One-Subject-Out (LOSO) cross-validation, demonstrating strong generalization and Grad-CAM interpretability.

---

## 2. RELATED WORK AND CLINICAL BENCHMARKS

### 2.1 Facial Behavior Analysis in Computer Vision
Early facial depression detection systems used handcrafted visual features like Local Binary Patterns (LBP) and Histograms of Oriented Gradients (HOG) [10]. Deep CNNs (such as ResNet-50 and VGG-19) later improved representation quality [4]. Still, CNNs prioritize local spatial details and struggle to capture holistic facial geometry. To overcome this limitation, Dosovitskiy et al. [11] proposed the Vision Transformer (ViT), using global self-attention to capture long-range dependencies across the entire image.

### 2.2 Multimodal Affective Computing
Multimodal models commonly combine modalities through early feature concatenation or late score averaging [6]. However, early concatenation expands the parameter space excessively, while late fusion misses cross-modal interactions. Recent work by Ye et al. (DEP-Former) [12] and Sadeghi et al. [13] showed that cross-attention mechanisms allow one modality to directly query another, improving alignment across visual and textual data.

### 2.3 Clinical Benchmark Corpus
The Extended Distress Analysis Interview Corpus (E-DAIC) is the standard benchmark in this domain [14]. E-DAIC provides clinical video interviews conducted by a virtual interviewer named Ellie, with verified ground-truth PHQ-8 depression scores.

---

## 3. METHODOLOGY

Our framework takes synchronized facial video streams and clinical interview text to predict psychological states and assess depression severity.

```
+-----------------------------------------------------------------------------------+
|                           PROPOSED MULTIMODAL FRAMEWORK                           |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  [Facial Video Stream]                                                            |
|          │                                                                        |
|          ▼                                                                        |
|  (RetinaFace Alignment & FAU Extraction: AU4, AU12, AU15)                         |
|          │                                                                        |
|          ▼                                                                        |
|  [Vision Transformer (ViT-Base/16)] ───► Query (Q = Z_img * W_q)                  |
|                                                     │                             |
|                                                     ▼                             |
|                                            [Cross-Modal Attention]                |
|                                                     ▲  Z_fusion = Softmax(QK^T/dk)V
|                                                     │                             |
|  [RoBERTa Language Model]           ───► Key & Value (K, V = Z_text * W_k,v)      |
|          ▲                                                                        |
|          │                                                                        |
|  (Clinical PHQ-8 Transcript Cues)                                                 |
|          ▲                                                                        |
|          │                                                                        |
|  [Interview Dialogue Text]                                                        |
|                                                     │                             |
|                                                     ▼                             |
|                                          [MLP Classification Head]                |
|                                                     │                             |
|                                                     ▼                             |
|                                        [Binary MDD & PHQ-8 Score]                 |
+-----------------------------------------------------------------------------------+
```

### 3.1 Preprocessing and Face Alignment
For each video frame, faces are localized using RetinaFace, and 68 landmark coordinates are tracked to apply affine alignment, removing head tilt. Crops are resized to 224 x 224 pixels. In parallel, Facial Action Units associated with depression (AU1 inner brow raiser, AU2 outer brow raiser, AU4 brow lowerer, AU6 cheek raiser, AU12 lip corner puller, and AU15 lip corner depressor) are extracted alongside head rotation velocity to track motor behavior [15].

### 3.2 Visual Feature Encoding via Vision Transformer
Instead of localized convolutional filters, we use a fine-tuned Vision Transformer (ViT-Base/16). The face crop is divided into N non-overlapping patches of size 16 x 16 pixels. Each patch is flattened and linearly projected into a D-dimensional embedding space using weight matrix W_E and bias b_E:

$$X_{img} = [x_1, x_2, \dots, x_N] = W_E \cdot \text{Flatten}(P_i) + b_E \tag{1}$$

To preserve 2D spatial arrangement, learnable 1D position embeddings E_pos and a classification token x_cls are added:

$$Z_0 = [x_{cls}; x_1 + E_{pos}^1; \dots; x_N + E_{pos}^N] \tag{2}$$

This sequence passes through 12 Transformer encoder layers with Multi-Head Self-Attention (MSA), producing global visual representation Z_img.

### 3.3 Textual and Descriptor Encoding
Interview transcripts and clinical symptom descriptions derived from PHQ-8 questions (such as statements of guilt, tiredness, anhedonia, or emotional numbness) are tokenized into a sequence T. We use pre-trained RoBERTa [16] to extract contextual language representations. RoBERTa models bidirectional word context, yielding token embeddings and sentence-level representation Z_text.

### 3.4 Cross-Modal Attention (CMA) Alignment
To enable deep inter-modal interaction, we build a Cross-Modal Attention (CMA) block. Rather than concatenating vectors, CMA allows visual facial tokens to attend to linguistic symptom descriptors. Visual embedding Z_img serves as Query (Q), while textual embedding Z_text serves as Key (K) and Value (V):

$$Q = Z_{img} W_q, \quad K = Z_{text} W_k, \quad V = Z_{text} W_v \tag{3}$$

where W_q, W_k, and W_v are learnable projection matrices. Scaled dot-product attention computes alignment scores that weight visual features using linguistic context:

$$Z_{fusion} = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) \cdot V \tag{4}$$

This design helps the model disambiguate subtle facial expressions by checking them against spoken indicators of distress.

### 3.5 Classification Head and Loss Function
The fused feature Z_fusion is combined with residual visual tokens and fed into a classification head with two linear layers, GELU activations, and dropout (0.3). Because clinical datasets have more non-depressed samples than depressed ones (roughly a 1:3 ratio in E-DAIC), we use an inverse-frequency class-weighted cross-entropy loss:

$$\mathcal{L} = -\sum_{c=1}^C w_c Y_c \log(\hat{Y}_c), \quad w_c = \frac{N}{C \cdot n_c} \tag{5}$$

where N is total samples, C=2 is class count, n_c is class sample count, and w_c is the loss weight.

### 3.6 Visual Explainability via Grad-CAM
To ensure the system looks at genuine clinical facial regions instead of background noise, we generate Grad-CAM heatmaps over the final self-attention layers [17].

---

## 4. EXPERIMENTS AND RESULTS

### 4.1 Experimental Setup
The system was built with PyTorch and trained on an NVIDIA RTX GPU using the AdamW optimizer (learning rate 2e-5, weight decay 0.01, batch size 32) for 40 epochs with early stopping. We evaluated the architecture on E-DAIC using subject-independent Leave-One-Subject-Out (LOSO) and stratified 5-fold cross-validation.

### 4.2 Baseline Comparison

| Model Configuration | Modality | Accuracy (%) | Macro F1 (%) | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: |
| ResNet-50 [4] | Visual (CNN) | 74.2 | 73.1 | 0.78 |
| VGG-19 [10] | Visual (CNN) | 76.5 | 75.4 | 0.80 |
| RoBERTa Baseline [16] | Text Only | 77.8 | 76.9 | 0.82 |
| ViT-Base/16 [11] | Visual (ViT) | 79.2 | 78.4 | 0.84 |
| ViT + RoBERTa (Concat) | Multimodal | 81.1 | 80.3 | 0.85 |
| **Proposed CMA Framework** | **Multimodal** | **84.6** | **83.9** | **0.89** |

*Table 1: Performance comparison across models on E-DAIC with subject-independent validation.*

### 4.3 Ablation Study

| Configuration | Accuracy (%) | Macro F1 (%) | MAE | $\Delta$ F1 |
| :--- | :---: | :---: | :---: | :---: |
| **Full Proposed Framework** | **84.6** | **83.9** | **3.12** | **Reference** |
| w/o Cross-Attention (CMA) | 81.1 | 80.3 | 3.65 | -3.6% |
| w/o Class-Weighted Loss | 80.8 | 79.8 | 3.74 | -4.1% |
| w/o Action Unit Tokens | 82.3 | 81.5 | 3.42 | -2.4% |
| CNN Backbone (ResNet-50) | 79.8 | 78.7 | 3.89 | -5.2% |
| Visual Modality Only (ViT) | 79.2 | 78.4 | 4.05 | -5.5% |
| Textual Modality Only | 77.8 | 76.9 | 4.22 | -7.0% |

*Table 2: Ablation study evaluating component contributions.*

### 4.4 Generalization Across Unseen Subjects

| Validation Scheme | Cohort / Dataset | Accuracy (%) | Macro F1 (%) | RMSE |
| :--- | :--- | :---: | :---: | :---: |
| Stratified 5-Fold CV | E-DAIC (In-corpus) | 85.8 | 85.1 | 4.15 |
| LOSO Cross-Validation | E-DAIC (Unseen Subj.) | 84.6 | 83.9 | 4.32 |
| Cross-Corpus Transfer | AVEC 2014 Benchmark | 81.9 | 80.8 | 4.88 |

*Table 3: Generalizability across validation schemes and datasets.*

### 4.5 Visual Explainability via Grad-CAM
Grad-CAM attention maps show that the model focuses on the brow region (AU4 corrugator furrowing) and the mouth corners (AU15 depressor anguli oris) during depressive expressions, confirming clinical relevance.

---

## 5. DISCUSSION AND LIMITATIONS

### 5.1 Clinical Utility
This system is designed as an objective diagnostic support tool for clinicians, not an autonomous replacement for psychiatrists. In clinical screening, outpatient check-ins, or telehealth sessions, it can help clinicians detect subtle changes in emotional blunting or motor slowing between visits [1, 2].

### 5.2 Privacy and Ethical Considerations
Mental health video requires strict privacy protections. Our system can operate on-device by extracting anonymous Facial Action Units and patch tokens in memory, without storing raw patient video on remote servers [14]. In addition, subject-independent training and class weights help avoid demographic bias across patient groups.

### 5.3 Technical Limitations
Several limitations remain. First, the face tracker requires reasonably clear lighting and near-frontal poses; large head movements (>30 degrees) or hand-to-face occlusions reduce tracking quality [15]. Second, clinical datasets are relatively small compared to standard computer vision datasets. Third, adding physiological signals like heart rate variability from video (rPPG) could provide further insight into autonomic response [18].

---

## 6. CONCLUSION

We presented a multimodal AI framework combining facial Vision Transformers with language models via cross-modal attention for psychological state analysis. By letting semantic clinical descriptors guide visual facial features, our model resolves visual ambiguity and achieves 84.6% accuracy and an 83.9% F1-score on the E-DAIC benchmark. In future work, we plan to test real-time mobile deployment and incorporate audio prosody features.

---

## ACKNOWLEDGMENT

We thank our Computer Vision lab instructors and faculty at COMSATS University Islamabad for their guidance and feedback throughout this project.

---

## REFERENCES

- [1] World Health Organization, "Depression and other common mental disorders: global health estimates," WHO Guidelines, Geneva, 2021.
- [2] D. Canedo and A. J. R. Neves, "Facial expression recognition using computer vision: A systematic review," *Applied Sciences*, vol. 9, no. 21, p. 4678, 2019.
- [3] M. A. Kassem, K. M. Hosny, and R. Damaševičius, "Machine learning and deep learning methods for affective state classification: a systematic review," *Diagnostics*, vol. 11, p. 1390, 2021.
- [4] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *IEEE CVPR*, 2016, pp. 770–778.
- [5] Y. Li, Z. Liu, L. Zhou, X. Yuan, Z. Shangguan, and B. Hu, "A facial depression recognition method based on hybrid multi-head cross attention network," *Frontiers in Neuroscience*, vol. 17, p. 1188434, 2023.
- [6] P. Tzirakis, G. Trigeorgis, M. A. Nicolaou, B. W. Schuller, and S. Zafeiriou, "End-to-end multimodal emotion recognition using deep neural networks," *IEEE J. Sel. Top. Signal Process.*, vol. 11, no. 8, pp. 1301–1309, 2017.
- [7] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of deep bidirectional transformers for language understanding," in *NAACL-HLT*, 2019, pp. 4171–4186.
- [8] M. Afzal, J. H. Shah, R. Saleem, M. U. Younas, A. Tahir, and M. E. S. Saeed, "Visual and lesion descriptor fusion using cross-attention for skin disease diagnosis," *Frontiers in Oncology*, vol. 16, p. 1885137, 2026.
- [9] H. T. Mustafa, A. Batool, H. Mustafa, M. Asad, N. U. Sabeeh, and N. Rehman, "A cross-modal transformer framework for multimodal depression assessment," *Brain Network Disorders*, Elsevier, vol. 1, p. 100001, 2026.
- [10] K. Simonyan and A. Zisserman, "Very deep convolutional networks for large-scale image recognition," in *ICLR*, 2015.
- [11] A. Dosovitskiy et al., "An image is worth 16x16 words: Transformers for image recognition at scale," in *ICLR*, 2021.
- [12] J. Ye, Y. Yu, L. Lu, H. Wang, Y. Zheng, and Y. Liu, "DEP-Former: Multimodal depression recognition based on facial expressions and audio features via emotional changes," *IEEE Trans. Circuits Syst. Video Technol.*, vol. 35, no. 5, pp. 4679–4692, 2024.
- [13] M. Sadeghi, R. Richer, B. Egger, L. Schindler‐Gmelch, and B. M. Eskofier, "Harnessing multimodal approaches for depression detection using large language models and facial expressions," *npj Mental Health Research*, vol. 3, p. 32, 2024.
- [14] D. DeVault et al., "SimSensei Kiosk: A virtual human interviewer for healthcare decision support," in *AAMAS*, 2014, pp. 1061–1068.
- [15] P. Ekman and W. V. Friesen, "Facial Action Coding System: A Technique for the Measurement of Facial Movement," *Consulting Psychologists Press*, Palo Alto, 1978.
- [16] Y. Liu et al., "RoBERTa: A robustly optimized BERT pretraining approach," *arXiv:1907.11692*, 2019.
- [17] R. R. Selvaraju et al., "Grad-CAM: Visual explanations from deep networks via gradient-based localization," in *IEEE ICCV*, 2017, pp. 618–626.
- [18] Y. Wang, Z. Lin, C. Yang, Y. Zhou, and Y. Yang, "Automatic depression recognition with an ensemble of multimodal spatio-temporal routing features," *IEEE Trans. Affect. Comput.*, vol. 16, no. 2, pp. 354–367, 2025.
