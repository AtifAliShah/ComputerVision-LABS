# Selected Reference Research Paper

## Primary Match

### **A cross-modal transformer framework for multimodal depression assessment**

* **Authors:** Hafiz Tayyab Mustafa, Alia Batool, Hamza Mustafa, Mujtaba Asad, Najam Ul Sabeeh, and Nadia Rehman
* **Publication Year:** 2026
* **Journal:** *Brain Network Disorders* (Elsevier)
* **Digital Object Identifier (DOI):** [10.1016/j.bnd.2026.09.001](https://doi.org/10.1016/j.bnd.2026.09.001)
* **Official Publisher Link:** [https://doi.org/10.1016/j.bnd.2026.09.001](https://doi.org/10.1016/j.bnd.2026.09.001)
* **Benchmark Corpus:** Extended Distress Analysis Interview Corpus (E-DAIC)
* **Validation Methodology:** Leave-One-Subject-Out (LOSO) cross-validation

---

### **Technical Summary and Architectural Alignment**

1. **Computer Vision Modality:**
   * Utilizes a compact Vision Transformer (ViT) architecture to model spatial facial representations, micro-expression dynamics, and affective facial cues from interview video sequences.
   * Overcomes the spatial locality constraints of traditional Convolutional Neural Networks (CNNs) by modeling long-range spatial and global affective dependencies across facial regions.

2. **Linguistic and Clinical Modality:**
   * Employs Robustly Optimized BERT Pretraining Approach (RoBERTa) to encode clinical dialogues and interview transcripts into dense semantic contextual embeddings.

3. **Cross-Attention Fusion Mechanism:**
   * Implements a cross-attention late fusion module to model bidirectional interactions between visual facial expressions and linguistic markers, dynamically aligning non-verbal affective behaviors with expressed semantic content.

4. **Relevance to Project ("Psychological State Analysis Using Multimodal AI"):**
   * Directly supports our three core clinical domains:
     * **Affective:** Captures sadness, blunted affect, and emotional dysregulation via facial Vision Transformer features.
     * **Cognitive:** Analyzes verbalized hopelessness, guilt, and negative self-evaluation via RoBERTa embeddings.
     * **Psychomotor:** Measures dynamic motor slowing, facial rigidity, and micro-movements across temporal interview sequences.
   * Directly parallels the architectural blueprint established in our professor's reference paper (Afzal et al., *Frontiers in Oncology*, 2026), translating the dual-branch Vision Transformer + BERT + Cross-Attention fusion paradigm from dermatological diagnostics to automated psychological state evaluation.

---

## Alternative Published Research Papers

### 1. **DEP-Former: Multimodal Depression Recognition Based on Facial Expressions and Audio Features via Emotional Changes**
* **Authors:** Jiayu Ye, Yanhong Yu, Lin Lu, Hao Wang, Yunshao Zheng, and Yang Liu
* **Year:** 2024
* **Venue:** *IEEE Transactions on Circuits and Systems for Video Technology* (TCSVT), Vol. 35, Issue 5, pp. 4679-4692
* **DOI:** [10.1109/TCSVT.2024.3491098](https://doi.org/10.1109/TCSVT.2024.3491098)
* **Official Link:** [https://doi.org/10.1109/TCSVT.2024.3491098](https://doi.org/10.1109/TCSVT.2024.3491098)
* **Relevance:** Introduces a multimodal cross-attention framework that maps facial expressions and acoustic emotional transitions into a shared space, validated on AVEC 2014 with interpretable attention query analysis.

### 2. **Harnessing multimodal approaches for depression detection using large language models and facial expressions**
* **Authors:** Misha Sadeghi, Robert Richer, Bernhard Egger, Lena Schindler‐Gmelch, Lydia Helene Rupp, Farnaz Rahimi, Matthias Berking, and Bjoern M. Eskofier
* **Year:** 2024
* **Venue:** *npj Mental Health Research* (Nature Portfolio), Vol. 3, Article 32
* **DOI:** [10.1038/s44184-024-00112-8](https://doi.org/10.1038/s44184-024-00112-8)
* **Official Link:** [https://doi.org/10.1038/s44184-024-00112-8](https://doi.org/10.1038/s44184-024-00112-8)
* **Relevance:** Parallels the professor's structured clinical descriptor paradigm by extracting domain-informed depression indicators aligned with PHQ-8 symptom criteria from transcripts and fusing them with facial visual features.

### 3. **A facial depression recognition method based on hybrid multi-head cross attention network**
* **Authors:** Yutong Li, Zhenyu Liu, Li Zhou, Xiaoyan Yuan, Zixuan Shangguan, Xiping Hu, and Bin Hu
* **Year:** 2023
* **Venue:** *Frontiers in Neuroscience*, Vol. 17, Article 1188434
* **DOI:** [10.3389/fnins.2023.1188434](https://doi.org/10.3389/fnins.2023.1188434)
* **Official Link:** [https://doi.org/10.3389/fnins.2023.1188434](https://doi.org/10.3389/fnins.2023.1188434)
* **Relevance:** Directly addresses Computer Vision limitations in modeling higher-order spatial interactions across facial sub-regions (eyes, mouth) using a Multi-Head Cross Attention Network (HMHN) on AVEC benchmark datasets.
