# **Examining Temporal Issues in EEG-based BCI: Predicting Imagined Words from EEG Signals**

This repository presents the **latest advancements in EEG-based Brain-Computer Interfaces (BCI) for imagined speech decoding**. It explores the use of **coherence-based EEG features**—namely **Mean Phase Coherence (MPC)** and **Magnitude-Squared Coherence (MSC)**—for predicting imagined words from EEG signals. Our approach integrates **deep learning models (Transformer, CNN-LSTM, Vanilla MLP)** to **improve classification accuracy** over previous studies.

---

## **Background & Motivation**

- **Why Imagined Speech BCI?**
  - Traditional BCIs often focus on motor imagery (e.g., moving a cursor). **Imagined speech** is an alternative paradigm, allowing users to control devices **without movement**—critical for individuals with **ALS, paralysis, or speech disabilities**.
  - However, decoding imagined speech from EEG is **extremely challenging** due to low signal-to-noise ratio and inter-subject variability.
  
- **Our Approach**
  - We extract **connectivity-based features** (MPC & MSC) to analyze neural synchronization.
  - We evaluate **deep learning architectures (Transformer, CNN-LSTM, Vanilla MLP)** to find the best classifier for imagined word recognition.
  - We investigate the impact of **PCA-based dimensionality reduction** on model generalization and performance.

---

## **Dataset & Labels**

1. **EEG Data Collection**
   - Recorded from **8 subjects** performing **imagined speech tasks**.
   - Each subject imagines a set of **20 distinct words**.
   - Signals recorded with **Emotiv EPOC+ EEG headset (14 channels, 256Hz sampling rate)**.
   - Data is **epoched** such that each trial corresponds to one imagined word.

2. **Labels**
   - Each EEG trial is labeled with an **integer (0–19)** representing the imagined word.
   - The goal is to **predict the correct word label** given an EEG trial.

3. **Augmentation Strategy**
   - Sliding window segmentation **increases dataset size** to improve model generalization.
   - Data augmentation resulted in **5000 total trials** for training.

4. **Consent & Usage**
   - Data is strictly used for **academic and non-commercial** purposes.

---

## **Project Workflow**

1. **EEG Preprocessing**
   - **Bandpass filtering** (1–50 Hz) to remove unwanted noise.
   - **Artifact removal** using **Independent Component Analysis (ICA)**.
   - **Segmentation & epoching** into meaningful word-based trials.

2. **Feature Extraction (MPC & MSC)**
   - **Mean Phase Coherence (MPC)**: Measures synchronization between EEG channels.
   - **Magnitude-Squared Coherence (MSC)**: Captures linear frequency-based coherence.
   - Extracted features are computed for **alpha, beta, and gamma bands**.
   - PCA reduces feature dimensions from **546 → 110 components**.

3. **Deep Learning Models**
   - **Transformer** (self-attention for long-range dependencies in EEG signals).
   - **CNN-LSTM** (spatial features via CNN, temporal dependencies via LSTM).
   - **Vanilla MLP** (baseline fully-connected network for comparison).
   - Models predict **the imagined word label** from EEG features.

4. **Evaluation Metrics**
   - **Accuracy**: Measures correct predictions.
   - **F1-Score**: Balances precision & recall.
   - **Confusion Matrix**: Analyzes model misclassifications.

---


### **Key Observations**
- **Transformer outperforms CNN-LSTM & MLP** due to its ability to capture **long-range dependencies**.
- **PCA enhances model performance** by removing redundant features and preventing overfitting.
- **MPC & MSC contribute significantly** to EEG-based imagined speech classification by capturing **functional brain connectivity**.

---

<!-- ## **Single Notebook Workflow**

All major steps are contained in **`test_feature.ipynb`**:
1. **Run Notebook**
   - Launch `test_feature.ipynb` in Jupyter.
   - Execute cells sequentially: preprocessing → feature extraction → model training → evaluation.

2. **Load & Preprocess EEG**
   - Load epoched EEG trials.
   - Apply filtering, artifact removal, and segmentation.

3. **Compute MPC & MSC**
   - Calculate coherence metrics for alpha, beta, gamma bands.
   - Store results as `.npy` files.

4. **Train Deep Learning Models**
   - Transformer, CNN-LSTM, and Vanilla MLP architectures.
   - Training for **60 epochs** (optimized for best generalization).

5. **Evaluation**
   - Accuracy, F1-score, confusion matrix analysis.
   - Compare Transformer vs. CNN-LSTM vs. MLP.

--- -->

## **Repository Contents**

- **`test_feature.ipynb` and `{model}.ipynb`**: 
  - End-to-end pipeline: EEG preprocessing → coherence feature extraction → model training → evaluation.

- **Coherence Feature Files (`.npy`)**:
  - Examples: `mpc_alpha.npy`, `msc_alpha.npy`, etc.

- **Dataset**
  - Data **is included in this repo** . Contact for support.

---

## **Setup & Installation**

1. **Clone the Repository**
   ```bash
   git clone https://github.com/fineanmol/EEG-based---BCI.git
   cd EEG-based---BCI
   ```

2. **Install Dependencies**
   ```bash
   pip install numpy scipy scikit-learn matplotlib torch
   ```

3. **Run Experiments**
   - Open `test_feature.ipynb` and execute step-by-step.

---

## **Future Directions**

1. **Enhancing Model Accuracy**
   - Investigate larger Transformer architectures (e.g., BERT-style EEG models).
   - Explore hybrid approaches combining Transformer with CNN-LSTM.

2. **Generalization Across Subjects**
   - Implement **transfer learning** to reduce subject-specific calibration.
   - Increase dataset diversity for better real-world applicability.

3. **Real-Time EEG Processing**
   - Optimize Transformer model for **low-latency classification**.
   - Implement faster feature extraction pipelines for **on-device EEG decoding**.

---

## **References**
1. **Hernández, T.** (2021). [Toward Asynchronous EEG](https://arxiv.org/pdf/2105.04294). *arXiv preprint*.
2. **Mean Phase Coherence (MPC)**: [Frontiers in Neuroscience](https://www.frontiersin.org/articles/10.3389/fnins.2021.656625/full)
3. **Magnitude-Squared Coherence (MSC)**: [IEEE Xplore](https://ieeexplore.ieee.org/document/9257383)
4. **PyTorch** for Transformer implementation: [PyTorch](https://pytorch.org/)

---

## **Contributor**
- **Anmol Agarwal** – Primary Researcher  
- **Dr. Aditya Mushyam** – Supervisor  

---

## **Summary**
This repository provides an end-to-end EEG-based BCI pipeline for **imagined speech recognition**. By leveraging **coherence-based features (MPC & MSC)** and **deep learning models (Transformer, CNN-LSTM, Vanilla MLP)**, we achieve significant **accuracy improvements** over traditional methods. The findings pave the way for **non-invasive BCIs** with **real-world applications in assistive communication and human-computer interaction**.
