# **Examining Temporal Issues in EEG-based BCI: Predicting Imagined Words from EEG Signals**

This repository demonstrates how **coherence-based EEG features**—derived from **Mean Phase Coherence (MPC)** and **Magnitude-Squared Coherence (MSC)**—can be used to **predict which word** a participant is imagining from a **small vocabulary**. We utilize a **Transformer-based model** to handle the “prediction” aspect, as recommended for tasks where we decode an actual word and than performing a yes/no classification using multiple classifiers.

---

## **Background & Related Work**

- **Toward Asynchronous EEG**  
  - Our work is inspired by **Dr. Tonatiuh Hernández (Ph.D.)** and his paper, [*Toward Asynchronous EEG*](https://arxiv.org/pdf/2105.04294), which explores asynchronous EEG-based BCI contexts.  

---

## **Dataset & Labels**

1. **Real-Time EEG Data**  
   - Collected from multiple subjects performing imagined speech tasks (up to 8 subjects).  
   - Each subject imagines a small set of words (e.g., 20 distinct words).  
   - Signals are recorded using a standard EEG headset with channels placed according to the 10-20 system.  
   - The dataset is **epoched** such that each trial corresponds to one imagined word.

2. **Labeling**  
   - Each trial is labeled with an **integer** representing the imagined word (e.g., 0–19 if there are 20 words).  
   - Thus, the model’s goal is to **predict** which word label corresponds to each EEG epoch.

3. **Consent & Usage**   
   - Data is used **only** for academic and non-commercial purposes, in line with the original publication’s guidelines.

---

## **Project Overview**

1. **Goal: Predict an Imagined Word**  
   - We aim to **predict the actual word** (e.g., among 10–20 possible words) that a subject imagines, based on EEG signals.

2. **EEG Preprocessing**  
   - **Filtering**: Remove noise and artifacts.  
   - **Epoching**: Segment continuous EEG into trials corresponding to each imagined word.  
   - **Frequency Bands**: Alpha, beta, and gamma band extraction for further analysis.

3. **Feature Extraction (MPC & MSC)**  
   - **Mean Phase Coherence (MPC)**: Measures phase synchronization between EEG channels.  
   - **Magnitude-Squared Coherence (MSC)**: Captures linear frequency-based coherence.  
   - Combined across alpha, beta, gamma → yields a multi-dimensional coherence feature set for each trial.

4. **Prediction Model: A Small Transformer**  
   - Treats each trial’s coherence features as input.  
   - Outputs a **predicted word** from the small vocabulary (e.g., 20 words).  
   - This addresses the professor’s requirement for a “predictor,” as it attempts to decode an actual word from EEG data rather than performing a simple yes/no classification.

---

## **Methodology**

1. **EEG Preprocessing**  
   - Filtering (e.g., 1–50 Hz), artifact removal (e.g., ICA or MARA), and segmentation into alpha, beta, gamma bands.  
   - Epoched signals shape: `(num_trials, num_channels, time_points)` or a similar structure, depending on the pipeline.

2. **Coherence Feature Extraction**  
   - **Mean Phase Coherence (MPC)**: measures phase synchronization across channels.  
   - **Magnitude-Squared Coherence (MSC)**: captures linear frequency coherence across channels.  
   - We compute these measures for alpha, beta, gamma bands, yielding `(num_trials, channels, channels)` arrays per band.  
   - **Flatten** or **combine** (and optionally apply PCA) to create final feature vectors.

3. **Word-Level Prediction (Transformer)**  
   - We train a **Transformer** with 2–3 encoder layers, `d_model ~ 64`, to predict which word label each trial represents.  
   - The model outputs a **single word label** from the possible vocabulary (e.g., 20 words).

---


## **Single Notebook Workflow**
All major steps are contained in **`test_feature.ipynb`**:
1. **Run Notebook**  
   - Launch `test_feature.ipynb` in Jupyter or similar.  
   - Execute cells sequentially: coherence calculation → data reshaping → model training → evaluation.

2. **Load & Preprocess EEG**  
   - Load epoched signals for each imagined word trial.  
   - Filter into alpha, beta, gamma bands if not already done.

3. **Compute MPC & MSC**  
   - Calculate coherence measures channel-by-channel.  
   - Store results in `.npy` arrays (e.g., `mpc_alpha.npy`, `msc_alpha.npy`, etc.).

4. **Combine & Reshape Features**  
   - Merge MPC/MSC across bands into a single feature array.  
   - Optionally **apply PCA** for dimensionality reduction.  
   - Reshape the data if necessary for a Transformer-based approach (e.g., `[num_samples, seq_len, feature_dim]`).

5. **Transformer Training**  
   - Define a **lightweight Transformer** with a couple of encoder layers, small hidden dimension (`d_model`), and an output layer sized to the **number of possible words**.  
   - Train for ~15–25 epochs.  

6. **Prediction & Evaluation**  
   - The model **predicts** which word each EEG trial corresponds to, effectively **decoding** the imagined word.  
  - Trains SVM, Random Forest, Logistic Regression on the same coherence features.
Compares performance with each other.
   - Evaluate performance (accuracy, confusion matrix, etc.).  


---

## **Repository Contents**

- **`test_feature.ipynb`**:  
  - **End-to-end** pipeline: coherence feature extraction → small Transformer “predictor” → performance evaluation.  
  - Logs and shape prints are included to clarify each step.

- **Coherence `.npy` Files**:  
  - Stored locally or generated at runtime.  
  - Examples: `mpc_alpha.npy`, `mpc_beta.npy`, `mpc_gamma.npy`, `msc_alpha.npy`, etc.

- **Labels/Word List**:  
  - A small set of integer-encoded words (e.g., 0–19 if you have 20 possible words).  

---

## **Data & Setup**

1. **EEG Dataset**  
   - Multi-subject recordings for imagined speech tasks.  
   - Not included in this repo; consult the instructions or external drive link to obtain the raw EEG or epoched signals.

2. **Environment Requirements**  
   ```bash
   pip install numpy scipy scikit-learn matplotlib torch
   ```
   - **PyTorch** is required for the Transformer-based predictor.

3. **Running**  
   - Launch `test_feature.ipynb` in Jupyter or similar environment.  
   - Run cells in order, ensuring the `.npy` files for coherence features are generated or available.

---

## **Model Explanation**

### **Small Transformer**  
- **Architecture**: 2–3 TransformerEncoder layers, `d_model ~ 64`, multi-head attention (`nhead=4`).  
- **Sequence Input**: The coherence feature vectors are reshaped as short sequences, allowing the Transformer to process them as a “temporal” or structured input.  
- **Output**: Predicts an integer label indicating the **imagined word** among the small vocabulary.

### **Why a Transformer?**  
- The professor’s requirement is a “predictor” that **outputs a word** (rather than a yes/no label).  
- A Transformer is more flexible for word-level decoding or classification from EEG signals than simpler yes/no classifiers.

---

## **References**

1. **Hernández, T.** (2021). [Toward Asynchronous EEG](https://arxiv.org/pdf/2105.04294). *arXiv preprint*.  
3. **Mean Phase Coherence**: [Frontiers in Neuroscience](https://www.frontiersin.org/articles/10.3389/fnins.2021.656625/full)  
4. **Magnitude-Squared Coherence**: [IEEE Xplore](https://ieeexplore.ieee.org/document/9257383)

---

1. [Mean Phase Coherence (MPC)](https://www.frontiersin.org/articles/10.3389/fnins.2021.656625/full)  
2. [Magnitude-Squared Coherence (MSC)](https://ieeexplore.ieee.org/document/9257383)  
3. [PyTorch](https://pytorch.org/) for building the Transformer-based predictor

---

## **Contributor**
- **[Anmol Agarwal]** – Primary Researcher
- **[Aditya Mushyam]** - Supervisor  

---

## **Summary**
Leveraging the **real-time EEG dataset** from Dr. Tonatiuh Hernández’s “Toward Asynchronous EEG” work, we extract **MPC & MSC** coherence features and employ a **Transformer** to **predict** which word each subject imagines. This approach meets the **word-level decoding** objective, aligning with the professor’s feedback that we need a **predictor** rather than a simple classifier. All core steps are consolidated in **`test_feature.ipynb`**, demonstrating the pipeline from raw signals to final accuracy metrics.