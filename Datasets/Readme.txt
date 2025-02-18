### **Updated README for the Project**

---

# **Examining Temporal Issues in EEG-based BCI: Detecting Imagined Words in Continuous EEG Signals**  

This repository contains the code and datasets for the research work on **detecting imagined words from continuous EEG signals using coherence-based features (Mean Phase Coherence (MPC) & Magnitude-Squared Coherence (MSC)) and Word2Vec-based similarity measures**.

---



## **📝 Project Overview**
This project aims to **examine the temporal characteristics of EEG-based BCI systems** and **detect imagined words from continuous EEG signals** using signal coherence measures and word similarity techniques.  

### **🔹 Key Components**
1. **EEG Data Preprocessing**  
   - Raw EEG signals are cleaned, filtered, and segmented.  
   - Epoching and artifact removal are applied.  

2. **Feature Extraction**  
   - **Mean Phase Coherence (MPC)**: Captures phase synchronization between channels.  
   - **Magnitude-Squared Coherence (MSC)**: Measures linear frequency-based coherence.  

3. **Word Representation**  
   - **Word2Vec embeddings** are used to compute the **semantic similarity** of predicted words.  
   - Levenshtein distance is used for further refinement.  

4. **Classification & Prediction**  
   - **Traditional ML models** (SVM, Random Forest) are trained on extracted coherence features.  
   - Model performance is evaluated using **accuracy, precision, recall, and F1-score**.  

---

## **📥 Required Datasets**
### **1️⃣ EEG Data**
- **Download EEG Datasets** (containing real-time data from 8 subjects) from:  
  🔗 [Google Drive Link](https://drive.google.com/drive/folders/1heMlFK8lZ9fpNG1KbXtmPzm1Y12Olgj2?usp=sharing)  
- This dataset contains **multi-channel EEG recordings** collected for imagined speech tasks.

### **2️⃣ Word2Vec Pretrained Model**
- Download **GoogleNews-vectors-negative300.bin** from:  
  🔗 [Kaggle Dataset Link](https://www.kaggle.com/datasets/leadbest/googlenewsvectorsnegative300)  
- Required for **Word2Vec-based word similarity calculations**.


---

## **📌 Notes**
- The dataset includes both **raw EEG** and **coherence feature matrices**.  
- The project **does NOT use deep learning models (Transformers)** but instead **focuses on traditional machine learning techniques** leveraging scientifically grounded EEG coherence measures.

---

## **🔗 References**
1. [Mean Phase Coherence in EEG](https://www.frontiersin.org/articles/10.3389/fnins.2021.656625/full)  
2. [Magnitude-Squared Coherence in Brain Networks](https://ieeexplore.ieee.org/document/9257383)  
3. [Word2Vec for Semantic Similarity](https://arxiv.org/abs/1301.3781)  

---

## **👨‍💻 Contributors**
- **[Anmol Agarwal]** - Primary Researcher  
- **[Aditya Mushyam]** - Supervisor  


---

## **✅ Final Summary**
This project applies **coherence-based EEG feature extraction (MPC & MSC) and classical ML models** to detect imagined words in EEG signals. The pipeline includes **data preprocessing, feature extraction, classification, and evaluation**. The ultimate goal is to provide a robust **non-deep-learning alternative** for EEG-based word prediction.  
