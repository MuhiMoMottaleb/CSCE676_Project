# Cracking Credit Card Fraud: Supervised Learning vs Anomaly Detection

CSCE 676 Course Project

Spring 2026


## 1. Overview

This project investigates credit card fraud detection using the Kaggle “Credit Card Fraud Detection” dataset. With only ~0.17% of transactions labeled as fraud, the core challenge is extreme class imbalance and the high cost of missing fraudulent events. We compare cost-aware supervised classifiers (class-weighted Random Forest, logistic regression) with anomaly detection methods (One-Class SVM and an autoencoder) to understand which approaches work best and whether anomaly detectors add value beyond a strong supervised baseline.

---

## 2. Main Notebook

👉 The main deliverable is: **`main_notebook.ipynb`** (https://github.com/MuhiMoMottaleb/CSCE676_Project/blob/56d6197a965cd7697771d8da652a5639adddf91f/notebooks/main_notebook.ipynb)

This Colab-based notebook contains:
- EDA and dataset description
- RQ1–RQ3 experimental setup and results
- Model comparisons of Supervised Classifiers (Logistics Regression, Decision Tree, Random Forest) vs Anomaly Detectors (One-Class SVM, Autoencoder)
- Final analysis and summary sections

---

## 3.🎥 Project Video

👉 **Project pitch video:** *[(https://www.youtube.com/watch?v=lDUp9OGqfaw)]*  

---

## 4. Research Questions

**RQ1 (Supervised Classification):**  
How do different supervised classifiers (Logistic Regression, Decision Tree, Random Forest) trade off precision, recall, and cost-sensitive performance for fraud detection under varying decision thresholds?

**RQ2 (Clustering):**  
Do fraud and non-fraud transactions exhibit distinct cluster structures in the PCA feature space, and can clustering help identify subtypes of fraudulent transactions? 

**RQ3 (External – One-Class SVM vs Autoencoder):**  
How do kernel-based (One-Class SVM) and neural (autoencoder) anomaly detection methods compare in detecting fraudulent transactions, and do they flag complementary sets of frauds relative to a supervised Random Forest classifier?

---

## 4. Framework Setup for Each RQs

Each of the RQs have the following framework Setup block diagram to show each RQ was conducted, as depicted below. All of the RQs and their blocks have common things (such as loading creditcard.csv) that aren't repeateadly called out, for brevity purposes. More detail is in the `main_notebook.ipynb` file.

**RQ1 Framework Setup**
<img width="1000" height="250" alt="image" src="https://github.com/user-attachments/assets/2a082036-6156-4d66-97a1-27977ba79735" />

**RQ2 Framework Setup**
<img width="1000" height="250" alt="image" src="https://github.com/user-attachments/assets/b2d40c3d-d3e1-4c95-ba12-590bc87794b1" />

**RQ3 Framework Setup**
<img width="1000" height="350" alt="image" src="https://github.com/user-attachments/assets/5c2ad250-b222-49be-8281-80f687a3fef2" />


---

## 5. Data

**Dataset:**  
- **Name:** Credit Card Fraud Detection  
- **Source:** Kaggle – https://www.kaggle.com/mlg-ulb/creditcardfraud  
- **Shape:** 284,807 transactions × 31 columns 
- **Features:**  
  - `Time`: seconds since first transaction  
  - `Amount`: transaction amount  
  - `V1`–`V28`: PCA-transformed features  
  - `Class`: 1 = fraud, 0 = non-fraud (492 frauds ≈ 0.17%) 

**Preprocessing steps:** 

- Removed no columns; dataset has no missing values.
- Split into **train/validation/test** with stratification:
  - 60% train, 20% validation, 20% test (fraud rate preserved in each split).  
- Standardized all numeric features (`Time`, `Amount`, `V1–V28`) using `StandardScaler`:
  - Required for scale-sensitive models (logistic regression, k-means, One-Class SVM, autoencoder).
- For clustering, created a subsample:
  - All frauds + 10× random sample of non-frauds (fraud rate ≈ 9.1%)
- For One-Class SVM:
  - Trained on a 50,000-sample subset of non-fraud transactions for computational feasibility 
- For the autoencoder:
  - Trained only on non-fraud training data to model “normal” behavior 

---

## 6. How to Reproduce

This project was developed and run in **Google Colab** using Python and standard ML libraries.

To run this project successfully, perform the steps defined below.

**Steps:**

1. **Download Credit Card Fraud Data**
   - Download data from https://www.kaggle.com/mlg-ulb/creditcardfraud
2. **Load downloaded Credit Card fraud data (`creditcard.csv`) into Google Colab directory**
3. **Mount Google Drive**
   - Ensure that the creditcard.csv file is in the same directory as the `main_notebook.ipynb` file. If not, the `main_notebook.ipynb` will NOT run. 
4. **Run the `main_notebook.ipynb` file**
   - The `main_notebook.ipynb` file contains the setup for importing methods/functions/calls, setups, and dependencies. No need to run individual sections, the notebook is self contained.
5. **Reproduce Results**
   - Ensure `RANDOM_STATE` is fixed (as in the notebook) to get comparable metrics and plot.

---

## 7. Key Dependencies and Versions
- Python: 3.12.13 
- Jupyter / Colab environment  
- pandas: 2.2.2 
- numpy: 2.0.2 
- scikit-learn: 1.6.1 (RandomForestClassifier, OneClassSVM, metrics, clustering) 
- imbalanced-learn: 0.14.1
- matplotlib: 3.10.0
- seaborn: 0.13.12 (plots & KDEs)
- Keras: 3.12.2 (autoencoder implementation)
- TensorFlow: 2.19.0

See `requirements.txt` file for full dependency list

---

## 8. Repository Structure

```text
CSCE676_Project/
├─ checkpoints/
│  ├─ checkpoint_1.ipynb     # Early EDA/notebook
│  └─ checkpoint2_2.ipynb    # RQ formation and finalization
├─ noteboooks/
│  └─ main_notebook.ipynb    # Main notebook for Project with results and analysis
├─ data/
│  └─ creditcard.md          # provides location to the creditcard.csv file, due to the .csv file being > 100 MB (~150 MB)
├─ requirements.txt          # Full dependency list
├─ LICENSE                   # Provides copyright and permission terms and conditions
└─ README.md                 # This file
```

---

## 9. Results Summary

- Supervised Random Forest (on test set):
   - ROC-AUC ≈ 0.953, PR-AUC ≈ 0.845 
   - At threshold 0.1%: precision ≈ 0.77, recall ≈ 0.86, F1 ≈ 0.81, with low expected cost per transaction under a cost ratio where false negatives are ten times more costly than false positives.

- Anomaly detectors (on test set):
   - One-Class SVM: ROC-AUC ≈ 0.94, PR-AUC ≈ 0.27.  
   - Autoencoder: ROC-AUC ≈ 0.96, PR-AUC ≈ 0.65.

- Complementarity (top 1% most suspicious transactions):
   - *Class-Weighted Random Forest detected 88 frauds*, *Autoencoder detected 81 frauds*, and *One-Class detected SVM 79 frauds*.  
   - All frauds flagged by the anomaly detectors at this top 1% are also found by the Class Weighted Random Forest; the Class Weighted Random Forest finds about 5 additional frauds that anomaly models missed.
     
