# CSCE676_Project
CSCE 676 Course Project
Spring 2026

# Cracking Credit Card Fraud: Supervised Learning vs Anomaly Detection

## 1. Overview

This project investigates credit card fraud detection using the Kaggle “Credit Card Fraud Detection” dataset. With only ~0.17% of transactions labeled as fraud, the core challenge is extreme class imbalance and the high cost of missing fraudulent events. We compare cost-aware supervised classifiers (especially a class-weighted Random Forest) with anomaly detection methods (One-Class SVM and an autoencoder) to understand which approaches work best and whether anomaly detectors add value beyond a strong supervised baseline.

---

## 2. Main Notebook

👉 The main deliverable is: **`main_notebook.ipynb`**  

This Colab-based notebook contains:
- EDA and dataset description
- RQ1–RQ3 experimental setup and results
- Model comparisons (Class Weighted Random Forest, One-Class SVM, Autoencoder)
- Final analysis and summary sections

---

## 3.🎥 Project Video

👉 **Project pitch video:** *[(https://www.youtube.com/watch?v=lDUp9OGqfaw)]*  

---

## 4. Research Questions

**RQ1 (Course – Supervised Classification):**  
How do different supervised classifiers (Logistic Regression, Decision Tree, Random Forest) trade off precision, recall, and cost-sensitive performance for fraud detection under varying decision thresholds?

**RQ2 (Course – Clustering):**  
Do fraud and non-fraud transactions exhibit distinct cluster structures in the PCA feature space, and can clustering help identify subtypes of fraudulent transactions? 

**RQ3 (External – One-Class SVM vs Autoencoder):**  
How do kernel-based (One-Class SVM) and neural (autoencoder) anomaly detection methods compare in detecting fraudulent transactions, and do they flag complementary sets of frauds relative to a supervised Random Forest classifier?

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
  - Required for scale-sensitive models (logistic regression, k-means, One-Class SVM, Isolation Forest, autoencoder).
- For clustering, created a subsample:
  - All frauds + 10× random sample of non-frauds (fraud rate ≈ 9.1%)
- For One-Class SVM:
  - Trained on a 50,000-sample subset of non-fraud transactions for computational feasibility 
- For the autoencoder:
  - Trained only on non-fraud training data to model “normal” behavior 

---

## 6. How to Reproduce

This project was developed and run in **Google Colab** using Python and standard ML libraries.

**Steps:**

1. **Clone the repo:**
   ```bash
   git clone https://github.com/your-username/credit-card-fraud-detection.git
   cd credit-card-fraud-detection

2. **Set up Environment:**
   - Colab:
      - Upload the repo (or mount Drive).
      - Run the initial setup cell in Project.ipynb to install dependencies as needed.
3. **Download the Dataset**
4. **Run the `main_notebook.ipynb` file**
5. **Reproduce Results**
   - Ensure `RANDOM_STATE` is fixed (as in the notebook) to get comparable metrics and plots

See `requirements.txt` file for full dependency list

## 7. Key Dependencies and Versions
- Python: 3.12.13 
- Jupyter / Colab environment  
- pandas: 2.2.2 
- numpy: 2.0.2 
- scikit-learn: 1.6.1 (RandomForestClassifier, OneClassSVM, metrics, clustering, etc.) 
- imbalanced-learn: 0.14.1
- matplotlib: 3.10.0
- seaborn: 0.13.12 (plots & KDEs)
- Keras: 3.12.2 (autoencoder implementation)
- TensorFlow: 2.19.0
## 8. Repository Structure
CSCE676_Project/
├─ README.md                  # This file
├─ requirements.txt           # Full dependency list
├─ data/
│  └─ creditcard.csv         # Kaggle dataset (not committed, user downloads)
├─ notebooks/
│  ├─ Project.ipynb          # Main checkpoint 2/3 deliverable <source_id data="1" title="Project.pdf" />
│  ├─ checkpoint_1.ipynb  # (Optional) early EDA/notebook
│  └─ checkpoint2_2.ipynb   # (Optional) RQ formation if separate

## 9. Results Summary

- Supervised Random Forest (test set):
   - ROC-AUC ≈ 0.953, PR-AUC ≈ 0.845 
   - At threshold 0.1: precision ≈ 0.77, recall ≈ 0.86, F1 ≈ 0.81, with low expected cost per transaction under a cost ratio where false negatives are ten times more costly than false positives.

- Anomaly detectors:
   - One-Class SVM: test ROC-AUC ≈ 0.94, PR-AUC ≈ 0.27.  
   - Autoencoder: test ROC-AUC ≈ 0.96, PR-AUC ≈ 0.65.

- Complementarity (top 1% most suspicious transactions):
   - Class Weighted Random Forest detects 88 frauds, Autoencoder 81, One-Class SVM 79.  
   - All frauds flagged by the anomaly detectors at this top 1% are also found by the Random Forest; the Random Forest finds about 5 additional frauds that anomaly models miss.
     
