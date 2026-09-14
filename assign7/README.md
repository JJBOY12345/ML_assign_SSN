# 📂 Assignment 7: Dimensionality Reduction (PCA) & 10-Model Comparative Benchmark

## 📌 Problem Overview & Objectives
This assignment investigates the empirical impact of **Principal Component Analysis (PCA)** for dimensionality reduction across a broad benchmark of machine learning classifiers.

Key objectives:
1. **PCA Variance Decomposition:** Apply PCA to standardized features to determine the minimum components required to retain **95% of total variance**.
2. **Benchmark 10 Classifiers:** Train and hyperparameter-tune 10 distinct models under two experimental configurations:
   * **Setting 1: Without PCA** (Original 30 continuous features).
   * **Setting 2: With PCA** (Reduced 10 principal components).
3. **Comprehensive Performance Analysis:** Compare Accuracy, Precision, Recall, F1-Score, ROC-AUC curves, PR curves, and confusion matrices across both settings.

---

## 📊 Dataset & PCA Summary

* **Dataset:** Wisconsin Diagnostic Breast Cancer (`datasets/wdbc.data`, 569 instances, 30 features).
* **Target Classes:** Benign (`B` $\to 0$) vs Malignant (`M` $\to 1$).
* **PCA Variance Target:** 95.0%
* **PCA Result:** **10 Principal Components** preserve **95.21%** of total dataset variance, reducing feature dimensionality by 66.7%.

| Setting | Feature Space Dimensions | Explained Variance (%) | Primary Benefit |
| :--- | :-: | :-: | :--- |
| **Without PCA** | 30 original features | 100.00% | Preserves raw feature interpretability. |
| **With PCA** | 10 principal components | 95.21% | Reduces computational overhead, removes multicollinearity, combats curse of dimensionality. |

---

## 🛠️ Environment Setup & Installation Guide

### Step 1: Create & Activate Virtual Environment (`.venv`)

* **Linux / macOS:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```
* **Windows:**
  ```cmd
  python -m venv .venv
  .venv\Scripts\activate
  ```

### Step 2: Install Dependencies
> ⚠️ **Note:** This assignment requires the `xgboost` library in addition to standard ML packages.

```bash
pip install numpy pandas scikit-learn matplotlib seaborn xgboost jupyter
```

### Step 3: Launch Notebook
```bash
jupyter notebook exp7.ipynb
```

---

## ⚠️ Common Errors & Troubleshooting

### 1. `ModuleNotFoundError: No module named 'xgboost'`
* **Cause:** XGBoost is not included in standard `scikit-learn` installations.
* **Solution:** Install XGBoost inside your active virtual environment:
  ```bash
  pip install xgboost
  ```
  Or run inside a notebook cell:
  ```python
  !pip install xgboost
  ```

### 2. PCA Applied Before Feature Standardization
* **Cause:** Running PCA directly on raw features with unaligned scales causes high-variance features to dominate principal components.
* **Solution:** Always apply `StandardScaler()` *before* fitting `PCA(n_components=0.95)`:
  ```python
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  pca = PCA(n_components=0.95)
  X_train_pca = pca.fit_transform(X_train_scaled)
  ```

### 3. `XGBClassifier` Target Label Warnings
* **Cause:** Passing string targets (`'M'`, `'B'`) directly to XGBoost.
* **Solution:** Encode target labels to integers (`0` and `1`) using `LabelEncoder()` or `map({'B': 0, 'M': 1})`.

---

## 🔬 Benchmark Models & Methodologies

The following 10 classifiers were benchmarked with 5-Fold Cross Validation:
1. **Support Vector Machine (SVM):** Tuned `C`, `kernel='rbf'/'linear'`, `gamma`.
2. **Naive Bayes:** Gaussian Naive Bayes baseline.
3. **K-Nearest Neighbors (KNN):** Tuned `n_neighbors`, `weights`, `p`.
4. **Logistic Regression:** Tuned regularization penalty `C` and solver `lbfgs`.
5. **Decision Tree:** Tuned `max_depth`, `min_samples_split`.
6. **Random Forest:** Tuned `n_estimators`, `max_features`, `max_depth`.
7. **AdaBoost:** Tuned `n_estimators`, `learning_rate`.
8. **Gradient Boosting:** Tuned `n_estimators`, `learning_rate`, `max_depth`.
9. **XGBoost Classifier:** Tuned `n_estimators`, `learning_rate`, `max_depth`, `subsample`.
10. **Stacking Classifier:** Meta-model combining SVM, Random Forest, and XGBoost.

---

## 📁 Directory File Structure
```text
assign7/
├── README.md                           # Assignment documentation
├── exp7.ipynb                          # Primary Jupyter Notebook
├── Academics_SSN (2).pdf               # Lab manual PDF
└── datasets/                           # WDBC dataset folder
    └── wdbc.data                       # Dataset file (569 rows)
```
