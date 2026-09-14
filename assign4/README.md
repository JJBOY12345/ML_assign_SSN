# 📂 Assignment 4: Logistic Regression Binary Classification

## 📌 Problem Overview & Objectives
This assignment focuses on constructing, evaluating, and validating a binary classification pipeline using **Logistic Regression** on high-dimensional tabular dataset attributes.

Key objectives:
1. Develop a standardized pre-processing pipeline utilizing `StandardScaler` and `SimpleImputer`.
2. Train a baseline Logistic Regression binary classifier.
3. Perform 5-Fold Cross-Validation to evaluate model generalization.
4. Assess classification performance using Confusion Matrix, Precision, Recall, F1-Score, and ROC-AUC curve metrics.

---

## 📊 Dataset Information

* **File Location:** `data/spambase_csv.csv`.
* **Sample Count:** 4,601 instances.
* **Feature Count:** 57 continuous attributes (word/character frequency percentages and capital letter run lengths).
* **Target Label:** Binary classification (`0` = Ham / Non-Spam, `1` = Spam).

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
```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

### Step 3: Run Jupyter Notebook
```bash
jupyter notebook Assign4.ipynb
```

---

## ⚠️ Common Errors & Troubleshooting

### 1. `lbfgs failed to converge (status=1): STOP: TOTAL NO. of ITERATIONS REACHED LIMIT`
* **Cause:** `LogisticRegression` default optimizer (`lbfgs`) exceeded 100 iterations on unscaled data.
* **Solution:** Standardize features using `StandardScaler()` prior to model training, and set `max_iter=1000`:
  ```python
  from sklearn.linear_model import LogisticRegression
  model = LogisticRegression(max_iter=1000)
  ```

### 2. `FileNotFoundError: data/spambase_csv.csv`
* **Cause:** Notebook executed outside the `assign4/` directory.
* **Solution:** Verify that your terminal current working directory is inside `assign4/` before launching `jupyter notebook`.

---

## 🔬 Key Methodologies & Workflow

1. **Preprocessing Pipeline:**
   * Missing value imputation using median strategy (`SimpleImputer(strategy='median')`).
   * Feature standardization using z-score normalization (`StandardScaler`).
2. **Model Training & Cross-Validation:**
   * Stratified train-test split (80% train, 20% test).
   * Evaluated model metrics across 5 folds (`cross_val_score` / `cross_validate`).
3. **Performance Metrics (`images/`):**
   * **Accuracy:** Percentage of correctly classified emails.
   * **Precision:** Ratio of true spam among predicted spam ($\frac{TP}{TP + FP}$).
   * **Recall:** Ratio of detected spam out of total actual spam ($\frac{TP}{TP + FN}$).
   * **ROC-AUC:** Area under the Receiver Operating Characteristic curve.

---

## 📁 Directory File Structure
```text
assign4/
├── README.md                           # Assignment documentation
├── Assign4.ipynb                       # Primary Jupyter Notebook
├── Assign4_Report_Template.tex         # LaTeX report source
├── Experiment_4.pdf                    # Lab manual PDF
├── data/spambase_csv.csv               # SpamBase dataset CSV
└── images/                             # Class distribution & confusion matrix plots
```
