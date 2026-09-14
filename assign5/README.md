# 📂 Assignment 5: Decision Trees & Random Forests

## 📌 Problem Overview & Objectives
This assignment presents a comparative study of single-estimator models (**Decision Trees**) versus ensemble-based estimators (**Random Forests**) on a medical diagnostic dataset.

Key research questions & objectives:
1. **Tree Depth & Overfitting:** How does restricting `max_depth` affect variance and generalization performance?
2. **Hyperparameter Sensitivity:** Which hyperparameters (`max_depth`, `n_estimators`, `max_features`, `min_samples_split`) exert the greatest influence on cross-validation stability?
3. **Ensemble Generalization:** How does bootstrap aggregation (bagging) and random feature selection in Random Forests reduce individual tree variance?
4. **Feature Importance:** Extracting feature importance rankings from tree nodes.

---

## 📊 Dataset Information

* **Dataset Name:** Wisconsin Diagnostic Breast Cancer (WDBC).
* **Location:** `datasets/wdbc.data` & `datasets/wdbc.names`.
* **Sample Count:** 569 instances.
* **Attributes:** 30 continuous features derived from digitized images of cell nuclei (radius, texture, perimeter, area, smoothness, compactness, concavity, etc.).
* **Target Label:** Binary (`M` = Malignant $\to 1$, `B` = Benign $\to 0$).

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

### Step 3: Launch Notebook
```bash
jupyter notebook ex5.ipynb
```

---

## ⚠️ Common Errors & Troubleshooting

### 1. `pandas.errors.ParserError` or Shifted Columns when Loading Dataset
* **Cause:** `wdbc.data` is a comma-separated file without a header row. Reading with default `pd.read_csv('datasets/wdbc.data')` treats line 1 as column names.
* **Solution:** Specify `header=None` when loading:
  ```python
  df = pd.read_csv('datasets/wdbc.data', header=None)
  ```

### 2. High Training Accuracy but Low Validation Accuracy (Overfitting)
* **Cause:** Fully grown Decision Trees (`max_depth=None`) memorize training noise.
* **Solution:** Constrain tree growth using `max_depth=3` or `4` and `min_samples_split=5`, or transition to a `RandomForestClassifier`.

### 3. Missing `datasets/` Folder
* **Cause:** Notebook launched from wrong directory.
* **Solution:** Ensure working directory is `assign5/` where `datasets/wdbc.data` exists.

---

## 🔬 Key Methodologies & Workflow

1. **Preprocessing:**
   * Encoded categorical target (`M` $\to 1$, `B` $\to 0$).
   * Dropped identifier column (`ID`).
   * Applied stratified 80/20 train-test split.
2. **Decision Tree Tuning (`DecisionTreeClassifier`):**
   * Grid search over `max_depth` $\in [1, 10]$, `min_samples_split` $\in [2, 10]$, `criterion='gini'/'entropy'`.
   * Evaluated training vs 5-fold cross-validation accuracy curves to pinpoint overfitting thresholds.
3. **Random Forest Tuning (`RandomForestClassifier`):**
   * Grid search over `n_estimators` $\in [10, 200]$, `max_features` $\in ['sqrt', 'log2']$, `bootstrap=True`.
4. **Evaluation:**
   * Confusion Matrix comparison.
   * ROC-AUC and Precision-Recall curve evaluation.
   * Feature importance visualization ranking top predictive attributes.

---

## 📁 Directory File Structure
```text
assign5/
├── README.md                           # Assignment documentation
├── ex5.ipynb                           # Primary Jupyter Notebook
├── Experiment_5.pdf                    # Lab manual PDF
└── datasets/                           # WDBC dataset folder
    ├── wdbc.data                       # Dataset file (569 rows)
    └── wdbc.names                      # Dataset attribute description
```
