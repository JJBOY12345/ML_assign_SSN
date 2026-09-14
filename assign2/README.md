# 📂 Assignment 2: Classification using KNN & Naive Bayes

## 📌 Problem Overview & Objectives
This assignment focuses on implementing and evaluating **K-Nearest Neighbors (KNN)** and **Naive Bayes** classifiers for document/email classification (Spam vs Ham).

Key goals include:
1. Building a **custom KNN algorithm from scratch** using Euclidean distance metrics.
2. Benchmarking **spatial indexing data structures (`KDTree` vs `BallTree`)** in `scikit-learn` for KNN prediction speed.
3. Comparing **Naive Bayes variants**: Gaussian, Multinomial, and Bernoulli Naive Bayes.
4. Performing **hyperparameter tuning** using `GridSearchCV` and `RandomizedSearchCV` with 5-Fold Cross Validation.

---

## 📊 Dataset Information

* **Primary Dataset:** Email Spam Detection (`emailSpam/emails.csv` & `spamEmail/spambase_csv.csv`).
* **Target Classes:** `0` (Ham / Legitimate Email) vs `1` (Spam Email).
* **Feature Representation:** Preprocessed word frequencies across email vocabulary.

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

### Step 2: Install Required Dependencies
```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

### Step 3: Launch Notebook
```bash
jupyter notebook Assign2.ipynb
```

---

## ⚠️ Common Errors & Troubleshooting

### 1. `MemoryError` or Slow Performance during Custom KNN Execution
* **Cause:** Computing pairwise Euclidean distances across high-dimensional text vectors for thousands of samples is computationally expensive ($O(N \cdot D)$).
* **Solution:** Subset training samples during scratch custom KNN testing, or leverage `scikit-learn`'s optimized C-backed `KNeighborsClassifier`.

### 2. Zero Variance / Continuous Feature Errors in `MultinomialNB`
* **Cause:** `MultinomialNB` expects non-negative integer word counts. Standardizing features with `StandardScaler` (producing negative values) will throw an error.
* **Solution:** Use `MinMaxScaler` or feed unscaled word frequencies into `MultinomialNB`, and reserve `StandardScaler` for `GaussianNB`.

### 3. Missing Plot Files in `images/`
* **Cause:** Plot generation cells at the end of `Assign2.ipynb` were not executed.
* **Solution:** Select **Cell -> Run All** in Jupyter Notebook to generate all ROC curves, PR curves, and confusion matrix PNGs.

---

## 🔬 Key Methodologies & Workflow

1. **Custom KNN Implementation:**
   * Calculated Euclidean distance $d(x, y) = \sqrt{\sum_{i=1}^n (x_i - y_i)^2}$.
   * Identified $K$ shortest distance neighbors and assigned the majority class label.
2. **Spatial Indexing Benchmark (`KDTree` vs `BallTree`):**
   * Benchmarking training and prediction execution times for low vs high dimensions.
3. **Naive Bayes Model Benchmarking:**
   * **GaussianNB:** Assumes continuous features follow Gaussian distribution.
   * **MultinomialNB:** Suited for discrete word frequency counts.
   * **BernoulliNB:** Suited for binary word presence/absence indicators.
4. **Hyperparameter Tuning & Visualizations (`images/`):**
   * Optimized $K \in [1, 30]$, distance weights (`uniform` vs `distance`), and Naive Bayes smoothing ($\alpha$).
   * Generated multi-model ROC curves, Precision-Recall curves, and confusion matrices.

---

## 📁 Directory File Structure
```text
assign2/
├── README.md                           # Assignment documentation
├── Assign2.ipynb                       # Primary Jupyter Notebook
├── Assign2_Report_Template.tex         # LaTeX report source
├── Experiment_2.pdf                    # Compiled PDF lab report
├── emailSpam/                          # Email dataset
├── spamEmail/                          # SpamBase dataset
└── images/                             # Generated ROC curves, PR curves & confusion matrices
```
