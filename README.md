# 🎓 Machine Learning Coursework (Semester 5)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter Notebook](https://img.shields.io/badge/jupyter-notebook-orange.svg)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F79A3E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Welcome to the **Machine Learning Coursework** repository for Semester 5. This repository serves as a comprehensive academic record of machine learning algorithms, exploratory data analysis (EDA), regression techniques, classification models, ensemble learning, dimensionality reduction (PCA), and unsupervised clustering.

* **Author:** JJBOY12345 ([jeswinjoel1088@gmail.com](mailto:jeswinjoel1088@gmail.com))
* **Institution:** Department of Computer Science & Engineering, SSN College of Engineering
* **Course:** Machine Learning (Semester 5)

---

## 📁 Repository Directory Structure

```text
machine_learning/
├── README.md                                   # Comprehensive repository documentation
├── requirements.txt                            # Root python dependencies file
├── format.tex                                  # Master LaTeX formatting template
├── assign1/                                    # Assignment 1: EDA & Preprocessing
│   ├── Assign1.ipynb                           # Notebook
│   ├── Assign1_Report_Template.tex             # LaTeX report source
│   ├── Assign1_ml.pdf                          # Compiled PDF report
│   ├── images/                                 # Generated plots & visualizations
│   └── [datasets: Iris, Loan, Diabetes, Digits, Email Spam]
├── assign2/                                    # Assignment 2: KNN & Naive Bayes
│   ├── Assign2.ipynb                           # Notebook
│   ├── Assign2_Report_Template.tex             # LaTeX report source
│   ├── Experiment_2.pdf                        # Compiled PDF report
│   ├── images/                                 # Confusion matrices, ROC/PR curves
│   └── emailSpam/ / spamEmail/                 # Email Spam datasets
├── assign3/                                    # Assignment 3: Regression Analysis
│   ├── Assign3.ipynb                           # Notebook
│   ├── Assign3_Report_Template.tex             # LaTeX report source
│   ├── Experiment_3_Lab_Manual.pdf             # Lab manual PDF
│   ├── images/                                 # Residual & coefficient plots
│   └── predict_loan_amount/                    # Train & Test CSV files
├── assign4/                                    # Assignment 4: Logistic Regression
│   ├── Assign4.ipynb                           # Notebook
│   ├── Assign4_Report_Template.tex             # LaTeX report source
│   ├── Experiment_4.pdf                        # Lab manual PDF
│   ├── images/                                 # Class distribution plots
│   └── data/spambase_csv.csv                   # SpamBase dataset
├── assign5/                                    # Assignment 5: Decision Trees & Random Forests
│   ├── ex5.ipynb                               # Notebook
│   ├── Experiment_5.pdf                        # Lab manual PDF
│   └── datasets/                               # WDBC Breast Cancer dataset
├── assign6/                                    # Assignment 6: Ensemble Methods (Bagging/Boosting/Stacking)
│   ├── ex6.ipynb                               # Notebook
│   ├── Experiment_6.pdf                        # Lab manual PDF
│   └── datasets/                               # WDBC Breast Cancer dataset
├── assign7/                                    # Assignment 7: PCA & 10-Model Comparative Benchmark
│   ├── exp7.ipynb                              # Notebook
│   ├── Academics_SSN (2).pdf                   # Lab manual PDF
│   └── datasets/                               # WDBC Breast Cancer dataset
└── assign8/                                    # Assignment 8: Unsupervised Clustering (HAR)
    ├── ex8.ipynb                               # Notebook
    ├── Experiment_8.pdf                        # Lab manual PDF
    └── dataset/                                # UCI Human Activity Recognition (HAR) dataset
```

---

## 🛠️ Environment Setup & Virtual Environment Guide

Follow these step-by-step instructions to create an isolated Python virtual environment (`.venv`), install all required dependencies, and launch the Jupyter Notebook environment.

### Prerequisites
* **Python 3.8** or higher installed on your system.
* **Git** installed for version control.

### Step 1: Clone the Repository
```bash
git clone https://github.com/JJBOY12345/ML_assign_SSN.git
cd ML_assign_SSN
```

### Step 2: Create a Virtual Environment (`.venv`)
Isolate project dependencies from your global Python environment:

* **Linux / macOS:**
  ```bash
  python3 -m venv .venv
  ```
* **Windows (Command Prompt / PowerShell):**
  ```cmd
  python -m venv .venv
  ```

### Step 3: Activate the Virtual Environment

* **Linux / macOS:**
  ```bash
  source .venv/bin/activate
  ```
* **Windows (Command Prompt):**
  ```cmd
  .venv\Scripts\activate.bat
  ```
* **Windows (PowerShell):**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```

> *When activated, your terminal prompt will show `(.venv)` at the beginning.*

### Step 4: Upgrade Pip & Install Dependencies

Install all required libraries using the provided [`requirements.txt`](./requirements.txt):

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Alternatively, you can install the packages manually:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn xgboost scipy jupyter
```

### Step 5: Launch Jupyter Notebook
```bash
jupyter notebook
```
Navigate to any assignment folder (e.g., `assign1/Assign1.ipynb`) to view and execute the notebooks.

---

## 📌 Master Assignment Index

| # | Assignment Title | Key Algorithms & Methods | Primary Dataset(s) | Notebook Link | Report PDF |
| :-: | :--- | :--- | :--- | :-: | :-: |
| **01** | **Exploratory Data Analysis & Preprocessing** | EDA, Imputation, Summary Stats, Heatmaps, Pairplots | Iris, Loan Approval, Diabetes, Digits, Spam | [`Assign1.ipynb`](./assign1/Assign1.ipynb) | [`Assign1_ml.pdf`](./assign1/Assign1_ml.pdf) |
| **02** | **Classification with KNN & Naive Bayes** | KNN (KDTree vs BallTree), Naive Bayes, GridSearchCV, ROC/PR | Email Spam (`emails.csv`, `spambase.csv`) | [`Assign2.ipynb`](./assign2/Assign2.ipynb) | [`Experiment_2.pdf`](./assign2/Experiment_2.pdf) |
| **03** | **Regression Analysis & Regularization** | Linear, Ridge, Lasso, Elastic Net, Bias-Variance Tradeoff | Predict Loan Amount (`train.csv`) | [`Assign3.ipynb`](./assign3/Assign3.ipynb) | [`Assign3_Report_Template.tex`](./assign3/Assign3_Report_Template.tex) |
| **04** | **Logistic Regression Binary Classification** | Logistic Regression, `StandardScaler`, 5-Fold CV | SpamBase (`spambase_csv.csv`) | [`Assign4.ipynb`](./assign4/Assign4.ipynb) | [`Experiment_4.pdf`](./assign4/Experiment_4.pdf) |
| **05** | **Decision Trees & Random Forests** | Decision Tree vs Random Forest, Tree Depth, Feature Importance | WDBC Breast Cancer (`wdbc.data`) | [`ex5.ipynb`](./assign5/ex5.ipynb) | [`Experiment_5.pdf`](./assign5/Experiment_5.pdf) |
| **06** | **Ensemble Learning (Bagging, Boosting & Stacking)** | Bagging, AdaBoost, Gradient Boosting, Stacking Ensemble | WDBC Breast Cancer (`wdbc.data`) | [`ex6.ipynb`](./assign6/ex6.ipynb) | [`Experiment_6.pdf`](./assign6/Experiment_6.pdf) |
| **07** | **PCA Dimensionality Reduction & 10-Model Benchmark** | PCA (95% Variance), 10 Classifiers Benchmark (With/Without PCA) | WDBC Breast Cancer (`wdbc.data`) | [`exp7.ipynb`](./assign7/exp7.ipynb) | [`Academics_SSN (2).pdf`](./assign7/Academics_SSN%20%282%29.pdf) |
| **08** | **Unsupervised Clustering on HAR Data** | K-Means, DBSCAN, Hierarchical Agglomerative, Internal/External Metrics | UCI Human Activity Recognition (HAR) | [`ex8.ipynb`](./assign8/ex8.ipynb) | [`Experiment_8.pdf`](./assign8/Experiment_8.pdf) |

---

## 🔍 Detailed Assignment Breakdowns

### 📂 Assignment 1: Exploratory Data Analysis & Data Preprocessing
* **Objective:** Establish an end-to-end data processing pipeline for loading, cleaning, exploring, and visualizing raw datasets prior to machine learning model training.
* **Datasets Analyzed:**
  1. *Iris Dataset*: 150 samples, 4 features (sepal/petal dimensions), 3 species classes.
  2. *Loan Approval Dataset*: Financial and credit attributes for loan decisioning.
  3. *Diabetes Prediction Dataset*: Clinical features for diabetes risk assessment.
  4. *Handwritten Digits Dataset*: $8 \times 8$ pixel array grayscale images ($64$ features).
  5. *Email Spam Dataset*: Word frequency attributes for spam detection.
* **Key Tasks & Methods:**
  * Missing value detection (`isNull()`) and strategy-based imputation (`SimpleImputer`).
  * Identification and removal of duplicate rows.
  * Statistical summary computation (mean, standard deviation, median, IQR).
  * Outlier visualization via boxplots; feature distribution plots via histograms.
  * Correlation matrix calculation and heatmap visualization (`seaborn`).
* **Notebook:** [`assign1/Assign1.ipynb`](./assign1/Assign1.ipynb)

---

### 📂 Assignment 2: Classification using KNN & Naive Bayes
* **Objective:** Implement and optimize K-Nearest Neighbors (KNN) and Naive Bayes classifiers for text document classification and benchmark spatial indexing algorithms.
* **Dataset:** Email Spam Dataset (`emails.csv`, `spambase_csv.csv`).
* **Key Tasks & Methods:**
  * **Custom Algorithm Implementation:** Built KNN from scratch using Euclidean distance calculation over word frequency vectors.
  * **Spatial Indexing Benchmark:** Evaluated training/prediction runtime performance comparing `KDTree` vs `BallTree` spatial data structures in `scikit-learn`.
  * **Naive Bayes Variants:** Implemented and compared Gaussian Naive Bayes (`GaussianNB`), Multinomial Naive Bayes (`MultinomialNB`), and Bernoulli Naive Bayes (`BernoulliNB`).
  * **Hyperparameter Tuning & Evaluation:** Used `GridSearchCV` and `RandomizedSearchCV` to optimize $K$ neighbors. Plotted multi-model ROC and Precision-Recall curves under 5-Fold Cross Validation.
* **Notebook:** [`assign2/Assign2.ipynb`](./assign2/Assign2.ipynb)

---

### 📂 Assignment 3: Regression Analysis & Regularization (Loan Amount Prediction)
* **Objective:** Build predictive linear models for continuous target variables (loan amount prediction) and prevent overfitting via regularization techniques.
* **Dataset:** Predict Loan Amount dataset (`train.csv`, `test.csv`).
* **Key Tasks & Methods:**
  * **Data Leakage Prevention:** Enforced strict train/test splitting *before* performing feature scaling (`StandardScaler`), categorical encoding, and imputation.
  * **Model Implementations:** Trained Ordinary Least Squares (OLS) Linear Regression, Ridge Regression ($L_2$), Lasso Regression ($L_1$), and Elastic Net Regression ($L_1 + L_2$).
  * **Regularization Optimization:** Grid search over regularization strength ($\alpha$) and Elastic Net mixing ratio ($l_1\text{-ratio}$).
  * **Diagnostic Analyses:** Built coefficient shrinkage comparison tables, analyzed residual error plots, and evaluated the Bias-Variance tradeoff.
* **Notebook:** [`assign3/Assign3.ipynb`](./assign3/Assign3.ipynb)

---

### 📂 Assignment 4: Logistic Regression Binary Classification
* **Objective:** Develop a baseline binary classifier using Logistic Regression on high-dimensional text data and evaluate classification stability.
* **Dataset:** SpamBase Dataset (`spambase_csv.csv`, 4,601 instances, 57 continuous attributes).
* **Key Tasks & Methods:**
  * Feature standardization using `StandardScaler` to ensure numerical convergence during gradient descent.
  * Missing value imputation using median strategy (`SimpleImputer`).
  * 5-Fold Cross Validation evaluation.
  * Evaluation via Confusion Matrix, Classification Report (Accuracy, Precision, Recall, F1-Score), and ROC-AUC metrics.
* **Notebook:** [`assign4/Assign4.ipynb`](./assign4/Assign4.ipynb)

---

### 📂 Assignment 5: Decision Trees & Random Forests
* **Objective:** Conduct a comparative study between single decision trees and ensemble random forests, focusing on hyperparameter influence and overfitting control.
* **Dataset:** Wisconsin Diagnostic Breast Cancer (`wdbc.data`, 569 samples, 30 numerical features).
* **Key Tasks & Methods:**
  * Class label encoding (Malignant $\to 1$, Benign $\to 0$) and removal of non-predictive identifier columns.
  * Decision Tree hyperparameter tuning (`max_depth`, `min_samples_split`, `criterion='gini'/'entropy'`) using 5-Fold Cross-Validation.
  * Random Forest hyperparameter tuning (`n_estimators`, `max_features`, `bootstrap`) using 5-Fold Cross-Validation.
  * Analysis of tree depth vs. model variance, bootstrap aggregation benefits, feature importance ranking, and ROC curve comparison.
* **Notebook:** [`assign5/ex5.ipynb`](./assign5/ex5.ipynb)

---

### 📂 Assignment 6: Ensemble Learning (Bagging, Boosting & Stacking)
* **Objective:** Evaluate advanced ensemble paradigms—Bagging, Boosting (AdaBoost & Gradient Boosting), and Stacking—on complex classification tasks.
* **Dataset:** Wisconsin Diagnostic Breast Cancer (`wdbc.data`).
* **Key Tasks & Methods:**
  * **Bagging Classifier:** Implemented Bootstrap Aggregating with decision tree base estimators.
  * **Boosting Classifiers:** Implemented AdaBoost (`AdaBoostClassifier`) and Gradient Boosting (`GradientBoostingClassifier`).
  * **Stacking Ensemble:** Built a heterogeneous multi-layer stacking model combining base estimators (Decision Tree, KNN, Logistic Regression) with a meta-learner classifier.
  * **Hyperparameter Optimization:** Conducted Grid Search over ensemble size, learning rates, and base estimator configurations. Formulated comparative evaluation tables.
* **Notebook:** [`assign6/ex6.ipynb`](./assign6/ex6.ipynb)

---

### 📂 Assignment 7: Dimensionality Reduction (PCA) & 10-Model Benchmark
* **Objective:** Analyze the impact of Principal Component Analysis (PCA) on predictive performance, computational complexity, and feature variance across 10 distinct machine learning classifiers.
* **Dataset:** Wisconsin Diagnostic Breast Cancer (`wdbc.data`, 30 continuous features).
* **Key Tasks & Methods:**
  * **PCA Analysis:** Applied PCA to standardized features. Identified that **10 principal components** retain **95.21%** of total dataset variance (reducing dimensionality from 30 to 10).
  * **Comprehensive 10-Model Benchmark:** Trained, hyperparameter-tuned (5-fold CV), and evaluated 10 classifiers under two settings: **Without PCA** vs **With PCA**:
    1. Support Vector Machine (SVM)
    2. Naive Bayes
    3. K-Nearest Neighbors (KNN)
    4. Logistic Regression
    5. Decision Tree
    6. Random Forest
    7. AdaBoost
    8. Gradient Boosting
    9. XGBoost (`XGBClassifier`)
    10. Stacking Classifier
  * **Evaluation:** Side-by-side metric comparison (Accuracy, Precision, Recall, F1-Score, ROC-AUC), confusion matrices, and ROC/Precision-Recall curves.
* **Notebook:** [`assign7/exp7.ipynb`](./assign7/exp7.ipynb)

---

### 📂 Assignment 8: Unsupervised Clustering on Human Activity Recognition Data
* **Objective:** Apply partition-based, density-based, and hierarchical clustering algorithms to high-dimensional sensor telemetry data and evaluate cluster quality using internal and external metrics.
* **Dataset:** UCI Human Activity Recognition (HAR) dataset ($561$ features derived from smartphone accelerometer/gyroscope sensors).
* **Key Tasks & Methods:**
  * **K-Means Clustering:** Evaluated cluster counts $K \in [2, 8]$; analyzed inertia elbow curves and silhouette scores.
  * **DBSCAN Clustering:** Performed grid search over neighborhood radius (`eps`) and minimum samples (`min_samples`) to identify core vs noise points.
  * **Hierarchical Agglomerative Clustering:** Generated dendrograms using Ward link distance to select optimal cluster cutoffs.
  * **Metric Evaluation:**
    * *Internal Metrics:* Silhouette Score, Calinski-Harabasz Index, Davies-Bouldin Index.
    * *External Metrics:* Adjusted Rand Index (ARI), Normalized Mutual Information (NMI).
  * **Visualization:** Projected high-dimensional clusters onto 2D space using PCA component plots.
* **Notebook:** [`assign8/ex8.ipynb`](./assign8/ex8.ipynb)

---

## 📦 Package & Dependencies Reference

| Package | Purpose & Usage in Repository |
| :--- | :--- |
| **`numpy`** | Numerical array operations, distance metrics, mathematical functions |
| **`pandas`** | Dataframe loading, data cleaning, statistical summaries, CSV I/O |
| **`scikit-learn`** | Core ML library: estimators, preprocessing, decomposition (PCA), cross-validation, metrics |
| **`matplotlib`** | Plot generation: ROC curves, PR curves, residual plots, dendrograms |
| **`seaborn`** | Statistical visualizations: correlation heatmaps, boxplots, pairplots |
| **`xgboost`** | Extreme Gradient Boosting classifier implementation (Assignment 7 benchmark) |
| **`scipy`** | Hierarchical clustering linkage matrices and dendrogram visualization (Assignment 8) |
| **`jupyter`** | Interactive environment for executing `.ipynb` notebooks |

---

## ✅ Progress & Completion Tracker

- [x] **Assignment 1:** Exploratory Data Analysis & Preprocessing
- [x] **Assignment 2:** Classification using KNN & Naive Bayes
- [x] **Assignment 3:** Regression Analysis & Regularization
- [x] **Assignment 4:** Logistic Regression Binary Classification
- [x] **Assignment 5:** Decision Trees & Random Forests
- [x] **Assignment 6:** Ensemble Learning (Bagging, Boosting, Stacking)
- [x] **Assignment 7:** PCA Dimensionality Reduction & 10-Model Benchmark
- [x] **Assignment 8:** Unsupervised Clustering on HAR Data

---

## 📄 Lab Reports & LaTeX Templates

Every assignment folder contains a custom LaTeX report template (e.g., [`assign1/Assign1_Report_Template.tex`](./assign1/Assign1_Report_Template.tex)) formatted using the repository's master styling configuration [`format.tex`](./format.tex). Compiled PDF versions of completed lab reports are available directly in each assignment directory.

---

## 📜 License & Acknowledgments

This repository is licensed under the [MIT License](https://opensource.org/licenses/MIT). Datasets originate from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php) and standard open-source datasets.
