# Machine Learning Coursework

This repository contains my coursework assignments for the Machine Learning course (Semester 5). The assignments cover a range of ML paradigms including exploratory data analysis, regression, classification (using various algorithms), and model optimization techniques.

The repository is structured modularly so that new assignments can be easily documented and linked here as they are completed.

---

## 📌 Assignment Index

| Assignment | Topic | Key Concepts / Models | Dataset(s) | Notebook Link |
| :--- | :--- | :--- | :--- | :--- |
| **Assignment 1** | Exploratory Data Analysis & Cleaning | EDA, handling missing/duplicate values, data visualization | Iris, Loan Approval, Diabetes, Digits, Email Spam | [Assign1.ipynb](./assign1/Assign1.ipynb) |
| **Assignment 2** | Classification: KNN & Naive Bayes | KNN (KDTree vs BallTree), Naive Bayes, GridSearchCV, ROC comparison | Email Spam Detection | [Assign2.ipynb](./assign2/Assign2.ipynb) |
| **Assignment 3** | Regression Analysis | Linear, Ridge, Lasso, Elastic Net, Bias-Variance, Overfitting | Predict Loan Amount | [Assign3.ipynb](./assign3/Assign3.ipynb) |
| **Assignment 4** | Logistic Regression | Baseline binary classification, standard scaling | SpamBase | [Assign4.ipynb](./assign4/Assign4.ipynb) |
| **Assignment 5** | Decision Trees & Random Forests | Comparative classification study, 5-fold CV hyperparameter tuning | Breast Cancer (WDBC) | [ex5.ipynb](./assign5/ex5.ipynb) |

---

## 🔍 Detailed Assignment Breakdowns

### 📂 Assignment 1: Exploratory Data Analysis & Preprocessing
* **Objective:** Establish a robust pipeline for loading, cleaning, exploring, and visualizing raw datasets to prepare them for machine learning models.
* **Key Tasks & Methods:**
  * Handling missing values (detection and imputation) and duplicate rows.
  * Statistical summaries (mean, median, standard deviation).
  * Feature distributions via histograms, scatter plots, and correlation heatmaps.
  * Applied EDA across five distinct datasets representing both classification and regression targets (Iris, Loan Approval, Diabetes, Handwritten Digits, and Email Spam).
* **Notebook:** [`assign1/Assign1.ipynb`](./assign1/Assign1.ipynb)

### 📂 Assignment 2: Classification using KNN and Naive Bayes
* **Objective:** Implement and optimize K-Nearest Neighbors (KNN) and Naive Bayes classifiers for document classification.
* **Key Tasks & Methods:**
  * Text representation and feature parsing from the Email Spam dataset.
  * Implementation and performance comparison of Naive Bayes variants: Gaussian, Multinomial, and Bernoulli Naive Bayes.
  * Hyperparameter tuning of KNN using `GridSearchCV` and `RandomizedSearchCV`.
  * Algorithmic runtime comparison: Benchmarking Training and Testing times for KNN using `KDTree` versus `BallTree` spatial indexing.
  * Evaluation: 5-Fold Cross-Validation and Multi-Model ROC curve comparison.
* **Notebook:** [`assign2/Assign2.ipynb`](./assign2/Assign2.ipynb)

### 📂 Assignment 3: Regression Analysis (Loan Amount Prediction)
* **Objective:** Predict numerical values using linear models and prevent overfitting using regularization techniques.
* **Key Tasks & Methods:**
  * **Data Leakage Prevention:** Splitting the dataset into train/test sets *before* performing preprocessing steps (imputation, scaling, encoding).
  * Implementation of Linear Regression, Ridge, Lasso, and Elastic Net Regression models.
  * Hyperparameter optimization for regularization strength ($\alpha$) and Elastic Net mixing ratio ($L1$ ratio).
  * Diagnostic analyses: Coefficient comparison table, overfitting vs. underfitting checks, and Bias-Variance tradeoff analysis.
* **Notebook:** [`assign3/Assign3.ipynb`](./assign3/Assign3.ipynb)

### 📂 Assignment 4: Logistic Regression (Spam Classification)
* **Objective:** Build a binary classifier using Logistic Regression on the high-dimensional SpamBase dataset.
* **Key Tasks & Methods:**
  * Standardizing features using `StandardScaler` to ensure numerical stability and correct convergence of gradient descent.
  * Missing value imputation using `SimpleImputer` (median strategy).
  * Training a baseline Logistic Regression model and evaluating binary classification accuracy.
* **Notebook:** [`assign4/Assign4.ipynb`](./assign4/Assign4.ipynb)

### 📂 Assignment 5: Decision Trees & Random Forests
* **Objective:** Perform a comparative study of single-estimator models (Decision Trees) versus ensemble-based estimators (Random Forests).
* **Key Tasks & Methods:**
  * Class label encoding (mapping 'Malignant' to 1 and 'Benign' to 0) and feature selection (dropping non-informative identifier fields).
  * Tuning Decision Tree hyperparameters (max depth, min samples split, splitting criterion) using 5-Fold Cross-Validation.
  * Tuning Random Forest hyperparameters (number of estimators, max features, bootstrap configuration) using 5-Fold Cross-Validation.
  * Model evaluation via confusion matrices, classification reports, and ROC curve comparison.
* **Notebook:** [`assign5/ex5.ipynb`](./assign5/ex5.ipynb)

---

## 🛠️ Environment Setup & Requirements

To run these notebooks locally, ensure you have Python 3.8+ installed. You can install the required libraries using pip:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

### Running the Notebooks
1. Clone the repository:
   ```bash
   git clone https://github.com/JJBOY12345/ML_assign_SSN.git
   cd ML_assign_SSN
   ```
2. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
3. Navigate to any assignment folder and open the `.ipynb` file.
