# 📂 Assignment 6: Ensemble Learning (Bagging, Boosting & Stacking)

## 📌 Problem Overview & Objectives
This assignment investigates advanced **Ensemble Learning** paradigms—Bagging, Boosting, and Stacking—to evaluate how combining multiple estimators improves predictive accuracy and reduces variance.

Key objectives:
1. **Bagging (Bootstrap Aggregating):** Implement `BaggingClassifier` with decision tree base estimators to reduce variance.
2. **Boosting (Sequential Ensembles):** Implement `AdaBoostClassifier` and `GradientBoostingClassifier` to iteratively focus on misclassified samples and reduce bias.
3. **Stacking (Heterogeneous Meta-Learning):** Implement `StackingClassifier` combining diverse base estimators (Decision Tree, KNN, Logistic Regression) using a meta-learner.
4. **Grid Search Optimization:** Tune ensemble size (`n_estimators`), learning rates, and meta-estimator parameters.

---

## 📊 Dataset Information

* **Dataset Name:** Wisconsin Diagnostic Breast Cancer (WDBC).
* **Location:** `datasets/wdbc.data`.
* **Sample Count:** 569 instances.
* **Feature Dimensions:** 30 continuous features.
* **Target Classes:** Malignant (`M` $\to 1$) vs Benign (`B` $\to 0$).

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
jupyter notebook ex6.ipynb
```

---

## ⚠️ Common Errors & Troubleshooting

### 1. `ValueError: Final estimator must be fitted.` in Stacking Classifier
* **Cause:** Passing uninitialized or improperly formatted base estimators into `StackingClassifier(estimators=...)`.
* **Solution:** Provide base models as a list of tuples `[('dt', DecisionTreeClassifier()), ('knn', KNeighborsClassifier()), ('lr', LogisticRegression())]` and pass a valid meta-learner like `final_estimator=LogisticRegression()`.

### 2. Extremely Long Execution Times during Grid Search
* **Cause:** Performing exhaustive Grid Search over deep trees, large `n_estimators` (e.g. 500+), and multiple cross-validation folds simultaneously.
* **Solution:** Enable parallel execution by setting `n_jobs=-1` in `GridSearchCV`:
  ```python
  grid = GridSearchCV(estimator, param_grid, cv=5, n_jobs=-1)
  ```

### 3. Gradient Boosting Overfitting
* **Cause:** High `n_estimators` coupled with high `learning_rate` (e.g. 1.0).
* **Solution:** Use a smaller learning rate (`learning_rate=0.05` or `0.1`) combined with early stopping or lower tree depth (`max_depth=3`).

---

## 🔬 Key Methodologies & Workflow

1. **Preprocessing & Stratified Splitting:**
   * Encoded class labels (`M` $\to 1$, `B` $\to 0$).
   * Standardized input features using `StandardScaler`.
2. **Model Architectures:**
   * **Bagging:** Resampled dataset with replacement; aggregated parallel decision tree predictions by majority vote.
   * **AdaBoost:** Reweighted misclassified instances across iterations using weak learners.
   * **Gradient Boosting:** Built sequential trees fitting residual errors of previous predictions using gradient descent optimization.
   * **Stacking:** Used 5-fold cross-validation predictions of base models as input features to train the final meta-classifier.
3. **Comprehensive Evaluation Table:**
   * Metrics recorded: Accuracy, Precision, Recall, F1-Score, ROC-AUC across Bagging, AdaBoost, Gradient Boosting, and Stacking.

---

## 📁 Directory File Structure
```text
assign6/
├── README.md                           # Assignment documentation
├── ex6.ipynb                           # Primary Jupyter Notebook
├── Experiment_6.pdf                    # Lab manual PDF
└── datasets/                           # WDBC dataset folder
    └── wdbc.data                       # Dataset file (569 rows)
```
