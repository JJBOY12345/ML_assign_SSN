# 📂 Assignment 3: Regression Analysis & Regularization

## 📌 Problem Overview & Objectives
This assignment investigates linear regression techniques and regularization methods to predict continuous numerical targets (**Loan Amount Prediction**) while preventing overfitting and data leakage.

Key objectives:
1. **Prevent Data Leakage:** Ensure scaling, imputation, and encoding fit strictly on training data before transforming test data.
2. **Implement Regularized Linear Models:** Train Ordinary Least Squares (OLS) Linear Regression, Ridge ($L_2$), Lasso ($L_1$), and Elastic Net ($L_1 + L_2$).
3. **Hyperparameter Tuning:** Optimize regularization penalty ($\alpha$) and Elastic Net mixing ratio ($l_1\text{-ratio}$).
4. **Model Diagnostics:** Analyze coefficient shrinkage tables, residual distribution plots, and the Bias-Variance tradeoff.

---

## 📊 Dataset Information

* **Directory:** `predict_loan_amount/` (`train.csv` & `test.csv`).
* **Target Variable:** `loan_amount` (Continuous numerical target).
* **Predictor Features:** Applicant income, credit score, education status, employment duration, loan term, asset evaluations.

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
jupyter notebook Assign3.ipynb
```

---

## ⚠️ Common Errors & Troubleshooting

### 1. `ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations.`
* **Cause:** Lasso or Elastic Net did not converge within default `max_iter=1000` for small values of $\alpha$ or unscaled features.
* **Solution:** Standardize continuous features with `StandardScaler()` and set `max_iter=5000` or `10000` in `Lasso(max_iter=10000)` / `ElasticNet(max_iter=10000)`.

### 2. Data Leakage Violation
* **Cause:** Calling `scaler.fit_transform(X)` on the full dataset prior to `train_test_split()`.
* **Solution:** Always perform `X_train, X_test = train_test_split(X)` first. Fit scalers and imputers *only* on `X_train`, then call `.transform()` on `X_test`.

### 3. Negative $R^2$ Score on Test Set
* **Cause:** Excessive regularization ($\alpha$ too large) causing extreme underfitting, or unscaled feature magnitudes.
* **Solution:** Tune $\alpha$ on a logarithmic scale (`np.logspace(-4, 2, 20)`) using `RidgeCV`, `LassoCV`, or `GridSearchCV`.

---

## 🔬 Key Methodologies & Workflow

1. **Preprocessing Pipeline:**
   * Median imputation via `SimpleImputer` for numeric features.
   * Categorical encoding via `OneHotEncoder` / `OrdinalEncoder`.
   * Feature standardization via `StandardScaler`.
2. **Model Training & Comparison:**
   * **Linear Regression:** Baseline OLS model (unregularized).
   * **Ridge Regression:** Penalizes sum of squared coefficients ($\alpha \sum \beta_j^2$).
   * **Lasso Regression:** Penalizes sum of absolute coefficients ($\alpha \sum |\beta_j|$); performs feature selection by shrinking coefficients to zero.
   * **Elastic Net:** Combines $L_1$ and $L_2$ penalties ($\alpha \cdot l_1 \sum |\beta_j| + \frac{1-l_1}{2} \alpha \sum \beta_j^2$).
3. **Diagnostic Analysis (`images/`):**
   * **Coefficient Shrinkage Plots:** Visualized feature weight decay as $\alpha$ increases.
   * **Residual Plots:** Checked homoscedasticity and zero-mean residual distributions.
   * **Predicted vs Actual Plots:** Evaluated regression alignment.

---

## 📁 Directory File Structure
```text
assign3/
├── README.md                           # Assignment documentation
├── Assign3.ipynb                       # Primary Jupyter Notebook
├── Assign3_Report_Template.tex         # LaTeX report template
├── Experiment_3_Lab_Manual.pdf         # Lab manual reference PDF
├── predict_loan_amount/                # Train and Test CSV dataset folder
└── images/                             # Coefficient plots, residual plots & error curves
```
