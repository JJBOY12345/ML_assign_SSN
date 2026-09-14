# 📂 Assignment 1: Exploratory Data Analysis & Preprocessing

## 📌 Problem Overview & Objectives
The goal of this assignment is to design and execute a comprehensive **Exploratory Data Analysis (EDA)** and **Data Preprocessing** pipeline across multiple datasets representing both classification and regression targets. 

Proper preprocessing ensures data quality by detecting missing values, handling duplicates, analyzing feature distributions, discovering correlations, and identifying potential outliers before model training.

---

## 📊 Datasets Analyzed

| Dataset Name | File Path | Samples / Dimensions | Target Variable / Classes | Key Attributes |
| :--- | :--- | :--- | :--- | :--- |
| **Iris Flower** | `iris/iris.data` | 150 samples, 4 features | `class` (3 classes: Setosa, Versicolor, Virginica) | Sepal/Petal length and width |
| **Loan Approval** | `LoanAmount/loan_approval_dataset.csv` | 4,269 rows, 13 features | `loan_status` (Approved / Rejected) | Income, CIBIL score, loan amount, assets |
| **Diabetes Risk** | `Diabetes_Prediction/diabetes_prediction_dataset.csv` | 100,000 rows, 9 features | `diabetes` (0 / 1) | Age, BMI, HbA1c level, blood glucose |
| **Handwritten Digits** | `digits.csv` | 1,797 samples, 64 features | `target` (Digits 0–9) | $8 \times 8$ pixel grayscale values |
| **Email Spam** | `emailSpam/emails.csv` | 5,172 samples, 3,002 features | `Prediction` (0 = Ham, 1 = Spam) | Word frequency vector representation |

---

## 🛠️ Environment Setup & Installation Guide

### Step 1: Create a Virtual Environment (`.venv`)
Open your terminal in the repository root or inside `assign1/`:

* **Linux / macOS:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```
* **Windows (Command Prompt / PowerShell):**
  ```cmd
  python -m venv .venv
  .venv\Scripts\activate
  ```

### Step 2: Install Required Dependencies
Install all required libraries for Assignment 1:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

### Step 3: Run the Jupyter Notebook
```bash
jupyter notebook Assign1.ipynb
```

---

## ⚠️ Common Errors & Troubleshooting

### 1. `FileNotFoundError: [Errno 2] No such file or directory`
* **Cause:** Running the notebook from a different working directory or missing dataset files.
* **Solution:** Ensure you start Jupyter Notebook from inside the `assign1/` folder, or verify dataset paths:
  * `iris/iris.data`
  * `LoanAmount/loan_approval_dataset.csv`
  * `Diabetes_Prediction/diabetes_prediction_dataset.csv`
  * `digits.csv`
  * `emailSpam/emails.csv`

### 2. `ModuleNotFoundError: No module named 'seaborn'` (or `sklearn`)
* **Cause:** The notebook kernel is executing in a system Python environment where dependencies are missing.
* **Solution:** Ensure your `.venv` is activated before running `jupyter notebook`. Inside Jupyter, check kernel path:
  ```python
  import sys
  print(sys.executable)
  ```

### 3. Missing Column Names in Iris Dataset
* **Cause:** `iris.data` does not contain a header row.
* **Solution:** Load using `pandas.read_csv('iris/iris.data', header=None, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])`.

---

## 🔬 Key Methodologies & Workflow

1. **Data Cleaning & Quality Checks:**
   * Detected and imputed missing values (`SimpleImputer` / median strategy).
   * Identified and removed duplicate rows (`df.drop_duplicates()`).
2. **Statistical Summaries:**
   * Calculated mean, standard deviation, median, and interquartile range (IQR).
3. **Visualization & Outlier Analysis (`images/`):**
   * **Histograms:** Visualized univariate feature distributions across all datasets.
   * **Boxplots:** Detected potential extreme values (e.g., CIBIL score, HbA1c levels, loan amounts).
   * **Correlation Heatmaps:** Evaluated linear relationships using Pearson correlation coefficients.
   * **Pairplots:** Analyzed pairwise scatter relationships between features colored by target class.

---

## 📁 Directory File Structure
```text
assign1/
├── README.md                           # Assignment guide & documentation
├── Assign1.ipynb                       # Primary Jupyter Notebook
├── Assign1_Report_Template.tex         # LaTeX report source
├── Assign1_ml.pdf                      # Compiled PDF assignment report
├── digits.csv                          # Digits dataset
├── file.txt                            # Metadata text file
├── Diabetes_Prediction/                # Diabetes dataset folder
├── LoanAmount/                         # Loan approval dataset folder
├── emailSpam/                          # Email spam dataset folder
├── iris/                               # Iris dataset folder
└── images/                             # Generated EDA plots & visualizations
```
