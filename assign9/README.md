# 📂 Assignment 9: Perceptron vs Multilayer Perceptron (A/B Experiment) with Hyperparameter Tuning

## 📌 Problem Overview & Objectives
This assignment conducts an **A/B experimental comparison** between a fundamental single-layer neural architecture (**Perceptron Learning Algorithm — PLA**) and a deep feedforward architecture (**Multilayer Perceptron — MLP**) on a challenging 62-class image recognition task.

Key research goals & objectives:
1. **Model A (PLA from Scratch):** Build a single-layer Perceptron Learning Algorithm from scratch using a One-vs-Rest (OvR) multi-class strategy and step activation function.
2. **Model B (MLP with Backpropagation):** Train a Multilayer Perceptron using `scikit-learn`'s `MLPClassifier` with non-linear activation functions (ReLU, Tanh) and backpropagation optimization.
3. **Grid Search Hyperparameter Tuning:** Systematically evaluate combinations of hidden layer depth/width, activation functions, optimization algorithms (SGD vs. Adam), and initial learning rates.
4. **Diagnostic & Comparative Analysis:** Evaluate performance using Accuracy, Precision, Recall, F1-Score, Confusion Matrices, Loss Convergence Curves, and Micro/Macro-averaged ROC-AUC curves.

---

## 📊 Dataset Information

* **Dataset Name:** English Handwritten Characters Dataset.
* **Directory Structure:** `dataset/english.csv` & `dataset/Img/`.
* **Total Instances:** 3,410 grayscale images of handwritten characters.
* **Class Count:** **62 classes** (Digits `0–9`, Uppercase `A–Z`, Lowercase `a–z`).
* **Preprocessing Pipeline:**
  * Images loaded via PIL (`Image.open()`) and converted to grayscale.
  * Resized to **$28 \times 28$ pixels**, yielding a **$784$-dimensional** continuous feature vector.
  * Pixel intensity values normalized to $[0, 1]$ via min-max scaling ($x / 255.0$).
  * Class labels integer-encoded ($0 \to 61$) using `LabelEncoder`.
  * Data split into **80% Training Set** ($2,728$ samples) and **20% Test Set** ($682$ samples) using stratified sampling.

---

## 🛠️ Environment Setup & Installation Guide

### Step 1: Create & Activate Virtual Environment (`.venv`)

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
> ⚠️ **Note:** Assignment 9 requires `Pillow` for image reading and preprocessing.

```bash
pip install numpy pandas scikit-learn matplotlib seaborn pillow jupyter
```

### Step 3: Run the Jupyter Notebook
```bash
jupyter notebook Experiment_9_PLA_vs_MLP.ipynb
```

---

## ⚠️ Common Errors & Troubleshooting

### 1. `ModuleNotFoundError: No module named 'PIL'`
* **Cause:** The `Pillow` library is missing in your active environment.
* **Solution:** Install Pillow in your active virtual environment:
  ```bash
  pip install pillow
  ```

### 2. `FileNotFoundError: dataset/english.csv` or Cannot Load Images
* **Cause:** Launching Jupyter Notebook outside the `assign9/` folder or missing dataset subfolders.
* **Solution:** Verify that your working directory is `assign9/` where both `dataset/english.csv` and `dataset/Img/` exist:
  ```python
  import os
  assert os.path.exists('dataset/english.csv'), "english.csv missing!"
  assert os.path.exists('dataset/Img'), "Img directory missing!"
  ```

### 3. Convergence Warnings during PLA Training
* **Cause:** The single-layer Perceptron uses a hard step function and weight update rule $w_{t+1} = w_t + \eta(y - \hat{y})x$. Because 62-class handwritten character features are non-linearly separable, PLA will fail to converge to zero training error.
* **Solution:** Set a maximum epoch limit (e.g., `epochs=100`) for PLA to prevent infinite training loops.

### 4. `ConvergenceWarning: Stochastic Optimizer: Maximum iterations reached` in MLP
* **Cause:** `MLPClassifier` default `max_iter=200` reached before cross-entropy loss flattened.
* **Solution:** Increase max iterations in `MLPClassifier(max_iter=500)` or use `early_stopping=True`.

---

## 🔬 Key Methodologies & Results

### 1. Model A — Perceptron Learning Algorithm (PLA)
* **Architecture:** 62 independent binary perceptrons (One-vs-Rest strategy).
* **Step Activation:** $f(z) = \begin{cases} 1 & \text{if } z \ge 0 \\ 0 & \text{otherwise} \end{cases}$
* **Weight Update:** $w^{(c)} \leftarrow w^{(c)} + \eta (y^{(c)} - \hat{y}^{(c)}) x$
* **Result:** Low classification accuracy (~3-5%). The linear decision boundary cannot separate complex pixel variations across 62 character classes.

### 2. Model B — Multilayer Perceptron (MLP)
* **Architecture:** Feedforward network with hidden layers, non-linear activations, and softmax output layer trained via backpropagation.
* **Hyperparameter Grid Search Space:**
  * `hidden_layer_sizes`: `(64,)`, `(128,)`, `(128, 64)`, `(256, 128, 64)`
  * `activation`: `'relu'`, `'tanh'`
  * `solver`: `'adam'`, `'sgd'`
  * `learning_rate_init`: `0.001`, `0.01`
* **Best Performing MLP Configuration:**
  * `hidden_layer_sizes=(128, 64)`
  * `activation='relu'`
  * `solver='adam'`
  * `learning_rate_init=0.001`
* **Result:** Achieved high accuracy (~70%+ on 62 classes), significantly outperforming PLA.

### 3. Generated Diagnostic Visualizations (`figures/`)
* **`pla_convergence.png`:** Epoch-by-epoch training error curves showing PLA non-convergence.
* **`mlp_loss_curve.png`:** Smooth cross-entropy loss reduction curve over iterations during backpropagation.
* **`confusion_matrices.png`:** $62 \times 62$ class confusion matrices comparing PLA vs best MLP.
* **`roc_curves.png`:** Micro-average and Macro-average ROC-AUC curves showing multi-class discrimination power.
* **`ab_bar_chart.png`:** Side-by-side performance metric comparison bar chart (Accuracy, Precision, Recall, F1-Score, ROC-AUC).

---

## ❓ Observation Questions & Analytical Takeaways

1. **Why does PLA underperform compared to MLP?**
   * PLA is a linear classifier with a step activation. It can only learn hyperplanes and fails when data is non-linearly separable. Handwritten characters require non-linear feature abstractions that MLP captures using hidden layers and non-linear activation functions (ReLU/Tanh).
2. **Which hyperparameters had the greatest impact on MLP?**
   * Activation function (`relu` outperformed `tanh`) and optimizer (`adam` converged faster and more reliably than standard `sgd`).
3. **Did optimizer choice affect convergence?**
   * Yes, `Adam` uses adaptive learning rates for individual parameters, navigating sparse image gradients much more efficiently than basic `SGD`.
4. **Did adding more hidden layers always improve results?**
   * No. Excessively deep architectures (e.g., 4+ layers) increased training time and risk of overfitting without significant gains over a 2-layer `(128, 64)` network.

---

## 📁 Directory File Structure
```text
assign9/
├── README.md                           # Assignment guide & documentation
├── Experiment_9_PLA_vs_MLP.ipynb       # Primary Jupyter Notebook
├── Experiment_9.pdf                    # Lab manual PDF
├── ml_assign9-1.pdf                    # Compiled PDF lab report (J Jeswin Joel)
├── summary_tables.png                  # Summary metrics table image
├── dataset/                            # Dataset directory
│   ├── english.csv                     # Mapping of image files to character labels
│   └── Img/                            # 3,410 handwritten character PNG images
└── figures/                            # Generated experimental plots
    ├── ab_bar_chart.png                # PLA vs MLP performance metric comparison
    ├── confusion_matrices.png          # 62x62 confusion matrices
    ├── mlp_loss_curve.png              # MLP backpropagation loss curve
    ├── pla_convergence.png             # PLA training error curve over epochs
    └── roc_curves.png                  # Micro and macro average ROC curves
```
