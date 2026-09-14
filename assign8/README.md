# 📂 Assignment 8: Unsupervised Clustering on Human Activity Recognition Data

## 📌 Problem Overview & Objectives
This assignment investigates **Unsupervised Clustering** algorithms applied to high-dimensional telemetry signal data derived from mobile smartphone sensors (Human Activity Recognition).

Key objectives:
1. **K-Means Clustering:** Evaluate cluster counts $K \in [2, 8]$, plot inertia elbow curves, and compute silhouette scores.
2. **DBSCAN Clustering:** Perform parameter sweeps over neighborhood distance (`eps`) and core point threshold (`min_samples`) to identify natural density clusters and isolate noise points.
3. **Hierarchical Agglomerative Clustering (HAC):** Compute linkage matrices, plot dendrogram trees using Ward distance, and select cluster cutoffs.
4. **Metric Validation:** Evaluate cluster quality using **Internal Metrics** (Silhouette Score, Calinski-Harabasz, Davies-Bouldin) and **External Metrics** (Adjusted Rand Index, Normalized Mutual Info).
5. **Dimensionality Reduction & Visualization:** Map high-dimensional cluster centroids onto 2D PCA component spaces.

---

## 📊 Dataset Information

* **Dataset Name:** UCI Human Activity Recognition (HAR) using Smartphones.
* **Location:** `dataset/` (`X_train.txt`, `y_train.txt`, `X_test.txt`, `y_test.txt`, `activity_labels.txt`, `features.txt`).
* **Feature Dimensions:** 561 continuous variables extracted from time and frequency domain sensor signals (triaxial acceleration and angular velocity).
* **Activity Categories (Ground Truth for External Evaluation):**
  1. WALKING (`1`)
  2. WALKING_UPSTAIRS (`2`)
  3. WALKING_DOWNSTAIRS (`3`)
  4. SITTING (`4`)
  5. STANDING (`5`)
  6. LAYING (`6`)

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
> ⚠️ **Note:** Hierarchical dendrogram plotting requires `scipy`.

```bash
pip install numpy pandas scikit-learn matplotlib seaborn scipy jupyter
```

### Step 3: Launch Notebook
```bash
jupyter notebook ex8.ipynb
```

---

## ⚠️ Common Errors & Troubleshooting

### 1. `ModuleNotFoundError: No module named 'scipy'`
* **Cause:** `scipy` is required for hierarchical dendrogram linkage computation (`scipy.cluster.hierarchy`).
* **Solution:** Install scipy inside `.venv`:
  ```bash
  pip install scipy
  ```

### 2. File Parsing Errors when Loading `X_train.txt`
* **Cause:** `X_train.txt` uses variable numbers of whitespace delimiters between columns. `pd.read_csv('X_train.txt', sep=' ')` will fail or create NaN columns.
* **Solution:** Use regex delimiter `sep=r'\s+'` with `header=None`:
  ```python
  X_train = pd.read_csv('dataset/train/X_train.txt', sep=r'\s+', header=None)
  ```

### 3. DBSCAN Assigning All Points to Noise (`-1`)
* **Cause:** Default `eps=0.5` is too small for high-dimensional feature spaces ($D=561$).
* **Solution:** Apply `StandardScaler()` first, or perform PCA dimensionality reduction prior to DBSCAN, and tune `eps` on normalized distance distribution graphs.

### 4. Memory Errors during Hierarchical Clustering Dendrogram Calculation
* **Cause:** Constructing an $N \times N$ distance matrix for thousands of samples requires excessive RAM.
* **Solution:** Subsample training instances or use a random stratified subset (e.g. 1,000–2,000 samples) for dendrogram visualization.

---

## 🔬 Clustering Algorithms & Evaluation Metrics

### 1. Algorithms Evaluated
* **K-Means:** Partitioning algorithm minimizing within-cluster sum of squares (WCSS).
* **DBSCAN:** Density-based spatial clustering identifying dense regions separated by low-density noise.
* **Agglomerative Hierarchical:** Bottom-up tree hierarchy using Ward's minimum variance criterion.

### 2. Metric Benchmark

| Metric Category | Metric Name | Optimal Value Direction | Description |
| :--- | :--- | :-: | :--- |
| **Internal Metric** | **Silhouette Score** | Closer to $+1.0$ | Measures how similar an object is to its own cluster compared to other clusters. |
| **Internal Metric** | **Calinski-Harabasz Index** | Higher is better | Ratio of between-cluster dispersion to within-cluster dispersion. |
| **Internal Metric** | **Davies-Bouldin Index** | Lower is better | Average similarity measure of each cluster with its most similar cluster. |
| **External Metric** | **Adjusted Rand Index (ARI)** | Closer to $+1.0$ | Measures similarity between predicted clusters and true activity labels, adjusted for chance. |
| **External Metric** | **Normalized Mutual Info (NMI)** | Closer to $+1.0$ | Normalizes mutual information shared between cluster assignments and ground truth. |

---

## 📁 Directory File Structure
```text
assign8/
├── README.md                           # Assignment documentation
├── ex8.ipynb                           # Primary Jupyter Notebook
├── Experiment_8.pdf                    # Lab manual PDF
└── dataset/                            # UCI HAR dataset directory
    ├── activity_labels.txt             # Activity ID to label mapping
    ├── features.txt                    # List of 561 sensor features
    ├── test/                           # Test dataset split
    └── train/                          # Train dataset split
```
