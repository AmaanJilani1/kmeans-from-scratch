# 🔵 K-Means Clustering from Scratch — Standard vs Restricted (Medoid)

A from-scratch implementation and comparison of **two K-Means variants** in pure Python/NumPy — one using numerical mean centroids, the other restricting centers to actual data points (medoid-style) — with convergence analysis and error benchmarking.

---

## 📌 Project Overview

Most ML courses teach K-Means by calling `sklearn.KMeans()`. This project goes deeper — implementing both variants **from scratch** using only NumPy, then comparing them on convergence speed, absolute error, and mean-squared error. Understanding the algorithm at this level reveals *why* the standard mean-based approach dominates in practice.

---

## 🧠 Algorithms Implemented

### Standard K-Means (Numerical Centroids)
Centroids are computed as the **arithmetic mean** of all points in a cluster:

$$\mu_k = \frac{1}{|C_k|} \sum_{x \in C_k} x$$

At each iteration, every point is assigned to its nearest centroid, then centroids are recomputed. Converges when centroids stop moving.

### Restricted K-Means (Medoid Centroids)
Centers are constrained to be **actual input data points** — specifically the point within each cluster that minimizes the total distance to all other cluster members:

$$m_k = \arg\min_{x \in C_k} \sum_{x' \in C_k} \|x - x'\|$$

This is conceptually similar to **K-Medoids (PAM)** but implemented from scratch within the K-Means assignment/update loop.

---

## 📊 Results

| Metric | Standard K-Means (Mean) | Restricted K-Means (Medoid) |
|--------|------------------------|------------------------------|
| **Iterations to converge** | ✅ Fewer | More |
| **Total Absolute Error** | ✅ Lower | Higher |
| **Total MSE** | ✅ Lower (~20–25% less) | Higher |

**Conclusion:** Standard K-Means converges faster and produces lower error on both metrics. The restricted variant pays a penalty because the best actual data point in a cluster is rarely as central as the true mean — especially in high-variance clusters.

---

## 🛠️ Technologies Used

| Tool | Purpose |
|------|---------|
| Python 3.x | Core language |
| NumPy | Distance calculations, centroid updates |
| Matplotlib | Cluster visualization side-by-side |
| Scikit-learn | `make_blobs` for synthetic dataset generation only |
| Jupyter Notebook | Interactive environment |

> No `sklearn.KMeans` was used — both algorithms are implemented from scratch.

---

## 📁 Repository Structure

```
kmeans-from-scratch/
│
├── k_means.ipynb    # Full implementation and comparison notebook
└── README.md        # Project documentation
```

---

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/AmaanJilani1/kmeans-from-scratch.git
cd kmeans-from-scratch
```

**2. Install dependencies**
```bash
pip install numpy matplotlib scikit-learn jupyter
```

**3. Launch the notebook**
```bash
jupyter notebook k_means.ipynb
```

> No dataset download needed — data is synthetically generated using `make_blobs`.

---

## 🔄 Algorithm Flow

```
Generate 300 points, 4 clusters (make_blobs)
         ↓
  ┌──────────────────────┬────────────────────────────┐
  │   Standard K-Means   │   Restricted K-Means        │
  │   centroid = mean    │   centroid = best data pt   │
  ├──────────────────────┼────────────────────────────┤
  │ Assign each point    │ Assign each point           │
  │ to nearest centroid  │ to nearest centroid         │
  │        ↓             │          ↓                  │
  │ New centroid =       │ New centroid =              │
  │ mean of cluster      │ medoid of cluster           │
  │        ↓             │          ↓                  │
  │ Repeat until         │ Repeat until                │
  │ convergence          │ convergence                 │
  └──────────────────────┴────────────────────────────┘
         ↓
  Compare: iterations · absolute error · MSE
         ↓
  Side-by-side cluster visualization
```

---

## 📉 Visualizations

- **Side-by-side scatter plots** of final cluster assignments for both algorithms
- Centroids marked with `X` (standard) and `*` (restricted) for easy comparison
- MSE and absolute error annotated directly on plot titles

---

## 💡 Key Insights

- The **mean** is the point that minimizes squared distance to all cluster members — so standard K-Means minimizes MSE by design
- The medoid approach is more **robust to outliers** (centers are always real data points) but sacrifices optimality in clean, well-separated clusters
- Standard K-Means is also faster per iteration since computing a mean is O(n), while finding a medoid is O(n²) within each cluster
- In practice, medoid-based clustering (K-Medoids / PAM) is preferred when **interpretability** matters — e.g., finding a representative customer or city

---

## 👤 Author

**Amaan Jilani**
[GitHub](https://github.com/AmaanJilani1) · [LinkedIn](https://www.linkedin.com/in/amaanjilani/)

---
