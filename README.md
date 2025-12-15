# Bachelor's Thesis: Recommendation Systems

**Comparative Study of Recommendation Algorithms with Production-Ready Two-Tower Model**

Author: Bartek | Date: December 2024

---

## 🎯 Project Overview

This project implements and compares **5 recommendation algorithms** for movie recommendations, with a focus on building a production-ready **Two-Tower Model** suitable for deployment with TensorFlow Serving and Qdrant vector database.

### Implemented Models

| Model                              | Type          | Framework               | Description                               |
| ---------------------------------- | ------------- | ----------------------- | ----------------------------------------- |
| **Collaborative Filtering**        | Memory-Based  | scikit-surprise         | SVD-based matrix factorization            |
| **Content-Based**                  | Feature-Based | TensorFlow + BERT       | Uses movie overviews with BERT embeddings |
| **KNN**                            | Memory-Based  | scikit-surprise         | Item-based K-Nearest Neighbors            |
| **Neural Collaborative Filtering** | Deep Learning | TensorFlow              | GMF + MLP fusion architecture             |
| **Two-Tower Model**                | Deep Learning | TensorFlow Recommenders | Query + Candidate tower retrieval         |

### Datasets

-   **MovieLens 32M**: 32 million ratings, 87,585 movies, 200,948 users (1995-2023)
-   **TMDB 2023**: Movie metadata with 930k+ movies (overviews, genres, budgets)

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Install dependencies with uv
uv sync

# Activate virtual environment
source .venv/bin/activate

# Install pre-commit hooks
pre-commit install
```

### 2. Download Datasets

```bash
# Download MovieLens 32M (~900MB)
bash model/data/download_movielens.sh

# Download TMDB 2023 (~270MB)
bash model/data/download_tmdb.sh
```

### 3. Preprocess Data

```bash
export TEST_MODE=false # to preprocess large dataset (32ml)
export TEST_MODE=true # to preprocess smaller dataset (latest_ml)
python -m model.src.data.preprocessing
```

This creates train/val/test splits in `model/data/processed/`.

### 4. Train Models

```bash
export TEST_MODE=false # to train on large dataset (32ml)
export TEST_MODE=true # to train on smaller dataset (latest_ml)
python ./train_all_model.py
```

### 5. Evaluate Models

```bash
# Run full evaluation suite
python -m model.src.evaluation.run_evaluation

# View results
cat model/metrics/results.json
```

### 6. Generate Thesis Plots

```bash
# Create all visualizations
python -m model.src.visualization.generate_plots

# Plots saved to model/plots/
```

---

## 📁 Project Structure

```
bachelors/
├── INSTRUCTIONS.md              # Master plan (detailed guide)
├── README.md                    # This file
├── pyproject.toml               # Dependencies
│
└── model/
    ├── data/
    │   ├── raw/                 # Downloaded datasets
    │   ├── processed/           # Train/val/test splits
    │   ├── download_movielens.sh
    │   └── download_tmdb.py
    │
    ├── src/
    │   ├── data/
    │   │   └── preprocessing.py  # Data pipeline
    │   ├── models/
    │   │   ├── base.py
    │   │   ├── collaborative.py
    │   │   ├── content_based.py
    │   │   ├── knn.py
    │   │   ├── ncf.py
    │   │   ├── two_tower.py
    │   │   └── export_two_tower.py
    │   ├── evaluation/
    │   │   ├── metrics.py
    │   │   └── run_evaluation.py
    │   └── visualization/
    │       ├── plots.py
    │       └── generate_plots.py
    │
    ├── plots/                    # Thesis figures
    ├── metrics/                  # Evaluation results
    ├── experiments/              # MLflow tracking
    └── saved_models/             # Exported models
```

---

## 📊 Evaluation Metrics

### Regression (Rating Prediction)

-   **RMSE** (Root Mean Squared Error)
-   **MAE** (Mean Absolute Error)

### Ranking (Top-K Recommendations)

-   **Precision@K** - Fraction of relevant items in top-K
-   **Recall@K** - Fraction of relevant items retrieved
-   **NDCG@K** - Normalized Discounted Cumulative Gain (position-aware)
-   **MRR** - Mean Reciprocal Rank

---

## 🎨 Thesis Visualizations

Generated plots (300 DPI, publication-quality):

-   `rmse_comparison.png` - RMSE across all models
-   `ndcg_comparison.png` - NDCG@10 comparison
-   `recall_at_k.png` - Recall@K curves
-   `learning_curves_ncf.png` - NCF training curves
-   `learning_curves_two_tower.png` - Two-Tower training curves
-   `embedding_tsne.png` - t-SNE of movie embeddings
-   `rating_distribution.png` - Rating histogram
-   `genre_distribution.png` - Genre frequencies

---

## 📝 Development

### MLflow Tracking

```bash
mlflow server --port 5000

# View at http://localhost:5000
```

---
