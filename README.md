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

## � Documentation

-   **[DATA_PREPROCESSING.md](DATA_PREPROCESSING.md)** - Detailed guide to dataset combination and preprocessing
-   **[DATA_PIPELINE.md](DATA_PIPELINE.md)** - Complete data pipeline architecture and flow
-   **[INSTRUCTIONS.md](INSTRUCTIONS.md)** - Master plan and implementation guide
-   **[TEST_MODE.md](TEST_MODE.md)** - Quick testing with small dataset

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

This creates train/val/test splits in `model/data/processed/`. See [DATA_PREPROCESSING.md](DATA_PREPROCESSING.md) for details.

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
    │       ├── generate_plots.py
    │       ├── embedding_plots.py
    │       ├── two_tower_plots.py
    │       ├── eda_plots.py
    │       └── dataset_eda.py
    │
    ├── plots/                    # Thesis figures
    │   ├── models/              # Model comparison plots
    │   ├── embeddings/          # Embedding visualizations
    │   ├── two_tower/           # Two-Tower specific plots
    │   └── eda/                 # Exploratory data analysis
    │       └── datasets/        # Dataset comparison plots
    ├── metrics/                  # Evaluation results + dataset stats
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

### Model Comparisons

-   `models/rmse_comparison.png` - RMSE across all models
-   `models/recall_at_k.png` - Recall@K curves
-   `models/porownanie_predykcji_modeli.png` - Model predictions comparison (Polish)

### Embeddings

-   `embeddings/rated_movies/rated_comparison.png` - Dimensionality reduction comparison (t-SNE, PCA, UMAP)
-   `embeddings/rated_movies/podobienstwo_kosinusowe.png` - Cosine similarity heatmap
-   `embeddings/tmdb_full/tmdb_comparison.png` - TMDB embeddings visualization

### Two-Tower Model

-   `two_tower/architektura_modelu.png` - Model architecture diagram
-   `two_tower/rekomendacje_przykladowe.png` - Example recommendations
-   `two_tower/macierz_podobienstwa.png` - Similarity heatmap

### Dataset Analysis

-   `eda/datasets/porownanie_rozmiarow.png` - Dataset sizes comparison
-   `eda/datasets/wspolne_filmy.png` - Movie overlap between datasets
-   `eda/datasets/wzbogacenie_tmdb.png` - TMDB metadata enrichment
-   `eda/datasets/porownanie_gatunkow.png` - Genre distribution comparison
-   `eda/datasets/pokrycie_czasowe.png` - Temporal coverage
-   `eda/datasets/pipeline_przetwarzania.png` - Data preprocessing pipeline

### Generate All Plots

```bash
# Main visualizations
python -m model.src.visualization.generate_plots

# Dataset EDA
python -m model.src.visualization.dataset_eda
```

---

## 📝 Development

### MLflow Tracking

```bash
mlflow server --port 5000

# View at http://localhost:5000
```

---
