# Bachelor's Thesis: Recommendation Systems

**Comparative Study of Recommendation Algorithms with Production-Ready Two-Tower Model**
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
