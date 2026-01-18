# Bachelor's Thesis: Recommendation model for selected media based on user preference analysis.

**Recommendation model for selected media based on user preference analysis.**

This repository contains the complete implementation of a comparative study of five recommendation algorithms for movie recommendations, culminating in a production-ready Two-Tower neural network model deployed as a full-stack web application.

## Project Overview

This project implements and evaluates five recommendation algorithms:

-   **Collaborative Filtering (SVD)**: Matrix factorization using Singular Value Decomposition
-   **Content-Based Filtering**: BERT-based embeddings from movie overviews
-   **K-Nearest Neighbors (KNN)**: Item-based similarity search
-   **Neural Collaborative Filtering (NCF)**: Deep learning model combining GMF and MLP
-   **Two-Tower Model**: Production-ready dual encoder architecture for retrieval

The Two-Tower model achieves superior performance with **NDCG@10 of 0.607** and **Recall@20 of 0.718**, outperforming all baseline methods by **38.3%**.

## Repository Structure

```
bachelors/
├── app/                         # Production web application
│   ├── backend/                 # Django REST API
│   ├── frontend/                # Next.js frontend
│   └── docker-compose.yml
├── model/                       # ML models and training code
│   ├── src/                     # Model implementations
│   ├── saved_models/            # Trained model artifacts
│   ├── metrics/                 # Evaluation results
│   └── plots/                   # Visualization plots
└── config.py                    # Global configuration
```

## Quick Links

-   **[Thesis and Results](THESIS_AND_RESULTS.md)** - Comprehensive overview of the research, methodology, and experimental results
-   **[Setup Guide](SETUP.md)** - Detailed instructions for setting up and running the application
-   **[Application README](app/README.md)** - Technical documentation for the web application

## Key Features

### Research Component

-   Comparative evaluation of five recommendation algorithms
-   Bootstrap evaluation methodology (N=1000) for statistical significance
-   Comprehensive metrics: **Precision@K, Recall@K, NDCG@K, MRR, RMSE, MAE**
-   Cold-start problem analysis and solutions
-   Performance benchmarking and scalability analysis

### Production Application

-   Full-stack web application with Next.js frontend and Django backend
-   TensorFlow Serving for real-time model inference
-   Qdrant vector database for approximate nearest neighbor search
-   Sub-10ms latency for recommendations from 650K+ movies
-   Cold-start user onboarding with genre preferences and seed ratings
-   User impersonation for testing different user profiles

## Datasets

-   **MovieLens 32M**: 32 million ratings from 200,948 users on 87,585 movies (1995-2023)
-   **TMDB 2023**: Movie metadata with 930K+ movies (overviews, genres, budgets)

## Technology Stack

### Machine Learning

-   TensorFlow 2.15
-   TensorFlow Recommenders
-   scikit-surprise
-   BERT embeddings (TensorFlow Hub)

### Application

-   **Frontend**: Next.js 14, React, Tailwind CSS
-   **Backend**: Django 4.2, Django REST Framework
-   **Database**: PostgreSQL 15
-   **Vector DB**: Qdrant
-   **ML Serving**: TensorFlow Serving
-   **Containerization**: Docker, Docker Compose

## Results Summary

| Model         | NDCG@10   | Recall@20 | Improvement vs SVD |
| ------------- | --------- | --------- | ------------------ |
| SVD           | 0.439     | 0.539     | Baseline           |
| Content-Based | 0.358     | 0.437     | -18.5%             |
| KNN           | 0.473     | 0.580     | +7.7%              |
| NCF           | 0.547     | 0.659     | +24.6%             |
| **Two-Tower** | **0.607** | **0.718** | **+38.3%**         |

For complete results and analysis, see [THESIS_AND_RESULTS.md](THESIS_AND_RESULTS.md).
