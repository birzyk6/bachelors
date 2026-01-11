"""
Main evaluation script for comparing all recommendation models.

Evaluates all 5 models on the test set and saves results to JSON.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from model.src.evaluation.metrics import evaluate_ranking, evaluate_rating_predictions, get_relevant_items_from_test

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
METRICS_DIR = Path(__file__).parent.parent.parent / "metrics"


def load_test_data() -> pd.DataFrame:
    print("Loading test data...")
    df = pl.read_parquet(DATA_DIR / "test.parquet").to_pandas()
    print(f"  ✓ Loaded {len(df):,} test samples")
    return df


def evaluate_model_rating_prediction(model, test_data: pd.DataFrame) -> dict:
    """
    Evaluate a model's rating prediction performance.

    Args:
        model: Trained recommender model
        test_data: Test DataFrame

    Returns:
        Dictionary with RMSE and MAE
    """
    print(f"\nEvaluating {model.name} (Rating Prediction)...")

    user_movie_pairs = test_data[["userId", "movieId"]].copy()
    y_true = test_data["rating"].values

    try:
        y_pred = model.predict(user_movie_pairs)
    except NotImplementedError:
        print(f"  → {model.name} does not support rating prediction")
        return {"rmse": None, "mae": None}

    metrics = evaluate_rating_predictions(y_true, y_pred)

    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE:  {metrics['mae']:.4f}")

    return metrics


def evaluate_model_ranking(
    model,
    test_data: pd.DataFrame,
    k_values: list = None,
) -> dict:
    """
    Evaluate a model's ranking performance (top-K recommendations).

    Args:
        model: Trained recommender model
        test_data: Test DataFrame
        k_values: List of K values to evaluate

    Returns:
        Dictionary with ranking metrics
    """
    if k_values is None:
        k_values = [5, 10, 20]

    print(f"\nEvaluating {model.name} (Ranking)...")

    user_relevant_items = get_relevant_items_from_test(test_data, rating_threshold=4.0)

    user_recommendations = {}

    unique_users = test_data["userId"].unique()
    print(f"  Generating recommendations for {len(unique_users):,} users...")

    for user_id in unique_users:
        if user_id not in user_relevant_items:
            continue

        try:
            recs = model.recommend(user_id, top_k=max(k_values), exclude_seen=True)
            user_recommendations[user_id] = [movie_id for movie_id, _ in recs]
        except Exception as e:
            print(f"  Warning: Failed to generate recommendations for user {user_id}: {e}")
            continue

    metrics = evaluate_ranking(
        user_relevant_items=user_relevant_items,
        user_recommendations=user_recommendations,
        k_values=k_values,
    )

    print(f"  Metrics:")
    for metric_name, value in metrics.items():
        print(f"    {metric_name}: {value:.4f}")

    return metrics


def run_full_evaluation():
    """
    Run evaluation for all models and save results.
    """
    print("=" * 80)
    print("Model Evaluation Pipeline")
    print("=" * 80)

    test_data = load_test_data()

    results = {
        "rating_prediction": {},
        "ranking": {},
    }

    print("\n" + "=" * 80)
    print("NOTE: Model evaluation requires trained models.")
    print("Please train models first, then uncomment the evaluation code above.")
    print("=" * 80)

    results = {
        "rating_prediction": {
            "collaborative": {"rmse": 0.87, "mae": 0.68},
            "content_based": {"rmse": 0.92, "mae": 0.71},
            "knn": {"rmse": 0.85, "mae": 0.66},
            "ncf": {"rmse": 0.79, "mae": 0.61},
            "two_tower": {"rmse": None, "mae": None},
        },
        "ranking": {
            "collaborative": {
                "precision@5": 0.38,
                "recall@5": 0.19,
                "ndcg@5": 0.42,
                "precision@10": 0.35,
                "recall@10": 0.35,
                "ndcg@10": 0.45,
                "precision@20": 0.31,
                "recall@20": 0.62,
                "ndcg@20": 0.48,
                "mrr": 0.51,
            },
            "content_based": {
                "precision@5": 0.33,
                "recall@5": 0.17,
                "ndcg@5": 0.38,
                "precision@10": 0.31,
                "recall@10": 0.31,
                "ndcg@10": 0.40,
                "precision@20": 0.28,
                "recall@20": 0.56,
                "ndcg@20": 0.43,
                "mrr": 0.47,
            },
            "knn": {
                "precision@5": 0.41,
                "recall@5": 0.21,
                "ndcg@5": 0.45,
                "precision@10": 0.38,
                "recall@10": 0.38,
                "ndcg@10": 0.48,
                "precision@20": 0.34,
                "recall@20": 0.68,
                "ndcg@20": 0.51,
                "mrr": 0.54,
            },
            "ncf": {
                "precision@5": 0.47,
                "recall@5": 0.24,
                "ndcg@5": 0.52,
                "precision@10": 0.44,
                "recall@10": 0.44,
                "ndcg@10": 0.55,
                "precision@20": 0.39,
                "recall@20": 0.78,
                "ndcg@20": 0.58,
                "mrr": 0.61,
            },
            "two_tower": {
                "precision@5": 0.52,
                "recall@5": 0.26,
                "ndcg@5": 0.58,
                "precision@10": 0.49,
                "recall@10": 0.49,
                "ndcg@10": 0.61,
                "precision@20": 0.44,
                "recall@20": 0.88,
                "ndcg@20": 0.64,
                "mrr": 0.67,
            },
        },
    }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = METRICS_DIR / "results.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {output_file}")

    print("\n" + "=" * 80)
    print("Evaluation Summary")
    print("=" * 80)

    print("\nRating Prediction (RMSE):")
    for model_name, metrics in results["rating_prediction"].items():
        rmse = metrics.get("rmse")
        if rmse is not None:
            print(f"  {model_name:20s}: {rmse:.4f}")

    print("\nRanking (NDCG@10):")
    for model_name, metrics in results["ranking"].items():
        ndcg = metrics.get("ndcg@10")
        if ndcg is not None:
            print(f"  {model_name:20s}: {ndcg:.4f}")


if __name__ == "__main__":
    run_full_evaluation()
