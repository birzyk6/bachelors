"""
Train K-Nearest Neighbors (KNN) model.

This script trains an item-based KNN collaborative filtering model.

Set TEST_MODE=true environment variable to use small dataset for faster testing.
"""

import json
import sys
from pathlib import Path

import mlflow
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import KNN_PARAMS, METRICS_DIR, MODELS_DIR, PROCESSED_DIR, TEST_MODE, print_config

SAVED_MODELS_DIR = MODELS_DIR
from model.src.evaluation.metrics import evaluate_rating_predictions
from model.src.models import KNNModel

MODELS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 80)
    print("Training K-Nearest Neighbors Model")
    print("=" * 80)

    print_config()

    print("\nLoading data...")
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    val = pd.read_parquet(PROCESSED_DIR / "val.parquet")
    test = pd.read_parquet(PROCESSED_DIR / "test.parquet")

    print(f"  Train: {len(train):,} ratings")
    print(f"  Val:   {len(val):,} ratings")
    print(f"  Test:  {len(test):,} ratings")

    params = {**KNN_PARAMS, "min_support": 1}

    mlflow.set_experiment("knn")

    with mlflow.start_run():
        mlflow.log_params(params)

        print("\nInitializing model...")
        model = KNNModel(**params)

        print("\nTraining model...")
        print("⚠️  Computing similarity matrix may take 10-15 minutes...")
        model.fit(train, verbose=True)

        print("\nEvaluating on validation set...")
        val_pairs = val[["userId", "movieId"]]
        y_val_true = val["rating"].values
        y_val_pred = model.predict(val_pairs)

        val_metrics = evaluate_rating_predictions(y_val_true, y_val_pred)

        print(f"  Val RMSE: {val_metrics['rmse']:.4f}")
        print(f"  Val MAE:  {val_metrics['mae']:.4f}")

        mlflow.log_metric("val_rmse", val_metrics["rmse"])
        mlflow.log_metric("val_mae", val_metrics["mae"])

        print("\nEvaluating on test set...")
        test_pairs = test[["userId", "movieId"]]
        y_test_true = test["rating"].values
        y_test_pred = model.predict(test_pairs)

        test_metrics = evaluate_rating_predictions(y_test_true, y_test_pred)

        print(f"  Test RMSE: {test_metrics['rmse']:.4f}")
        print(f"  Test MAE:  {test_metrics['mae']:.4f}")

        mlflow.log_metric("test_rmse", test_metrics["rmse"])
        mlflow.log_metric("test_mae", test_metrics["mae"])

        results = {
            "model": "knn",
            "params": params,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        }

        with open(METRICS_DIR / "knn_results.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved to {METRICS_DIR / 'knn_results.json'}")

        model_path = SAVED_MODELS_DIR / "knn_model"
        model.save(model_path)
        print(f"✓ Model saved to {model_path}")

        print("\nGenerating sample recommendations...")
        user_id = 42
        recommendations = model.recommend(user_id, top_k=10)

        print(f"\nTop 10 recommendations for user {user_id}:")
        for i, (movie_id, score) in enumerate(recommendations, 1):
            print(f"  {i}. Movie {movie_id}: score {score:.3f}")

        print("\nFinding similar movies (nearest neighbors)...")
        sample_movie_id = train["movieId"].iloc[0]
        neighbors = model.get_neighbors(sample_movie_id, k=5)

        print(f"\nTop 5 movies similar to movie {sample_movie_id}:")
        for i, (neighbor_id, similarity) in enumerate(neighbors, 1):
            print(f"  {i}. Movie {neighbor_id}: similarity {similarity:.3f}")

    print("\n" + "=" * 80)
    print("✓ Training Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
