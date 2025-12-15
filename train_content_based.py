"""
Train Content-Based Filtering model using BERT embeddings.

This script trains a content-based model that uses BERT to encode movie overviews.

Set TEST_MODE=true environment variable to use small dataset for faster testing.
"""

import json
import sys
from pathlib import Path

import mlflow
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import CONTENT_BASED_PARAMS, METRICS_DIR, MODELS_DIR, PROCESSED_DIR, TEST_MODE, print_config

SAVED_MODELS_DIR = MODELS_DIR  # MODELS_DIR is already the saved_models directory

from model.src.evaluation.metrics import evaluate_rating_predictions
from model.src.models import ContentBasedModel

# Create directories
MODELS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 80)
    print("Training Content-Based Filtering Model (Universal Sentence Encoder)")
    print("=" * 80)

    # Show configuration
    print_config()

    # Load data
    print("\nLoading data...")
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    val = pd.read_parquet(PROCESSED_DIR / "val.parquet")
    test = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    movies = pd.read_parquet(PROCESSED_DIR / "movies.parquet")

    print(f"  Train: {len(train):,} ratings")
    print(f"  Val:   {len(val):,} ratings")
    print(f"  Test:  {len(test):,} ratings")
    print(f"  Movies: {len(movies):,}")

    # Model hyperparameters from config
    params = CONTENT_BASED_PARAMS

    # Start MLflow run
    mlflow.set_experiment("content_based")

    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(params)

        # Initialize model
        print("\nInitializing model...")
        model = ContentBasedModel(**params)

        # Train (pre-compute embeddings)
        print("\nTraining model (encoding movie overviews with BERT)...")
        print("⚠️  This may take 30-60 minutes depending on your hardware...")
        model.fit(movies, train_ratings=train, verbose=True)

        # Evaluate on validation set
        print("\nEvaluating on validation set...")
        val_pairs = val[["userId", "movieId"]]
        y_val_true = val["rating"].values
        y_val_pred = model.predict(val_pairs)

        val_metrics = evaluate_rating_predictions(y_val_true, y_val_pred)

        print(f"  Val RMSE: {val_metrics['rmse']:.4f}")
        print(f"  Val MAE:  {val_metrics['mae']:.4f}")

        # Log validation metrics
        mlflow.log_metric("val_rmse", val_metrics["rmse"])
        mlflow.log_metric("val_mae", val_metrics["mae"])

        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_pairs = test[["userId", "movieId"]]
        y_test_true = test["rating"].values
        y_test_pred = model.predict(test_pairs)

        test_metrics = evaluate_rating_predictions(y_test_true, y_test_pred)

        print(f"  Test RMSE: {test_metrics['rmse']:.4f}")
        print(f"  Test MAE:  {test_metrics['mae']:.4f}")

        # Log test metrics
        mlflow.log_metric("test_rmse", test_metrics["rmse"])
        mlflow.log_metric("test_mae", test_metrics["mae"])

        # Save metrics to JSON
        results = {
            "model": "content_based",
            "params": params,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        }

        with open(METRICS_DIR / "content_based_results.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved to {METRICS_DIR / 'content_based_results.json'}")

        # Save model
        model_path = SAVED_MODELS_DIR / "content_based_model"
        model.save(model_path)
        print(f"✓ Model saved to {model_path}")

        # Test recommendations
        print("\nGenerating sample recommendations...")
        user_id = 42
        recommendations = model.recommend(user_id, top_k=10)

        print(f"\nTop 10 recommendations for user {user_id}:")
        for i, (movie_id, score) in enumerate(recommendations, 1):
            print(f"  {i}. Movie {movie_id}: score {score:.3f}")

    print("\n" + "=" * 80)
    print("✓ Training Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
