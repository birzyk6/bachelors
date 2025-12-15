"""
Train Two-Tower Retrieval Model.

This script trains a production-ready two-tower model for large-scale retrieval.

Set TEST_MODE=true environment variable to use small dataset for faster testing.
"""

import json
import sys
from pathlib import Path

import mlflow
import pandas as pd
import tensorflow as tf

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    METRICS_DIR,
    MODELS_DIR,
    PROCESSED_DIR,
    TEST_MODE,
    TWO_TOWER_BATCH_SIZE,
    TWO_TOWER_EPOCHS,
    TWO_TOWER_PARAMS,
    print_config,
)
from model.src.models import TwoTowerModel
from model.src.models.export_two_tower import export_two_tower_model

# Create directories
MODELS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 80)
    print("Training Two-Tower Retrieval Model (Pure TensorFlow)")
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

    # Sample if dataset is too large (>10M ratings)
    MAX_TRAIN_SAMPLES = 10_000_000
    if len(train) > MAX_TRAIN_SAMPLES:
        print(f"\n⚠️  Dataset too large, sampling {MAX_TRAIN_SAMPLES:,} ratings for training")
        train = train.sample(n=MAX_TRAIN_SAMPLES, random_state=42)
        print(f"  Sampled train: {len(train):,} ratings")

    # Get unique IDs
    unique_user_ids = sorted(train["userId"].unique().tolist())
    unique_movie_ids = sorted(train["movieId"].unique().tolist())

    print(f"  Unique users: {len(unique_user_ids):,}")
    print(f"  Unique movies: {len(unique_movie_ids):,}")

    # Model hyperparameters from config
    params = TWO_TOWER_PARAMS

    # Training parameters from config
    epochs = TWO_TOWER_EPOCHS
    batch_size = TWO_TOWER_BATCH_SIZE

    # Start MLflow run
    mlflow.set_experiment("two_tower")

    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("num_users", len(unique_user_ids))
        mlflow.log_param("num_movies", len(unique_movie_ids))

        # Initialize model
        print("\nInitializing Two-Tower model...")
        print("⚠️  This will load BERT and may take a few minutes...")
        model = TwoTowerModel(
            user_ids=unique_user_ids,
            movie_ids=unique_movie_ids,
            **params,
        )

        # Print architecture
        print("\nUser Tower:")
        model.user_tower.summary()
        print("\nMovie Tower:")
        model.movie_tower.summary()

        # Train
        print("\nTraining model...")
        print("⚠️  Training may take 20-30 minutes...")
        print("⚠️  First epoch includes BERT encoding of all movie overviews...")

        history = model.fit(
            train_data=train,
            movie_data=movies,
            validation_data=None,  # Two-tower doesn't use validation during training
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
        )

        # Log training metrics
        for epoch in range(len(history.history["loss"])):
            mlflow.log_metric("train_loss", history.history["loss"][epoch], step=epoch)
            if "factorized_top_k/top_100_categorical_accuracy" in history.history:
                acc = history.history["factorized_top_k/top_100_categorical_accuracy"][epoch]
                mlflow.log_metric("train_top_100_accuracy", acc, step=epoch)

        # Note: Two-Tower is optimized for retrieval, not rating prediction
        # Evaluation focuses on ranking metrics (covered in full evaluation script)

        print("\nTwo-Tower model training complete!")
        print("Note: This model is optimized for retrieval (ranking), not rating prediction.")

        # Test recommendations
        print("\nGenerating sample recommendations...")
        user_id = 42
        recommendations = model.recommend(user_id, top_k=10)

        print(f"\nTop 10 recommendations for user {user_id}:")
        for i, (movie_id, score) in enumerate(recommendations, 1):
            print(f"  {i}. Movie {movie_id}: score {score:.3f}")

        # Export model for deployment
        print("\nExporting model for deployment...")
        export_two_tower_model(model, output_dir=MODELS_DIR)

        # Save training info
        results = {
            "model": "two_tower",
            "params": params,
            "training_params": {
                "epochs": epochs,
                "batch_size": batch_size,
                "num_users": len(unique_user_ids),
                "num_movies": len(unique_movie_ids),
            },
            "final_epoch": len(history.history["loss"]),
            "note": "Optimized for retrieval; use ranking metrics for evaluation",
        }

        with open(METRICS_DIR / "two_tower_results.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved to {METRICS_DIR / 'two_tower_results.json'}")

    print("\n" + "=" * 80)
    print("✓ Training Complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Run full evaluation: python -m model.src.evaluation.run_evaluation")
    print("  2. Deploy with TensorFlow Serving (see model/saved_models/DEPLOYMENT.md)")


if __name__ == "__main__":
    main()
