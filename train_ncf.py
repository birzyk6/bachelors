"""
Train Neural Collaborative Filtering (NCF) model.

This script trains a deep learning model combining GMF and MLP.

Set TEST_MODE=true environment variable to use small dataset for faster testing.
"""

import json
import sys
from pathlib import Path

import mlflow
import pandas as pd
import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).parent))

from config import METRICS_DIR, MODELS_DIR, NCF_BATCH_SIZE, NCF_EPOCHS, NCF_PARAMS, PROCESSED_DIR, TEST_MODE, print_config
from model.src.evaluation.metrics import evaluate_rating_predictions
from model.src.models import NeuralCollaborativeFiltering

MODELS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 80)
    print("Training Neural Collaborative Filtering (NCF) Model")
    print("=" * 80)

    print_config()

    print("\nLoading data...")
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    val = pd.read_parquet(PROCESSED_DIR / "val.parquet")
    test = pd.read_parquet(PROCESSED_DIR / "test.parquet")

    print(f"  Train: {len(train):,} ratings")
    print(f"  Val:   {len(val):,} ratings")
    print(f"  Test:  {len(test):,} ratings")

    num_users = train["userId"].nunique()
    num_movies = train["movieId"].nunique()

    print(f"  Users: {num_users:,}")
    print(f"  Movies: {num_movies:,}")

    params = {
        "num_users": num_users + 1,
        "num_movies": num_movies + 1,
        **NCF_PARAMS,
    }

    epochs = NCF_EPOCHS
    batch_size = NCF_BATCH_SIZE

    mlflow.set_experiment("ncf")

    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)

        print("\nInitializing model...")
        model = NeuralCollaborativeFiltering(**params)

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=params["learning_rate"]),
            loss="mse",
            metrics=["mae"],
        )

        print("\nModel architecture:")
        model.model.summary()

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=2,
                min_lr=1e-6,
            ),
        ]

        print("\nTraining model...")
        history = model.fit(
            train_data=train,
            validation_data=val,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=callbacks,
        )

        for epoch in range(len(history.history["loss"])):
            mlflow.log_metric("train_loss", history.history["loss"][epoch], step=epoch)
            mlflow.log_metric("train_mae", history.history["mae"][epoch], step=epoch)
            if "val_loss" in history.history:
                mlflow.log_metric("val_loss", history.history["val_loss"][epoch], step=epoch)
                mlflow.log_metric("val_mae", history.history["val_mae"][epoch], step=epoch)

        print("\nEvaluating on validation set...")
        val_pairs = val[["userId", "movieId"]]
        y_val_true = val["rating"].values
        y_val_pred = model.predict(val_pairs)

        val_metrics = evaluate_rating_predictions(y_val_true, y_val_pred)

        print(f"  Val RMSE: {val_metrics['rmse']:.4f}")
        print(f"  Val MAE:  {val_metrics['mae']:.4f}")

        print("\nEvaluating on test set...")
        test_pairs = test[["userId", "movieId"]]
        y_test_true = test["rating"].values
        y_test_pred = model.predict(test_pairs)

        test_metrics = evaluate_rating_predictions(y_test_true, y_test_pred)

        print(f"  Test RMSE: {test_metrics['rmse']:.4f}")
        print(f"  Test MAE:  {test_metrics['mae']:.4f}")

        mlflow.log_metric("final_val_rmse", val_metrics["rmse"])
        mlflow.log_metric("final_val_mae", val_metrics["mae"])
        mlflow.log_metric("final_test_rmse", test_metrics["rmse"])
        mlflow.log_metric("final_test_mae", test_metrics["mae"])

        model_path = MODELS_DIR / "ncf_model"
        model.save(model_path)
        print(f"\n✓ Model saved to {model_path}")

        results = {
            "model": "ncf",
            "params": params,
            "training_params": {"epochs": epochs, "batch_size": batch_size},
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "final_epoch": len(history.history["loss"]),
        }

        with open(METRICS_DIR / "ncf_results.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"✓ Results saved to {METRICS_DIR / 'ncf_results.json'}")

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
