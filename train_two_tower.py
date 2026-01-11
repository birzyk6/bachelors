"""
Train Two-Tower Retrieval Model.

This script trains a production-ready two-tower model for large-scale retrieval.
Includes support for cold-start recommendations using the content-only tower.

Set TEST_MODE=true environment variable to use small dataset for faster testing.
"""

import gc
import json
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    METRICS_DIR,
    MODELS_DIR,
    PROCESSED_DIR,
    TEST_MODE,
    TWO_TOWER_BATCH_SIZE,
    TWO_TOWER_CHUNK_SIZE,
    TWO_TOWER_EPOCHS,
    TWO_TOWER_PARAMS,
    print_config,
)
from model.src.models import TwoTowerModel
from model.src.models.export_two_tower import export_two_tower_model

MODELS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def generate_tmdb_movie_embeddings(
    model: TwoTowerModel,
    output_dir: Path,
    batch_size: int = 1024,
) -> None:
    """
    Generate movie embeddings for all TMDB movies using the content-only tower.

    This enables cold-start recommendations for movies not in the training set.

    Args:
        model: Trained TwoTowerModel with content-only tower
        output_dir: Directory to save embeddings
        batch_size: Batch size for embedding generation
    """
    tmdb_bert_path = output_dir / "tmdb_bert_embeddings.npz"

    if not tmdb_bert_path.exists():
        print(f"  ✗ TMDB BERT embeddings not found: {tmdb_bert_path}")
        return

    print(f"\nLoading TMDB BERT embeddings from {tmdb_bert_path}...")
    data = np.load(tmdb_bert_path)
    bert_embeddings = data["embeddings"]
    movie_ids = data["movie_ids"]

    print(f"  Loaded {len(movie_ids):,} TMDB movies")
    print(f"  BERT embedding shape: {bert_embeddings.shape}")

    print(f"\nGenerating movie embeddings (batch_size={batch_size})...")

    all_embeddings = []
    num_batches = (len(bert_embeddings) + batch_size - 1) // batch_size

    for i in range(0, len(bert_embeddings), batch_size):
        batch = bert_embeddings[i : i + batch_size]
        batch_num = i // batch_size + 1

        embeddings = model.get_cold_start_embeddings_batch(batch)
        all_embeddings.append(embeddings)

        if batch_num % 100 == 0 or batch_num == num_batches:
            print(f"  Batch {batch_num}/{num_batches} ({i + len(batch):,}/{len(bert_embeddings):,})")

        if batch_num % 500 == 0:
            gc.collect()

    final_embeddings = np.vstack(all_embeddings)
    print(f"\nFinal embedding shape: {final_embeddings.shape}")

    output_path = output_dir / "tmdb_movie_embeddings.npz"
    np.savez_compressed(
        output_path,
        embeddings=final_embeddings,
        movie_ids=movie_ids,
    )

    metadata = {
        "num_movies": len(movie_ids),
        "embedding_dim": final_embeddings.shape[1],
        "source": "tmdb_content_only_tower",
        "model_params": model.get_params(),
    }
    metadata_path = output_dir / "tmdb_movie_embeddings_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    file_size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"\n✓ Saved TMDB movie embeddings to {output_path} ({file_size_mb:.1f} MB)")
    print(f"✓ Saved metadata to {metadata_path}")
    print(f"\nThese embeddings enable cold-start recommendations for {len(movie_ids):,} TMDB movies!")


def main():
    print("=" * 80)
    print("Training Two-Tower Retrieval Model (Pure TensorFlow)")
    print("=" * 80)

    print_config()

    print("\nLoading data...")
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    val = pd.read_parquet(PROCESSED_DIR / "val.parquet")
    test = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    movies = pd.read_parquet(PROCESSED_DIR / "movies.parquet")

    print(f"  Train: {len(train):,} ratings")
    print(f"  Val:   {len(val):,} ratings")
    print(f"  Test:  {len(test):,} ratings")
    print(f"  Movies: {len(movies):,}")

    if len(train) > 10_000_000:
        print(f"\n✓ Large dataset detected ({len(train):,} ratings)")
        print("  Using memory-efficient generator-based training")

    unique_user_ids = sorted(train["userId"].unique().tolist())
    unique_movie_ids = sorted(train["movieId"].unique().tolist())

    print(f"  Unique users: {len(unique_user_ids):,}")
    print(f"  Unique movies: {len(unique_movie_ids):,}")

    params = TWO_TOWER_PARAMS

    epochs = TWO_TOWER_EPOCHS
    batch_size = TWO_TOWER_BATCH_SIZE

    mlflow.set_experiment("two_tower")

    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("num_users", len(unique_user_ids))
        mlflow.log_param("num_movies", len(unique_movie_ids))

        print("\nInitializing Two-Tower model...")
        print("⚠️  This will load BERT and may take a few minutes...")
        model = TwoTowerModel(
            user_ids=unique_user_ids,
            movie_ids=unique_movie_ids,
            **params,
        )

        print("\nUser Tower:")
        model.user_tower.summary()
        print("\nMovie Tower:")
        model.movie_tower.summary()

        print("\nTraining model...")
        print("⚠️  Training may take 20-30 minutes...")
        print("⚠️  First epoch includes BERT encoding of all movie overviews...")

        history = model.fit(
            train_data=train,
            movie_data=movies,
            validation_data=None,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            chunk_size=TWO_TOWER_CHUNK_SIZE,
        )

        for epoch in range(len(history.history["loss"])):
            mlflow.log_metric("train_loss", history.history["loss"][epoch], step=epoch)
            if "factorized_top_k/top_100_categorical_accuracy" in history.history:
                acc = history.history["factorized_top_k/top_100_categorical_accuracy"][epoch]
                mlflow.log_metric("train_top_100_accuracy", acc, step=epoch)

        print("\nTwo-Tower model training complete!")
        print("Note: This model is optimized for retrieval (ranking), not rating prediction.")

        print("\nGenerating sample recommendations...")
        user_id = 42
        recommendations = model.recommend(user_id, top_k=10)

        print(f"\nTop 10 recommendations for user {user_id}:")
        for i, (movie_id, score) in enumerate(recommendations, 1):
            print(f"  {i}. Movie {movie_id}: score {score:.3f}")

        print("\nExporting model for deployment...")
        export_two_tower_model(model, output_dir=MODELS_DIR)

        tmdb_bert_path = MODELS_DIR / "tmdb_bert_embeddings.npz"
        if tmdb_bert_path.exists():
            print("\n" + "=" * 80)
            print("Generating embeddings for TMDB catalog (cold-start support)")
            print("=" * 80)
            generate_tmdb_movie_embeddings(model, MODELS_DIR)
        else:
            print("\n⚠️  TMDB BERT embeddings not found. Skipping cold-start embeddings.")
            print("   Run 'python precompute_embeddings.py --tmdb' first to enable cold-start.")

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
            "cold_start_enabled": tmdb_bert_path.exists(),
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
