"""
Export Two-Tower model components for deployment.

Saves:
1. User Tower (SavedModel format for TensorFlow Serving)
2. Movie Tower (SavedModel format for TensorFlow Serving)
3. Pre-computed movie embeddings (NumPy format for Qdrant indexing)
"""

from pathlib import Path

import numpy as np

from model.src.models.two_tower import TwoTowerModel

SAVED_MODELS_DIR = Path(__file__).parent.parent.parent / "saved_models"


def export_two_tower_model(model: TwoTowerModel, output_dir: Path = None):
    """
    Export trained Two-Tower model for deployment.

    Args:
        model: Trained TwoTowerModel instance
        output_dir: Directory to save model components (default: model/saved_models/)
    """
    if output_dir is None:
        output_dir = SAVED_MODELS_DIR

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Two-Tower Model Export for Deployment")
    print("=" * 80)

    if not model.is_fitted:
        raise ValueError("Model must be trained before export")

    print("\n1. Exporting User Tower...")
    user_tower_path = output_dir / "user_tower"
    model.save_user_tower(user_tower_path)

    print("   Testing User Tower loading...")
    import tensorflow as tf

    loaded_user_tower = tf.keras.models.load_model(user_tower_path)
    print(f"   ✓ User Tower successfully loaded (input: {loaded_user_tower.input_shape})")

    print("\n2. Exporting Movie Tower...")
    movie_tower_path = output_dir / "movie_tower"
    model.save_movie_tower(movie_tower_path)

    print("   Testing Movie Tower loading...")
    loaded_movie_tower = tf.keras.models.load_model(movie_tower_path)
    print(f"   ✓ Movie Tower successfully loaded (input: {loaded_movie_tower.input_shape})")

    print("\n2b. Exporting Content-Only Tower (for cold-start)...")
    content_tower_path = output_dir / "content_only_tower"
    model.save_content_only_tower(content_tower_path)

    print("   Testing Content-Only Tower loading...")
    loaded_content_tower = tf.keras.models.load_model(content_tower_path)
    print(f"   ✓ Content-Only Tower successfully loaded (input: {loaded_content_tower.input_shape})")

    print("\n3. Exporting movie embeddings for Qdrant...")

    if model.movie_embeddings_cache is None:
        print("   Warning: Movie embeddings not pre-computed. Skipping.")
    else:
        embeddings_path = output_dir / "movie_embeddings.npy"
        model.save_movie_embeddings(embeddings_path)

        embeddings_json_path = output_dir / "movie_embeddings_metadata.json"
        import json

        metadata = {
            "num_movies": len(model.movie_embeddings_cache),
            "embedding_dim": model.embedding_dim,
            "movie_ids": list(model.movie_embeddings_cache.keys()),
        }

        with open(embeddings_json_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"   ✓ Saved embeddings for {len(model.movie_embeddings_cache)} movies")
        print(f"   ✓ Metadata saved to {embeddings_json_path}")

    print("\n" + "=" * 80)
    print("✓ Export Complete!")
    print("=" * 80)
    print(f"\nAll components saved to: {output_dir.absolute()}")


def main():
    """
    Main entry point for model export.

    """
    print("=" * 80)
    print("Two-Tower Model Export")
    print("=" * 80)
    print("\nThis script exports the trained Two-Tower model for deployment.")
    print("\nUsage:")
    print("  1. Train the Two-Tower model")
    print("  2. Load the trained model")
    print("  3. Call export_two_tower_model(model)")
    print("\nExample:")
    print("  from model.src.models.two_tower import TwoTowerModel")
    print("  model = TwoTowerModel.load('path/to/saved/model')")
    print("  export_two_tower_model(model)")


if __name__ == "__main__":
    main()
