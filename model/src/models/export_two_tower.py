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

# Paths
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

    # Check if model is trained
    if not model.is_fitted:
        raise ValueError("Model must be trained before export")

    # ========================================================================
    # 1. Export User Tower
    # ========================================================================
    print("\n1. Exporting User Tower...")
    user_tower_path = output_dir / "user_tower"
    model.save_user_tower(user_tower_path)

    # Test loading
    print("   Testing User Tower loading...")
    import tensorflow as tf

    loaded_user_tower = tf.keras.models.load_model(user_tower_path)
    print(f"   ✓ User Tower successfully loaded (input: {loaded_user_tower.input_shape})")

    # ========================================================================
    # 2. Export Movie Tower
    # ========================================================================
    print("\n2. Exporting Movie Tower...")
    movie_tower_path = output_dir / "movie_tower"
    model.save_movie_tower(movie_tower_path)

    # Test loading
    print("   Testing Movie Tower loading...")
    loaded_movie_tower = tf.keras.models.load_model(movie_tower_path)
    print(f"   ✓ Movie Tower successfully loaded (input: {loaded_movie_tower.input_shape})")

    # ========================================================================
    # 2b. Export Content-Only Tower (for cold-start movies)
    # ========================================================================
    print("\n2b. Exporting Content-Only Tower (for cold-start)...")
    content_tower_path = output_dir / "content_only_tower"
    model.save_content_only_tower(content_tower_path)

    # Test loading
    print("   Testing Content-Only Tower loading...")
    loaded_content_tower = tf.keras.models.load_model(content_tower_path)
    print(f"   ✓ Content-Only Tower successfully loaded (input: {loaded_content_tower.input_shape})")

    # ========================================================================
    # 3. Export Pre-computed Movie Embeddings
    # ========================================================================
    print("\n3. Exporting movie embeddings for Qdrant...")

    if model.movie_embeddings_cache is None:
        print("   Warning: Movie embeddings not pre-computed. Skipping.")
    else:
        embeddings_path = output_dir / "movie_embeddings.npy"
        model.save_movie_embeddings(embeddings_path)

        # Also save as JSON for inspection
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

    # ========================================================================
    # 4. Create deployment instructions
    # ========================================================================
    print("\n4. Creating deployment guide...")

    deployment_guide = f"""
# Two-Tower Model Deployment Guide

## Model Components

This directory contains the exported Two-Tower model components:

1. **user_tower/** - User embedding model (TensorFlow SavedModel)
2. **movie_tower/** - Movie embedding model (TensorFlow SavedModel)
3. **movie_embeddings.npy** - Pre-computed movie embeddings (NumPy array)
4. **movie_embeddings_metadata.json** - Metadata for movie embeddings

## Deployment Options

### Option 1: TensorFlow Serving (REST API)

```bash
# Start TensorFlow Serving for User Tower
docker run -p 8501:8501 \\
  --mount type=bind,source={user_tower_path.absolute()},target=/models/user_tower \\
  -e MODEL_NAME=user_tower \\
  tensorflow/serving

# Start TensorFlow Serving for Movie Tower
docker run -p 8502:8501 \\
  --mount type=bind,source={movie_tower_path.absolute()},target=/models/movie_tower \\
  -e MODEL_NAME=movie_tower \\
  tensorflow/serving
```

### Option 2: Django + Qdrant Architecture

**Django Backend:**
- Load User Tower directly: `tf.keras.models.load_model('user_tower')`
- For each user request, generate user embedding in real-time
- Query Qdrant vector database for similar movie embeddings

**Qdrant Vector Database:**
```python
from qdrant_client import QdrantClient
import numpy as np

# Initialize Qdrant
client = QdrantClient(host="localhost", port=6333)

# Load movie embeddings
embeddings = np.load("movie_embeddings.npy", allow_pickle=True).item()

# Index movies in Qdrant
from qdrant_client.models import Distance, VectorParams

client.recreate_collection(
    collection_name="movies",
    vectors_config=VectorParams(size={model.embedding_dim}, distance=Distance.DOT),
)

# Upload embeddings
points = [
    {{"id": movie_id, "vector": embedding.tolist()}}
    for movie_id, embedding in embeddings.items()
]
client.upsert(collection_name="movies", points=points)
```

**Query for Recommendations:**
```python
# Get user embedding from User Tower
user_embedding = user_tower.predict([[user_id]])

# Query Qdrant for top-K similar movies
results = client.search(
    collection_name="movies",
    query_vector=user_embedding[0].tolist(),
    limit=10,
)

recommended_movie_ids = [hit.id for hit in results]
```

## Model Information

- **Embedding Dimension:** {model.embedding_dim}
- **Number of Movies Indexed:** {len(model.movie_embeddings_cache) if model.movie_embeddings_cache else 'N/A'}
- **Text Features:** {'Enabled (BERT)' if model.use_text_features else 'Disabled'}

## Testing

```python
import tensorflow as tf
import numpy as np

# Load models
user_tower = tf.keras.models.load_model('user_tower')
movie_tower = tf.keras.models.load_model('movie_tower')

# Test inference
user_id = 42
movie_id = 123

user_emb = user_tower.predict([[user_id]])
movie_emb = movie_tower.predict({{'movie_id': [[movie_id]]}})

# Compute affinity score
score = np.dot(user_emb[0], movie_emb[0])
print(f"Affinity score: {{score}}")
```

Generated by: model/src/models/export_two_tower.py
"""

    deployment_guide_path = output_dir / "DEPLOYMENT.md"
    with open(deployment_guide_path, "w") as f:
        f.write(deployment_guide)

    print(f"   ✓ Deployment guide saved to {deployment_guide_path}")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("✓ Export Complete!")
    print("=" * 80)
    print(f"\nAll components saved to: {output_dir.absolute()}")
    print("\nNext steps:")
    print("  1. Review DEPLOYMENT.md for integration instructions")
    print("  2. Test model loading with TensorFlow Serving")
    print("  3. Index movie embeddings in Qdrant vector database")
    print("  4. Integrate User Tower into Django backend")


def main():
    """
    Main entry point for model export.

    This is a placeholder - in practice, you would:
    1. Load the trained Two-Tower model from disk
    2. Call export_two_tower_model()
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
