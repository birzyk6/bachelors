#!/usr/bin/env python3
"""
Pre-compute BERT embeddings for movie overviews.

This script computes BERT embeddings for all movie overviews and saves them
to disk, so they can be loaded during training without recomputing.

Usage:
    python precompute_embeddings.py
"""

import gc
import json
from pathlib import Path

import numpy as np
import polars as pl
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text  # Required for BERT preprocessing ops

from config import MODELS_DIR, PROCESSED_DIR, print_config


def precompute_bert_embeddings(
    batch_size: int = 64,
    output_file: str = "bert_embeddings.npz",
) -> None:
    """
    Pre-compute BERT embeddings for all movie overviews.

    Args:
        batch_size: Number of texts to encode at once (lower = less memory)
        output_file: Output filename
    """
    print("=" * 80)
    print("Pre-computing BERT Embeddings for Movie Overviews")
    print("=" * 80)

    print_config()

    # Load movies
    movies_file = PROCESSED_DIR / "movies.parquet"
    if not movies_file.exists():
        print(f"✗ Movies file not found: {movies_file}")
        print("  Run preprocessing first: python -m model.src.data.preprocessing")
        return

    print(f"\nLoading movies from {movies_file}...")
    movies_df = pl.read_parquet(movies_file).to_pandas()
    print(f"  Loaded {len(movies_df):,} movies")

    # Get movie IDs and overviews
    movie_ids = movies_df["movieId"].tolist()

    # Get overview column
    if "overview" in movies_df.columns:
        overviews = movies_df["overview"].fillna("").tolist()
    else:
        print("  ⚠ No overview column found, using empty strings")
        overviews = [""] * len(movie_ids)

    # Get title for fallback
    title_col = "title_ml" if "title_ml" in movies_df.columns else "title"
    titles = movies_df[title_col].fillna("").tolist()

    # Combine title + overview for better embeddings
    texts = []
    for title, overview in zip(titles, overviews):
        if overview and len(overview) > 10:
            text = f"{title}. {overview}"
        else:
            text = title
        # Truncate to BERT max length
        texts.append(text[:512])

    print(f"  Prepared {len(texts):,} texts for encoding")

    # Load BERT
    print("\nLoading BERT model from TensorFlow Hub...")
    bert_preprocess_url = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
    bert_encoder_url = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/2"

    bert_preprocess = hub.KerasLayer(bert_preprocess_url)
    bert_encoder = hub.KerasLayer(bert_encoder_url)
    print("  ✓ BERT loaded")

    # Compute embeddings in batches
    print(f"\nComputing embeddings (batch_size={batch_size})...")
    embeddings = []
    num_batches = (len(texts) + batch_size - 1) // batch_size

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_num = i // batch_size + 1

        # Preprocess and encode
        preprocessed = bert_preprocess(batch_texts)
        batch_embeddings = bert_encoder(preprocessed)["pooled_output"].numpy()
        embeddings.append(batch_embeddings)

        # Progress
        if batch_num % 100 == 0 or batch_num == num_batches:
            print(f"  Batch {batch_num}/{num_batches} ({i + len(batch_texts):,}/{len(texts):,} movies)")

        # Force garbage collection every 500 batches to prevent memory buildup
        if batch_num % 500 == 0:
            gc.collect()

    # Concatenate all embeddings
    print("\nConcatenating embeddings...")
    all_embeddings = np.vstack(embeddings)
    print(f"  Shape: {all_embeddings.shape}")

    # Create output directory
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MODELS_DIR / output_file

    # Save as compressed npz with movie ID mapping
    print(f"\nSaving to {output_path}...")
    np.savez_compressed(
        output_path,
        embeddings=all_embeddings,
        movie_ids=np.array(movie_ids),
    )

    # Also save a JSON metadata file
    metadata = {
        "num_movies": len(movie_ids),
        "embedding_dim": all_embeddings.shape[1],
        "bert_model": "small_bert/bert_en_uncased_L-4_H-512_A-8/2",
    }
    metadata_path = MODELS_DIR / "bert_embeddings_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  ✓ Saved {output_path.name} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"  ✓ Saved {metadata_path.name}")

    print("\n" + "=" * 80)
    print("✓ Pre-computation complete!")
    print("=" * 80)
    print(f"\nEmbeddings saved to: {output_path}")
    print(f"Shape: {all_embeddings.shape} ({all_embeddings.shape[0]:,} movies × {all_embeddings.shape[1]} dims)")


if __name__ == "__main__":
    precompute_bert_embeddings(batch_size=32)  # Small batch size to prevent OOM
