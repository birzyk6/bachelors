"""
Generate Combined Movie Embeddings (MovieLens + TMDB Cold-Start).

This script combines:
1. MovieLens movie embeddings (from trained Two-Tower model)
2. TMDB cold-start embeddings (from content-only tower)

This enables recommendations for ALL movies, not just those with ratings.

SAFE: This script creates NEW files and does not overwrite existing ones.
"""

import gc
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import MODELS_DIR

# Output paths (NEW files - won't overwrite anything)
COMBINED_EMBEDDINGS_PATH = MODELS_DIR / "combined_movie_embeddings.npy"
COMBINED_METADATA_PATH = MODELS_DIR / "combined_movie_embeddings_metadata.json"
TMDB_BERT_MERGED_PATH = MODELS_DIR / "tmdb_bert_embeddings.npz"
TMDB_MOVIE_EMBEDDINGS_PATH = MODELS_DIR / "tmdb_movie_embeddings.npz"


def merge_tmdb_chunks() -> bool:
    """
    Merge TMDB BERT embedding chunks into a single file.

    Returns:
        True if successful, False otherwise
    """
    if TMDB_BERT_MERGED_PATH.exists():
        print(f"✓ TMDB BERT embeddings already merged: {TMDB_BERT_MERGED_PATH}")
        return True

    chunks_dir = MODELS_DIR / "tmdb_chunks"
    if not chunks_dir.exists():
        print(f"✗ TMDB chunks directory not found: {chunks_dir}")
        print("  Run: python precompute_embeddings.py --tmdb")
        return False

    chunk_files = sorted(chunks_dir.glob("chunk_*.npz"))
    if not chunk_files:
        print(f"✗ No chunk files found in {chunks_dir}")
        return False

    print(f"\nMerging {len(chunk_files)} TMDB chunks...")

    all_embeddings = []
    all_movie_ids = []

    for i, chunk_path in enumerate(chunk_files):
        data = np.load(chunk_path)
        all_embeddings.append(data["embeddings"])
        all_movie_ids.append(data["movie_ids"])
        print(f"  Chunk {i+1}/{len(chunk_files)}: {len(data['movie_ids']):,} movies")
        del data
        gc.collect()

    # Concatenate
    embeddings = np.vstack(all_embeddings)
    movie_ids = np.concatenate(all_movie_ids)

    # Free memory
    del all_embeddings, all_movie_ids
    gc.collect()

    print(f"\nTotal: {len(movie_ids):,} TMDB movies with BERT embeddings")
    print(f"Embedding shape: {embeddings.shape}")

    # Save merged file
    np.savez_compressed(
        TMDB_BERT_MERGED_PATH,
        embeddings=embeddings,
        movie_ids=movie_ids,
    )

    file_size_mb = TMDB_BERT_MERGED_PATH.stat().st_size / 1024 / 1024
    print(f"✓ Saved merged BERT embeddings to {TMDB_BERT_MERGED_PATH} ({file_size_mb:.1f} MB)")

    return True


def generate_tmdb_movie_embeddings() -> bool:
    """
    Generate movie embeddings for TMDB movies using content-only tower.

    Returns:
        True if successful, False otherwise
    """
    if TMDB_MOVIE_EMBEDDINGS_PATH.exists():
        print(f"✓ TMDB movie embeddings already exist: {TMDB_MOVIE_EMBEDDINGS_PATH}")
        return True

    # Check prerequisites
    content_tower_path = MODELS_DIR / "content_only_tower"
    if not content_tower_path.exists():
        print(f"✗ Content-only tower not found: {content_tower_path}")
        print("  Train the Two-Tower model first: python train_two_tower.py")
        return False

    if not TMDB_BERT_MERGED_PATH.exists():
        print(f"✗ TMDB BERT embeddings not found: {TMDB_BERT_MERGED_PATH}")
        return False

    # Load content-only tower
    print("\nLoading content-only tower...")
    content_tower = tf.keras.models.load_model(content_tower_path)
    print("✓ Content-only tower loaded")

    # Load TMDB BERT embeddings
    print(f"\nLoading TMDB BERT embeddings from {TMDB_BERT_MERGED_PATH}...")
    tmdb_data = np.load(TMDB_BERT_MERGED_PATH)
    bert_embeddings = tmdb_data["embeddings"]
    movie_ids = tmdb_data["movie_ids"]
    print(f"  Loaded {len(movie_ids):,} movies")

    # Generate movie embeddings in batches
    print("\nGenerating movie embeddings (content-only tower)...")
    batch_size = 1024
    all_embeddings = []
    num_batches = (len(bert_embeddings) + batch_size - 1) // batch_size

    for i in range(0, len(bert_embeddings), batch_size):
        batch = bert_embeddings[i : i + batch_size]
        batch_num = i // batch_size + 1

        # Use content-only tower (takes BERT embedding as input)
        emb = content_tower.predict(batch, verbose=0)
        all_embeddings.append(emb)

        if batch_num % 100 == 0 or batch_num == num_batches:
            print(f"  Batch {batch_num}/{num_batches} ({i + len(batch):,}/{len(bert_embeddings):,})")

        # Garbage collection every 500 batches
        if batch_num % 500 == 0:
            gc.collect()

    # Concatenate
    tmdb_movie_embeddings = np.vstack(all_embeddings)
    print(f"\nFinal embedding shape: {tmdb_movie_embeddings.shape}")

    # Free memory
    del all_embeddings, bert_embeddings
    gc.collect()

    # Save
    np.savez_compressed(
        TMDB_MOVIE_EMBEDDINGS_PATH,
        embeddings=tmdb_movie_embeddings,
        movie_ids=movie_ids,
    )

    file_size_mb = TMDB_MOVIE_EMBEDDINGS_PATH.stat().st_size / 1024 / 1024
    print(f"✓ Saved TMDB movie embeddings to {TMDB_MOVIE_EMBEDDINGS_PATH} ({file_size_mb:.1f} MB)")

    return True


def combine_embeddings() -> bool:
    """
    Combine MovieLens and TMDB movie embeddings.

    MovieLens embeddings take priority for overlapping movie IDs.

    Returns:
        True if successful, False otherwise
    """
    if COMBINED_EMBEDDINGS_PATH.exists():
        print(f"\n⚠️  Combined embeddings already exist: {COMBINED_EMBEDDINGS_PATH}")
        response = input("  Overwrite? (y/N): ").strip().lower()
        if response != "y":
            print("  Skipping...")
            return True

    # Load MovieLens embeddings
    ml_path = MODELS_DIR / "movie_embeddings.npy"
    if not ml_path.exists():
        print(f"✗ MovieLens embeddings not found: {ml_path}")
        print("  Train the Two-Tower model first: python train_two_tower.py")
        return False

    print(f"\nLoading MovieLens embeddings from {ml_path}...")
    ml_embeddings = np.load(ml_path, allow_pickle=True).item()
    print(f"  MovieLens: {len(ml_embeddings):,} movies")

    # Get embedding dimension
    sample_emb = next(iter(ml_embeddings.values()))
    embedding_dim = len(sample_emb)
    print(f"  Embedding dimension: {embedding_dim}")

    # Load TMDB embeddings (if available)
    tmdb_count = 0
    if TMDB_MOVIE_EMBEDDINGS_PATH.exists():
        print(f"\nLoading TMDB embeddings from {TMDB_MOVIE_EMBEDDINGS_PATH}...")
        tmdb_data = np.load(TMDB_MOVIE_EMBEDDINGS_PATH)
        tmdb_embeddings = tmdb_data["embeddings"]
        tmdb_ids = tmdb_data["movie_ids"]
        print(f"  TMDB: {len(tmdb_ids):,} movies")

        # Merge (MovieLens takes priority for overlapping IDs)
        combined = dict(ml_embeddings)
        added = 0
        overlapping = 0

        for i, mid in enumerate(tmdb_ids):
            mid_int = int(mid)
            if mid_int not in combined:
                combined[mid_int] = tmdb_embeddings[i]
                added += 1
            else:
                overlapping += 1

        tmdb_count = added
        print(f"\n  Overlapping movies (kept MovieLens): {overlapping:,}")
        print(f"  New movies from TMDB (cold-start): {added:,}")
    else:
        print(f"\n⚠️  TMDB embeddings not found: {TMDB_MOVIE_EMBEDDINGS_PATH}")
        print("  Combined embeddings will only include MovieLens movies")
        combined = dict(ml_embeddings)

    print(f"\nCombined total: {len(combined):,} movies")

    # Save combined embeddings
    print(f"\nSaving combined embeddings...")
    np.save(COMBINED_EMBEDDINGS_PATH, combined)

    file_size_mb = COMBINED_EMBEDDINGS_PATH.stat().st_size / 1024 / 1024
    print(f"✓ Saved to {COMBINED_EMBEDDINGS_PATH} ({file_size_mb:.1f} MB)")

    # Save metadata
    metadata = {
        "num_movies": len(combined),
        "embedding_dim": embedding_dim,
        "sources": {
            "movielens_two_tower": len(ml_embeddings),
            "tmdb_cold_start": tmdb_count,
        },
        "note": "MovieLens movies use full Two-Tower embeddings (ID + content). "
        "TMDB-only movies use content-only tower (cold-start).",
    }

    with open(COMBINED_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Saved metadata to {COMBINED_METADATA_PATH}")

    return True


def main():
    print("=" * 80)
    print("Generate Combined Movie Embeddings (MovieLens + TMDB)")
    print("=" * 80)
    print(f"\nModels directory: {MODELS_DIR}")

    # Step 1: Merge TMDB chunks (if not already done)
    print("\n" + "-" * 40)
    print("Step 1: Merge TMDB BERT embedding chunks")
    print("-" * 40)
    merge_tmdb_chunks()

    # Step 2: Generate TMDB movie embeddings using content-only tower
    print("\n" + "-" * 40)
    print("Step 2: Generate TMDB movie embeddings")
    print("-" * 40)
    generate_tmdb_movie_embeddings()

    # Step 3: Combine MovieLens + TMDB embeddings
    print("\n" + "-" * 40)
    print("Step 3: Combine all embeddings")
    print("-" * 40)
    combine_embeddings()

    print("\n" + "=" * 80)
    print("✓ Done!")
    print("=" * 80)
    print("\nNew files created (existing files were NOT modified):")
    print(f"  - {TMDB_BERT_MERGED_PATH.name}")
    print(f"  - {TMDB_MOVIE_EMBEDDINGS_PATH.name}")
    print(f"  - {COMBINED_EMBEDDINGS_PATH.name}")
    print(f"  - {COMBINED_METADATA_PATH.name}")
    print("\nTo use combined embeddings in the CLI, update embedding_store to load:")
    print(f"  {COMBINED_EMBEDDINGS_PATH}")


if __name__ == "__main__":
    main()
