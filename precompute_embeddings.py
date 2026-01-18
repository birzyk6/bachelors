"""
Pre-compute BERT embeddings for movie overviews.

This script computes BERT embeddings for:
1. MovieLens movies (matched with TMDB) - for training
2. All TMDB movies (1.3M) - for cold-start recommendations

Usage:
    python precompute_embeddings.py              # MovieLens movies only
    python precompute_embeddings.py --tmdb       # Full TMDB catalog (1.3M movies)
    python precompute_embeddings.py --all        # Both MovieLens and TMDB
"""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text  # Required for BERT preprocessing ops

from config import MODELS_DIR, PROCESSED_DIR, TMDB_FILE, print_config


def precompute_bert_embeddings(
    batch_size: int = 64,
    output_file: str = "bert_embeddings.npz",
    force: bool = False,
) -> None:
    """
    Pre-compute BERT embeddings for all movie overviews.

    Args:
        batch_size: Number of texts to encode at once (lower = less memory)
        output_file: Output filename
        force: If True, recompute even if file exists
    """
    print("=" * 80)
    print("Pre-computing BERT Embeddings for Movie Overviews")
    print("=" * 80)

    output_path = MODELS_DIR / output_file
    if output_path.exists() and not force:
        print(f"\n✓ Embeddings already exist at {output_path}")
        print("  Use --force to recompute")
        return

    print_config()

    movies_file = PROCESSED_DIR / "movies.parquet"
    if not movies_file.exists():
        print(f"✗ Movies file not found: {movies_file}")
        print("  Run preprocessing first: python -m model.src.data.preprocessing")
        return

    print(f"\nLoading movies from {movies_file}...")
    movies_df = pl.read_parquet(movies_file).to_pandas()
    print(f"  Loaded {len(movies_df):,} movies")

    movie_ids = movies_df["movieId"].tolist()

    if "overview" in movies_df.columns:
        overviews = movies_df["overview"].fillna("").tolist()
    else:
        print("  ⚠ No overview column found, using empty strings")
        overviews = [""] * len(movie_ids)

    title_col = "title_ml" if "title_ml" in movies_df.columns else "title"
    titles = movies_df[title_col].fillna("").tolist()

    texts = []
    for title, overview in zip(titles, overviews):
        if overview and len(overview) > 10:
            text = f"{title}. {overview}"
        else:
            text = title
        texts.append(text[:512])

    print(f"  Prepared {len(texts):,} texts for encoding")

    print("\nLoading BERT model from TensorFlow Hub...")
    bert_preprocess_url = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
    bert_encoder_url = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/2"

    bert_preprocess = hub.KerasLayer(bert_preprocess_url)
    bert_encoder = hub.KerasLayer(bert_encoder_url)
    print("  ✓ BERT loaded")

    print(f"\nComputing embeddings (batch_size={batch_size})...")
    embeddings = []
    num_batches = (len(texts) + batch_size - 1) // batch_size

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_num = i // batch_size + 1

        preprocessed = bert_preprocess(batch_texts)
        batch_embeddings = bert_encoder(preprocessed)["pooled_output"].numpy()
        embeddings.append(batch_embeddings)

        if batch_num % 100 == 0 or batch_num == num_batches:
            print(f"  Batch {batch_num}/{num_batches} ({i + len(batch_texts):,}/{len(texts):,} movies)")

        if batch_num % 500 == 0:
            gc.collect()

    print("\nConcatenating embeddings...")
    all_embeddings = np.vstack(embeddings)
    print(f"  Shape: {all_embeddings.shape}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MODELS_DIR / output_file

    print(f"\nSaving to {output_path}...")
    np.savez_compressed(
        output_path,
        embeddings=all_embeddings,
        movie_ids=np.array(movie_ids),
    )

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


def precompute_tmdb_bert_embeddings(
    batch_size: int = 64,
    output_file: str = "tmdb_bert_embeddings.npz",
    max_movies: int | None = None,
    chunk_size: int = 50000,
) -> None:
    """
    Pre-compute BERT embeddings for ALL TMDB movies (1.3M+).

    Args:
        batch_size: Number of texts to encode at once (lower = less memory)
        output_file: Output filename
        max_movies: Limit number of movies (for testing). None = all movies.
        chunk_size: Number of movies per chunk (saved to disk). Default 50K.
    """
    print("=" * 80)
    print("Pre-computing BERT Embeddings for Full TMDB Catalog")
    print("=" * 80)

    print_config()

    if not TMDB_FILE.exists():
        print(f"✗ TMDB file not found: {TMDB_FILE}")
        print("  Run: bash model/data/download_tmdb.sh")
        return

    checkpoint_dir = MODELS_DIR / "tmdb_chunks"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading TMDB data from {TMDB_FILE}...")

    tmdb_df = pd.read_csv(
        TMDB_FILE,
        usecols=["id", "title", "overview"],
        dtype={"id": "int64", "title": "str", "overview": "str"},
        low_memory=True,
    )

    print(f"  Loaded {len(tmdb_df):,} TMDB movies")

    tmdb_df = tmdb_df[tmdb_df["overview"].notna() & (tmdb_df["overview"].str.len() > 10)]
    print(f"  {len(tmdb_df):,} movies have valid overviews")

    if max_movies is not None:
        tmdb_df = tmdb_df.head(max_movies)
        print(f"  Limited to {len(tmdb_df):,} movies (max_movies={max_movies})")

    movie_ids = tmdb_df["id"].values
    titles = tmdb_df["title"].fillna("").values
    overviews = tmdb_df["overview"].fillna("").values

    texts = []
    for title, overview in zip(titles, overviews):
        text = f"{title}. {overview}" if overview else title
        texts.append(text[:512])

    total_movies = len(texts)
    num_chunks = (total_movies + chunk_size - 1) // chunk_size
    print(f"  Prepared {total_movies:,} texts for encoding")
    print(f"  Will process in {num_chunks} chunks of {chunk_size:,} movies each")

    existing_chunks = sorted(checkpoint_dir.glob("chunk_*.npz"))
    start_chunk = len(existing_chunks)

    if start_chunk > 0:
        print(f"\n✓ Found {start_chunk} existing chunks. Resuming from chunk {start_chunk + 1}...")

    print("\nLoading BERT model from TensorFlow Hub...")
    bert_preprocess_url = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
    bert_encoder_url = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/2"

    bert_preprocess = hub.KerasLayer(bert_preprocess_url)
    bert_encoder = hub.KerasLayer(bert_encoder_url)
    print("  ✓ BERT loaded")

    print(f"\nComputing embeddings (batch_size={batch_size}, chunk_size={chunk_size:,})...")

    for chunk_idx in range(start_chunk, num_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, total_movies)
        chunk_texts = texts[chunk_start:chunk_end]
        chunk_ids = movie_ids[chunk_start:chunk_end]

        print(f"\n--- Chunk {chunk_idx + 1}/{num_chunks} (movies {chunk_start:,}-{chunk_end:,}) ---")

        chunk_embeddings = []
        num_batches = (len(chunk_texts) + batch_size - 1) // batch_size

        for i in range(0, len(chunk_texts), batch_size):
            batch_texts = chunk_texts[i : i + batch_size]
            batch_num = i // batch_size + 1

            try:
                preprocessed = bert_preprocess(batch_texts)
                batch_embeddings = bert_encoder(preprocessed)["pooled_output"].numpy()
                chunk_embeddings.append(batch_embeddings)
            except Exception as e:
                print(f"  ⚠️  Error in batch {batch_num}: {e}")
                chunk_embeddings.append(np.zeros((len(batch_texts), 512), dtype=np.float32))

            if batch_num % 100 == 0 or batch_num == num_batches:
                progress = (chunk_start + i + len(batch_texts)) / total_movies * 100
                print(f"  Batch {batch_num}/{num_batches} - Total progress: {progress:.1f}%")

        chunk_emb_array = np.vstack(chunk_embeddings)
        chunk_path = checkpoint_dir / f"chunk_{chunk_idx:04d}.npz"
        np.savez_compressed(chunk_path, embeddings=chunk_emb_array, movie_ids=chunk_ids)
        print(f"  ✓ Saved chunk {chunk_idx + 1} to {chunk_path.name} ({chunk_emb_array.shape[0]:,} movies)")

        del chunk_embeddings, chunk_emb_array
        gc.collect()

    print("\n" + "=" * 80)
    print("Merging all chunks into final file...")

    all_chunks = sorted(checkpoint_dir.glob("chunk_*.npz"))
    all_embeddings = []
    all_movie_ids = []

    for chunk_path in all_chunks:
        data = np.load(chunk_path)
        all_embeddings.append(data["embeddings"])
        all_movie_ids.append(data["movie_ids"])
        print(f"  Loaded {chunk_path.name}")

    final_embeddings = np.vstack(all_embeddings)
    final_movie_ids = np.concatenate(all_movie_ids)

    print(f"  Final shape: {final_embeddings.shape}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MODELS_DIR / output_file

    print(f"\nSaving to {output_path}...")
    np.savez_compressed(
        output_path,
        embeddings=final_embeddings,
        movie_ids=final_movie_ids,
    )

    metadata = {
        "num_movies": len(final_movie_ids),
        "embedding_dim": final_embeddings.shape[1],
        "bert_model": "small_bert/bert_en_uncased_L-4_H-512_A-8/2",
        "source": "tmdb_full_catalog",
    }
    metadata_path = MODELS_DIR / "tmdb_bert_embeddings_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    file_size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"  ✓ Saved {output_path.name} ({file_size_mb:.1f} MB)")
    print(f"  ✓ Saved {metadata_path.name}")

    print("\nCleaning up chunk files...")
    for chunk_path in all_chunks:
        chunk_path.unlink()
    checkpoint_dir.rmdir()
    print("  ✓ Removed temporary chunk files")

    print("\n" + "=" * 80)
    print("✓ TMDB Pre-computation complete!")
    print("=" * 80)
    print(f"\nEmbeddings saved to: {output_path}")
    print(f"Shape: {final_embeddings.shape} ({final_embeddings.shape[0]:,} movies × {final_embeddings.shape[1]} dims)")
    print("\nThese embeddings can be used for cold-start recommendations")
    print("via the content-only tower in the Two-Tower model.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-compute BERT embeddings for movie recommendations")
    parser.add_argument(
        "--tmdb",
        action="store_true",
        help="Compute embeddings for full TMDB catalog (1.3M movies)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Compute embeddings for both MovieLens and TMDB",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for BERT encoding (default: 32)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50000,
        help="Chunk size for TMDB processing - saves to disk per chunk (default: 50000)",
    )
    parser.add_argument(
        "--max-movies",
        type=int,
        default=None,
        help="Limit number of TMDB movies (for testing)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recompute even if embeddings already exist",
    )

    args = parser.parse_args()

    if args.all:
        print("Computing embeddings for MovieLens movies...")
        precompute_bert_embeddings(batch_size=args.batch_size, force=args.force)
        print("\n" + "=" * 80 + "\n")
        print("Computing embeddings for TMDB catalog...")
        precompute_tmdb_bert_embeddings(
            batch_size=args.batch_size,
            max_movies=args.max_movies,
            chunk_size=args.chunk_size,
        )
    elif args.tmdb:
        precompute_tmdb_bert_embeddings(
            batch_size=args.batch_size,
            max_movies=args.max_movies,
            chunk_size=args.chunk_size,
        )
    else:
        precompute_bert_embeddings(batch_size=args.batch_size, force=args.force)
