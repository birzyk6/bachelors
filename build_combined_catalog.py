"""
Build Combined Movie Catalog (MovieLens + TMDB).

Creates a combined movies.parquet file that includes:
1. All MovieLens movies (with ratings data)
2. Additional TMDB movies (for cold-start recommendations)

SAFE: Creates a NEW file, does not overwrite existing movies.parquet.
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import MODELS_DIR, PROCESSED_DIR, RAW_DIR

TMDB_CSV = RAW_DIR / "tmdb" / "TMDB_movie_dataset_v11.csv"
ML_MOVIES = PROCESSED_DIR / "movies.parquet"
COMBINED_MOVIES = PROCESSED_DIR / "movies_combined.parquet"


def parse_tmdb_genres(genres_str: str) -> str:
    """
    Parse TMDB genres JSON string to pipe-separated format.

    TMDB format: "[{'id': 28, 'name': 'Action'}, {'id': 12, 'name': 'Adventure'}]"
    Output format: "Action|Adventure"
    """
    if pd.isna(genres_str) or not genres_str:
        return ""

    try:
        import ast

        genres_list = ast.literal_eval(genres_str)
        if isinstance(genres_list, list):
            return "|".join([g.get("name", "") for g in genres_list if isinstance(g, dict)])
    except (ValueError, SyntaxError):
        pass

    return ""


def extract_year(release_date: str) -> int | None:
    """Extract year from release date string."""
    if pd.isna(release_date) or not release_date:
        return None

    try:
        return int(str(release_date)[:4])
    except (ValueError, TypeError):
        return None


def main():
    print("=" * 80)
    print("Build Combined Movie Catalog (MovieLens + TMDB)")
    print("=" * 80)

    combined_emb_path = MODELS_DIR / "combined_movie_embeddings.npy"
    if not combined_emb_path.exists():
        print(f"✗ Combined embeddings not found: {combined_emb_path}")
        print("  Run generate_combined_embeddings.py first")
        return

    import numpy as np

    print("\nLoading combined embeddings to get movie ID list...")
    combined_embeddings = np.load(combined_emb_path, allow_pickle=True).item()
    all_movie_ids = set(combined_embeddings.keys())
    print(f"  Total movie IDs with embeddings: {len(all_movie_ids):,}")

    print(f"\nLoading MovieLens movies from {ML_MOVIES}...")
    ml_df = pd.read_parquet(ML_MOVIES)
    print(f"  MovieLens movies: {len(ml_df):,}")

    ml_ids = set(ml_df["movieId"].tolist())
    print(f"  MovieLens movie IDs: {len(ml_ids):,}")

    needed_tmdb_ids = all_movie_ids - ml_ids
    print(f"  TMDB IDs to add: {len(needed_tmdb_ids):,}")

    print(f"\nLoading TMDB metadata from {TMDB_CSV}...")
    print("  (This may take a minute...)")

    tmdb_df = pd.read_csv(
        TMDB_CSV,
        usecols=["id", "title", "genres", "release_date", "overview", "popularity", "vote_average", "vote_count"],
        dtype={"id": int},
    )
    print(f"  TMDB total movies: {len(tmdb_df):,}")

    tmdb_df = tmdb_df[tmdb_df["id"].isin(needed_tmdb_ids)]
    print(f"  TMDB movies to add (with embeddings): {len(tmdb_df):,}")

    print("\nProcessing TMDB metadata...")
    tmdb_processed = pd.DataFrame(
        {
            "movieId": tmdb_df["id"],
            "title": tmdb_df["title"],
            "title_ml": tmdb_df["title"],
            "genres": tmdb_df["genres"].apply(parse_tmdb_genres),
            "year": tmdb_df["release_date"].apply(extract_year),
            "overview": tmdb_df["overview"].fillna(""),
            "popularity": tmdb_df["popularity"].fillna(0.0),
            "vote_average": tmdb_df["vote_average"].fillna(0.0),
            "vote_count": tmdb_df["vote_count"].fillna(0).astype(int),
        }
    )

    ml_df["source"] = "movielens"
    tmdb_processed["source"] = "tmdb"

    common_cols = ["movieId", "title", "genres", "year", "overview", "source", "popularity", "vote_average", "vote_count"]

    if "title_ml" in ml_df.columns and "title" not in ml_df.columns:
        ml_df["title"] = ml_df["title_ml"]

    if "popularity" not in ml_df.columns:
        ml_df["popularity"] = 0.0
    else:
        ml_df["popularity"] = ml_df["popularity"].fillna(0.0)
    if "vote_average" not in ml_df.columns:
        ml_df["vote_average"] = 0.0
    else:
        ml_df["vote_average"] = ml_df["vote_average"].fillna(0.0)
    ml_df["vote_count"] = 0

    for col in common_cols:
        if col not in ml_df.columns:
            ml_df[col] = "" if col in ["title", "genres", "overview", "source"] else 0

    ml_subset = ml_df[common_cols].copy()
    tmdb_subset = tmdb_processed[common_cols].copy()

    print("\nCombining catalogs...")
    combined_df = pd.concat([ml_subset, tmdb_subset], ignore_index=True)
    print(f"  Combined total: {len(combined_df):,} movies")

    combined_df = combined_df.drop_duplicates(subset=["movieId"], keep="first")
    print(f"  After dedup: {len(combined_df):,} movies")

    print(f"\nSaving to {COMBINED_MOVIES}...")
    combined_df.to_parquet(COMBINED_MOVIES, index=False)

    file_size_mb = COMBINED_MOVIES.stat().st_size / 1024 / 1024
    print(f"✓ Saved combined catalog to {COMBINED_MOVIES} ({file_size_mb:.1f} MB)")

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"  MovieLens movies: {len(ml_ids):,}")
    print(f"  TMDB movies added: {len(tmdb_processed):,}")
    print(f"  Combined total: {len(combined_df):,}")

    print("\nTop genres in combined catalog:")
    all_genres = combined_df["genres"].str.split("|").explode()
    genre_counts = all_genres.value_counts().head(10)
    for genre, count in genre_counts.items():
        if genre:
            print(f"    {genre}: {count:,}")


if __name__ == "__main__":
    main()
