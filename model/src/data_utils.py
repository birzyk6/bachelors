"""Data loading helpers for recommender experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
from sklearn.model_selection import train_test_split
from src import config


@dataclass
class MovieIndex:
    matrix: np.ndarray
    movie_ids: list[int]
    id_to_idx: dict[int, int]
    meta: pd.DataFrame


def _load_parquet(name: str) -> pl.DataFrame:
    path = config.DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing processed file: {path}")
    return pl.read_parquet(path)


def load_cf_ratings(sample: bool = config.USE_SAMPLE) -> pd.DataFrame:
    ratings = _load_parquet("ratings_cf.parquet")
    if sample:
        eligible = ratings.group_by("userId").agg(pl.len().alias("n")).filter(pl.col("n") >= config.CF_MIN_INTERACTIONS)
        eligible = eligible.sample(min(config.CF_SAMPLE_USERS, eligible.height), seed=config.SEED)
        sampled = ratings.join(eligible.select("userId"), on="userId", how="inner")
        if sampled.height > config.CF_MAX_ROWS:
            sampled = sampled.sample(n=config.CF_MAX_ROWS, seed=config.SEED)
        ratings = sampled
    ratings = ratings.filter(pl.col("rating").is_not_null())
    ratings = ratings.sort(["userId", "timestamp"])
    return ratings.to_pandas()


def load_movie_metadata() -> pd.DataFrame:
    return _load_parquet("movies_cb.parquet").to_pandas()


def load_movie_embeddings(candidate_ids: Iterable[int]) -> MovieIndex:
    movies = load_movie_metadata().reset_index(drop=True)
    movies = movies.reset_index().rename(columns={"index": "row_idx"})
    filtered = movies[movies["id"].isin(candidate_ids)]
    if filtered.empty:
        raise ValueError("No overlap between ratings and movie metadata")
    embeddings = np.load(config.DATA_DIR / "bert_embeddings_cb.npy", mmap_mode="r")
    rows = filtered["row_idx"].to_list()
    matrix = embeddings[rows]
    id_to_idx = {mid: idx for idx, mid in enumerate(filtered["id"].tolist())}
    meta = filtered.set_index("id")
    return MovieIndex(matrix=matrix, movie_ids=list(id_to_idx.keys()), id_to_idx=id_to_idx, meta=meta)


def load_two_tower_interactions(
    sample: bool = config.USE_SAMPLE,
    max_rows: int | None = None,
) -> pd.DataFrame:
    interactions = _load_parquet("two_tower_train.parquet").select(["userId", "tmdbId", "rating"]).drop_nulls()
    limit = max_rows or config.TWO_TOWER_MAX_ROWS
    if sample and limit:
        interactions = interactions.sample(min(limit, interactions.height), seed=config.SEED)
    return interactions.to_pandas().sample(frac=1.0, random_state=config.SEED).reset_index(drop=True)


def split_train_val(df: pd.DataFrame, test_size: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(df, test_size=test_size, random_state=config.SEED)


def find_movie_id(title: str, movies_df: Optional[pd.DataFrame] = None) -> int:
    if movies_df is None:
        movies_df = load_movie_metadata()
    match = movies_df.loc[movies_df["title"].str.lower() == title.lower(), "id"]
    if match.empty:
        raise ValueError(f"Movie '{title}' not found in metadata")
    return int(match.iloc[0])


def pick_representative_user(
    ratings_df: pd.DataFrame,
    tmdb_id: int,
    min_rating: float = 4.5,
    allowed_users: Optional[set[int]] = None,
) -> int:
    fans = ratings_df.loc[(ratings_df["tmdbId"] == tmdb_id) & (ratings_df["rating"] >= min_rating), "userId"]
    if allowed_users is not None:
        fans = fans[fans.isin(list(allowed_users))]
    if fans.empty:
        raise ValueError(
            f"No eligible users rated {tmdb_id} above {min_rating} within allowed set"
            if allowed_users is not None
            else f"No users rated {tmdb_id} above {min_rating}"
        )
    return int(fans.sample(n=1, random_state=config.SEED).iloc[0])


def candidate_movie_ids(ratings_df: pd.DataFrame) -> list[int]:
    return sorted(ratings_df["tmdbId"].unique().tolist())


def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-9, None)
    return matrix / norms
