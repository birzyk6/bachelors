"""Content-based recommender built on precomputed movie embeddings."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from src import config
from src.data_utils import MovieIndex, normalize_matrix
from src.metrics import RankingMetrics, aggregate_ranking_metrics, mae, rmse
from src.plotting import plot_training_curves
from tqdm.auto import tqdm


@dataclass
class ContentResults:
    metrics: Dict[str, float]
    ranking: RankingMetrics
    history_plot: str


class ContentBasedModel:
    def __init__(self, movie_index: MovieIndex, param_grid: Dict | None = None, logger: logging.Logger | None = None) -> None:
        self.movie_index = movie_index
        self.movie_matrix = normalize_matrix(movie_index.matrix.copy())
        self.param_grid = param_grid or {
            "profile_power": [1.0, 1.5, 2.0, 2.5],
            "min_rating": [2.5, 3.0, 3.5, 4.0],
        }
        self.user_profiles: Dict[int, np.ndarray] = {}
        self.best_params: Dict[str, float] | None = None
        self.history_plot: str | None = None
        self.global_mean: float = 3.0
        self.logger = logger or logging.getLogger(__name__)

    def _build_profiles(self, train_df: pd.DataFrame, profile_power: float, min_rating: float) -> Dict[int, np.ndarray]:
        profiles: Dict[int, np.ndarray] = {}
        for uid, group in train_df.groupby("userId"):
            filtered = group[group["rating"] >= min_rating]
            if filtered.empty:
                continue
            weights = np.power(filtered["rating"].to_numpy(), profile_power)
            vectors = []
            for tmdb_id in filtered["tmdbId"]:
                idx = self.movie_index.id_to_idx.get(tmdb_id)
                if idx is None:
                    continue
                vectors.append(self.movie_matrix[idx])
            if not vectors:
                continue
            mat = np.stack(vectors, axis=0)
            profile = np.average(mat, axis=0, weights=weights[: mat.shape[0]])
            norm = np.linalg.norm(profile)
            if norm > 0:
                profile = profile / norm
            profiles[int(uid)] = profile
        return profiles

    def _predict(self, user_profiles: Dict[int, np.ndarray], user_id: int, movie_id: int) -> float:
        profile = user_profiles.get(user_id)
        movie_idx = self.movie_index.id_to_idx.get(movie_id)
        if profile is None or movie_idx is None:
            return self.global_mean
        similarity = float(np.dot(profile, self.movie_matrix[movie_idx]))
        return 0.5 + 4.5 * (similarity + 1) / 2

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
        self.global_mean = float(train_df["rating"].mean()) if not train_df.empty else 3.0
        results = []
        best_score = float("inf")
        best_profiles: Dict[int, np.ndarray] = {}
        for params in tqdm(ParameterGrid(self.param_grid), desc="Content grid", leave=False):
            profiles = self._build_profiles(train_df, **params)
            preds = np.array([self._predict(profiles, row.userId, row.tmdbId) for row in val_df.itertuples()])
            targets = val_df["rating"].to_numpy()
            mask = ~np.isnan(preds)
            if not mask.any():
                score = float("inf")
            else:
                score = rmse(targets[mask], preds[mask])
            results.append({**params, "rmse": score})
            if score < best_score:
                best_score = score
                best_profiles = profiles
                self.best_params = params
                self.logger.info("Content grid improved -> %s (rmse=%.4f)", params, score)
        self.user_profiles = best_profiles
        history = {
            "rmse": [entry["rmse"] for entry in results],
        }
        self.history_plot = str(plot_training_curves(history, "Content grid search", "content_grid.png"))

    def predict(self, user_id: int, movie_id: int) -> float:
        return self._predict(self.user_profiles, user_id, movie_id)

    def recommend(
        self,
        user_id: int,
        history: set[int],
        topk: int = config.TOP_K,
    ) -> List[int]:
        profile = self.user_profiles.get(user_id)
        if profile is None:
            return []
        scores = self.movie_matrix @ profile
        mask_idx = [self.movie_index.id_to_idx[mid] for mid in history if mid in self.movie_index.id_to_idx]
        if mask_idx:
            scores[mask_idx] = -np.inf
        top_idx = np.argsort(-scores)[:topk]
        return [self.movie_index.movie_ids[idx] for idx in top_idx]

    def evaluate(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        candidate_ids: Sequence[int],
        topk: int = config.TOP_K,
    ) -> ContentResults:
        preds = np.array([self.predict(row.userId, row.tmdbId) for row in test_df.itertuples()])
        targets = test_df["rating"].to_numpy()
        mask = ~np.isnan(preds)
        error_metrics = {
            "rmse": rmse(targets[mask], preds[mask]) if mask.any() else float("inf"),
            "mae": mae(targets[mask], preds[mask]) if mask.any() else float("inf"),
        }
        self.logger.info(
            "Content eval complete | sampled_users=%d rmse=%.4f mae=%.4f",
            min(config.GRID_EVAL_USERS, test_df["userId"].nunique()),
            error_metrics["rmse"],
            error_metrics["mae"],
        )
        sampled_users = (
            test_df["userId"]
            .drop_duplicates()
            .sample(min(config.GRID_EVAL_USERS, test_df["userId"].nunique()), random_state=config.SEED)
        )
        user_truth = {uid: set(test_df.loc[test_df["userId"] == uid, "tmdbId"].tolist()) for uid in sampled_users}
        user_histories = {uid: set(train_df.loc[train_df["userId"] == uid, "tmdbId"].tolist()) for uid in sampled_users}
        user_rankings = {uid: self.recommend(uid, user_histories.get(uid, set()), topk) for uid in sampled_users}
        ranking_metrics = aggregate_ranking_metrics(user_truth, user_rankings, topk, catalog_size=len(candidate_ids))
        return ContentResults(
            metrics=error_metrics,
            ranking=ranking_metrics,
            history_plot=self.history_plot or "",
        )
