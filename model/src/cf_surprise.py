"""Collaborative filtering utilities powered by scikit-surprise."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from src import config
from src.metrics import RankingMetrics, aggregate_ranking_metrics, mae, rmse
from src.plotting import plot_training_curves
from surprise import SVD
from surprise import Dataset as SurpriseDataset
from surprise import Reader
from surprise.model_selection import GridSearchCV


@dataclass
class CFResults:
    metrics: Dict[str, float]
    ranking: RankingMetrics
    history_plot: str


class CollaborativeFilteringModel:
    def __init__(self, param_grid: Dict | None = None, logger: logging.Logger | None = None) -> None:
        self.param_grid = param_grid or config.CF_PARAM_GRID
        self.model: SVD | None = None
        self.history_plot: str | None = None
        self.cv_results: pd.DataFrame | None = None
        self.logger = logger or logging.getLogger(__name__)

    def fit(self, train_df: pd.DataFrame) -> None:
        reader = Reader(rating_scale=(train_df["rating"].min(), train_df["rating"].max()))
        self.logger.info(
            "CF training start | rows=%s | rating_range=(%.2f, %.2f)",
            len(train_df),
            train_df["rating"].min(),
            train_df["rating"].max(),
        )
        grid_df = train_df
        if config.CF_MAX_GRID_ROWS and len(train_df) > config.CF_MAX_GRID_ROWS:
            grid_df = train_df.sample(config.CF_MAX_GRID_ROWS, random_state=config.SEED)
            self.logger.info(
                "CF grid sampling %s rows out of %s for search",
                len(grid_df),
                len(train_df),
            )
        dataset = SurpriseDataset.load_from_df(grid_df[["userId", "tmdbId", "rating"]], reader)

        full_grid = list(ParameterGrid(self.param_grid))
        total_combos = len(full_grid)
        self.logger.info("CF grid has %s combinations before truncation", total_combos)
        param_grid = self.param_grid
        combos_used = total_combos
        if config.CF_MAX_GRID_COMBOS and total_combos > config.CF_MAX_GRID_COMBOS:
            rng = np.random.default_rng(config.SEED)
            selected_idx = sorted(rng.choice(total_combos, size=config.CF_MAX_GRID_COMBOS, replace=False).tolist())
            truncated = [full_grid[i] for i in selected_idx]
            param_grid = [{k: [v] for k, v in combo.items()} for combo in truncated]
            combos_used = len(truncated)
            self.logger.info(
                "CF grid truncated to %s combos (from %s)",
                combos_used,
                total_combos,
            )

        self.logger.info(
            "CF grid search starting | combos=%s | cv=%s | n_jobs=%s",
            combos_used,
            3,
            config.CF_GRID_N_JOBS,
        )
        search = GridSearchCV(
            SVD,
            param_grid,
            measures=["rmse", "mae"],
            cv=3,
            n_jobs=config.CF_GRID_N_JOBS,
        )
        search.fit(dataset)
        best_params = search.best_params["rmse"]
        self.logger.info(
            "CF grid best params=%s | rmse=%.4f | n_epochs=%s",
            best_params,
            search.best_score["rmse"],
            best_params.get("n_epochs", "N/A"),
        )
        self.model = SVD(**best_params)
        self.logger.info(
            "CF final training on full dataset (%s rows) starting | n_epochs=%s",
            len(train_df),
            best_params.get("n_epochs", "N/A"),
        )
        full_dataset = SurpriseDataset.load_from_df(train_df[["userId", "tmdbId", "rating"]], reader)
        self.model.fit(full_dataset.build_full_trainset())
        self.logger.info("CF final training complete")
        self.cv_results = pd.DataFrame(search.cv_results)
        history = {
            "rmse": self.cv_results["mean_test_rmse"].tolist(),
            "mae": self.cv_results["mean_test_mae"].tolist(),
        }
        self.history_plot = str(plot_training_curves(history, "SVD grid search", "cf_grid_search.png"))

    def predict(self, user_id: int, movie_id: int) -> float:
        if self.model is None:
            raise RuntimeError("Model not trained")
        return float(self.model.predict(user_id, movie_id, clip=True).est)

    def recommend(
        self,
        user_id: int,
        candidate_ids: Sequence[int],
        history: set[int],
        topk: int = config.TOP_K,
    ) -> List[int]:
        if self.model is None:
            raise RuntimeError("Model not trained")
        predictions = []
        for mid in candidate_ids:
            if mid in history:
                continue
            est = self.model.predict(user_id, mid, clip=False).est
            predictions.append((mid, est))
        predictions.sort(key=lambda x: x[1], reverse=True)
        return [mid for mid, _ in predictions[:topk]]

    def evaluate(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        candidate_ids: Sequence[int],
        topk: int = config.TOP_K,
    ) -> CFResults:
        if self.model is None:
            raise RuntimeError("Model not trained")
        y_true = test_df["rating"].tolist()
        y_pred = [self.model.predict(row.userId, row.tmdbId).est for row in test_df.itertuples()]
        error_metrics = {
            "rmse": rmse(y_true, y_pred),
            "mae": mae(y_true, y_pred),
        }
        self.logger.info("CF evaluation complete | rmse=%.4f mae=%.4f", error_metrics["rmse"], error_metrics["mae"])
        if len(test_df) > 50:
            sampled_users = (
                test_df["userId"]
                .drop_duplicates()
                .sample(
                    min(config.GRID_EVAL_USERS, test_df["userId"].nunique()),
                    random_state=config.SEED,
                )
                .tolist()
            )
        else:
            sampled_users = test_df["userId"].unique().tolist()
        user_truth: Dict[int, set[int]] = {
            uid: set(test_df.loc[test_df["userId"] == uid, "tmdbId"].tolist()) for uid in sampled_users
        }
        user_histories: Dict[int, set[int]] = {
            uid: set(train_df.loc[train_df["userId"] == uid, "tmdbId"].tolist()) for uid in sampled_users
        }
        user_rankings: Dict[int, List[int]] = {}
        for uid in sampled_users:
            ranking = self.recommend(uid, candidate_ids, user_histories.get(uid, set()), topk)
            user_rankings[uid] = ranking
        ranking_metrics = aggregate_ranking_metrics(
            user_truth,
            user_rankings,
            topk,
            catalog_size=len(candidate_ids),
        )
        return CFResults(
            metrics=error_metrics,
            ranking=ranking_metrics,
            history_plot=self.history_plot or "",
        )
