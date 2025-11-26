"""Entry point that trains/evaluates all recommenders and generates plots."""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":  # pragma: no cover - script execution shim
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src import config
from src.cf_surprise import CollaborativeFilteringModel
from src.content_based import ContentBasedModel
from src.data_utils import (
    MovieIndex,
    candidate_movie_ids,
    find_movie_id,
    load_cf_ratings,
    load_movie_embeddings,
    load_movie_metadata,
    load_two_tower_interactions,
    pick_representative_user,
    split_train_val,
)
from src.metrics import RankingMetrics
from src.plotting import plot_metric_bars, plot_rating_distribution, plot_rating_predictions, plot_recommendation_overlap
from src.two_tower import TwoTowerTrainer

logger = logging.getLogger(__name__)


def _format_recommendations(movie_ids: List[int], movie_index: MovieIndex) -> List[Dict[str, str]]:
    rows = []
    for rank, mid in enumerate(movie_ids, start=1):
        if mid in movie_index.meta.index:
            row = movie_index.meta.loc[mid]
            title = row.get("title", f"TMDB {mid}")
            release = row.get("release_date", "")
        else:
            title, release = f"TMDB {mid}", ""
        rows.append({"rank": rank, "tmdbId": mid, "title": title, "release_date": release})
    return rows


def _rating_predictions_for_user(
    user_id: int,
    ratings_df: pd.DataFrame,
    movie_means: pd.Series,
    movie_index: MovieIndex,
    cf_model: CollaborativeFilteringModel,
    content_model: ContentBasedModel,
    tt_model: TwoTowerTrainer,
) -> Path:
    user_history = ratings_df.loc[ratings_df["userId"] == user_id]
    sample = user_history.sample(n=min(10, len(user_history)), random_state=config.SEED)
    titles = []
    dataset_avg = []
    actual = []
    cf_preds = []
    cb_preds = []
    tt_preds = []
    for row in sample.itertuples():
        mid = int(row.tmdbId)
        title = movie_index.meta.loc[mid]["title"] if mid in movie_index.meta.index else str(mid)
        titles.append(title)
        dataset_avg.append(float(movie_means.get(mid, np.nan)))
        actual.append(float(row.rating))
        cf_preds.append(cf_model.predict(row.userId, mid))
        cb_preds.append(content_model.predict(row.userId, mid))
        tt_preds.append(tt_model.predict(row.userId, mid))
    return plot_rating_predictions(
        titles,
        actual,
        {
            "cf": cf_preds,
            "content": cb_preds,
            "two_tower": tt_preds,
        },
        dataset_avg,
        filename="rating_comparison.png",
    )


def _jaccard(a: List[int], b: List[int]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0


def _collect_metrics(
    name: str,
    error_metrics: Dict[str, float],
    ranking_metrics: RankingMetrics,
) -> pd.DataFrame:
    rows = []
    for metric, value in error_metrics.items():
        rows.append({"model": name, "metric": metric, "value": value})
    rows.extend(
        [
            {"model": name, "metric": "precision", "value": ranking_metrics.precision},
            {"model": name, "metric": "recall", "value": ranking_metrics.recall},
            {"model": name, "metric": "ndcg", "value": ranking_metrics.ndcg},
            {"model": name, "metric": "coverage", "value": ranking_metrics.coverage},
        ]
    )
    return pd.DataFrame(rows)


def run(full_data: bool) -> None:
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
        format=config.LOG_FORMAT,
    )
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    logger.info("Run start | full_data=%s", full_data)
    ratings_df = load_cf_ratings(sample=not full_data)
    movie_means = ratings_df.groupby("tmdbId")["rating"].mean()
    movies_df = load_movie_metadata()
    candidate_ids = candidate_movie_ids(ratings_df)
    movie_index = load_movie_embeddings(candidate_ids)
    rating_dist_path = plot_rating_distribution(ratings_df, "rating_distribution.png")
    logger.info("Loaded ratings df=%s rows; distribution plot -> %s", len(ratings_df), rating_dist_path)

    cf_train, cf_test = split_train_val(ratings_df)
    cf_model = CollaborativeFilteringModel()
    logger.info("Training collaborative filtering model...")
    cf_model.fit(cf_train)
    cf_results = cf_model.evaluate(cf_train, cf_test, candidate_ids)
    logger.info(
        "CF metrics rmse=%.4f mae=%.4f precision=%.4f",
        cf_results.metrics["rmse"],
        cf_results.metrics["mae"],
        cf_results.ranking.precision,
    )

    content_train, content_val = split_train_val(ratings_df)
    content_model = ContentBasedModel(movie_index)
    logger.info("Training content-based model...")
    content_model.fit(content_train, content_val)
    content_results = content_model.evaluate(content_train, content_val, candidate_ids)
    logger.info(
        "Content metrics rmse=%.4f mae=%.4f precision=%.4f",
        content_results.metrics["rmse"],
        content_results.metrics["mae"],
        content_results.ranking.precision,
    )

    two_tower_df = load_two_tower_interactions(sample=not full_data, max_rows=config.TWO_TOWER_MAX_ROWS)
    tt_train_df, tt_val_df = split_train_val(two_tower_df)
    tt_trainer = TwoTowerTrainer(movie_index)
    logger.info(
        "Training two-tower model on %s rows (val %s) | max_rows=%s",
        len(tt_train_df),
        len(tt_val_df),
        config.TWO_TOWER_MAX_ROWS,
    )
    tt_trainer.fit(tt_train_df, tt_val_df, use_grid=not full_data)
    tt_results = tt_trainer.evaluate(tt_train_df, tt_val_df, candidate_ids)
    logger.info(
        "Two-tower metrics rmse=%.4f mae=%.4f precision=%.4f",
        tt_results.metrics["rmse"],
        tt_results.metrics["mae"],
        tt_results.ranking.precision,
    )

    metrics_frames = [
        _collect_metrics("Collaborative Filtering", cf_results.metrics, cf_results.ranking),
        _collect_metrics("Content-Based", content_results.metrics, content_results.ranking),
        _collect_metrics("Two-Tower", tt_results.metrics, tt_results.ranking),
    ]
    metrics_df = pd.concat(metrics_frames, ignore_index=True)
    metrics_plot_path = plot_metric_bars(metrics_df, "model_metrics.png")

    # Create nested dict structure for metrics_summary.json
    metrics_summary = {
        "collaborative_filtering": {
            **cf_results.metrics,
            "precision": cf_results.ranking.precision,
            "recall": cf_results.ranking.recall,
            "ndcg": cf_results.ranking.ndcg,
            "coverage": cf_results.ranking.coverage,
        },
        "content_based": {
            **content_results.metrics,
            "precision": content_results.ranking.precision,
            "recall": content_results.ranking.recall,
            "ndcg": content_results.ranking.ndcg,
            "coverage": content_results.ranking.coverage,
        },
        "two_tower": {
            **tt_results.metrics,
            "precision": tt_results.ranking.precision,
            "recall": tt_results.ranking.recall,
            "ndcg": tt_results.ranking.ndcg,
            "coverage": tt_results.ranking.coverage,
        },
    }

    metrics_path = config.ARTIFACTS_DIR / "metrics_summary.json"
    metrics_path.write_text(json.dumps(metrics_summary, indent=2), encoding="utf-8")
    logger.info("Metrics written: %s | plot -> %s", metrics_path, metrics_plot_path)

    avengers_id = find_movie_id(config.AVENGERS_TITLE, movies_df)
    tt_users = set(tt_trainer.user_map.keys()) if tt_trainer.user_map else set()
    try:
        target_user = pick_representative_user(ratings_df, avengers_id, allowed_users=tt_users)
    except ValueError:
        logger.warning("No Avengers superfan present in two-tower sample; relaxing constraints.")
        tt_fans = two_tower_df.loc[
            (two_tower_df["tmdbId"] == avengers_id) & (two_tower_df["rating"] >= 4.0),
            "userId",
        ]
        if not tt_fans.empty:
            target_user = int(tt_fans.sample(n=1, random_state=config.SEED).iloc[0])
        else:
            if not tt_users:
                raise RuntimeError("Two-tower training set is empty; cannot pick representative user")
            target_user = int(random.choice(list(tt_users)))
            logger.warning(
                "Falling back to random two-tower user %s because no Avengers ratings were found in the sample",
                target_user,
            )
    history_source = ratings_df if target_user in ratings_df["userId"].values else two_tower_df
    history = set(history_source.loc[history_source["userId"] == target_user, "tmdbId"].tolist())
    cf_recs = cf_model.recommend(target_user, candidate_ids, history)
    cb_recs = content_model.recommend(target_user, history)
    tt_recs = tt_trainer.recommend(target_user, history)

    rec_frames = {
        "cf": _format_recommendations(cf_recs, movie_index),
        "content": _format_recommendations(cb_recs, movie_index),
        "two_tower": _format_recommendations(tt_recs, movie_index),
    }
    rec_path = config.ARTIFACTS_DIR / "avengers_recommendations.json"
    rec_path.write_text(json.dumps(rec_frames, indent=2), encoding="utf-8")

    rec_overlap = plot_recommendation_overlap(
        ["CF", "Content", "Two-Tower"],
        [cf_recs, cb_recs, tt_recs],
        filename="recommendation_overlap.png",
    )
    rating_plot = _rating_predictions_for_user(
        target_user,
        ratings_df,
        movie_means,
        movie_index,
        cf_model,
        content_model,
        tt_trainer,
    )

    similarity = {
        "cf_vs_content": _jaccard(cf_recs, cb_recs),
        "cf_vs_two_tower": _jaccard(cf_recs, tt_recs),
        "content_vs_two_tower": _jaccard(cb_recs, tt_recs),
    }
    similarity_path = config.ARTIFACTS_DIR / "recommendation_similarity.json"
    similarity_path.write_text(json.dumps(similarity, indent=2), encoding="utf-8")

    summary = {
        "config": {
            "full_data": full_data,
            "device": str(config.DEVICE),
            "ratings_rows": len(ratings_df),
            "two_tower_rows": len(two_tower_df),
        },
        "artifacts": {
            "metrics": str(metrics_path),
            "recommendations": str(rec_path),
            "similarity": str(similarity_path),
            "rating_plot": str(rating_plot),
            "overlap_plot": str(rec_overlap),
            "cf_grid": cf_results.history_plot,
            "content_grid": content_results.history_plot,
            "two_tower_loss": tt_results.loss_plot,
        },
    }
    summary_path = config.ARTIFACTS_DIR / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Run complete. Summary saved to %s", summary_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and compare recommendation models")
    parser.add_argument(
        "--full-data",
        action="store_true",
        help="Train on the entire dataset (default uses sampled subsets)",
    )
    args = parser.parse_args()
    run(full_data=args.full_data)
