"""Evaluation helpers for recommendation models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


@dataclass
class RankingMetrics:
    precision: float
    recall: float
    ndcg: float
    coverage: float


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def precision_at_k(truth: set[int], ranked_items: Sequence[int], k: int) -> float:
    if k == 0:
        return 0.0
    hits = sum(1 for item in ranked_items[:k] if item in truth)
    return hits / k


def recall_at_k(truth: set[int], ranked_items: Sequence[int], k: int) -> float:
    if not truth:
        return 0.0
    hits = sum(1 for item in ranked_items[:k] if item in truth)
    return hits / len(truth)


def ndcg_at_k(truth: set[int], ranked_items: Sequence[int], k: int) -> float:
    if not truth:
        return 0.0
    dcg = 0.0
    for idx, item in enumerate(ranked_items[:k]):
        if item in truth:
            dcg += 1.0 / np.log2(idx + 2)
    ideal_hits = min(len(truth), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return float(dcg / idcg) if idcg else 0.0


def aggregate_ranking_metrics(
    user_truth: Dict[int, set[int]],
    user_rankings: Dict[int, Sequence[int]],
    k: int,
    catalog_size: int,
) -> RankingMetrics:
    precision_scores = []
    recall_scores = []
    ndcg_scores = []
    recommended_items = set()
    for user_id, truth in user_truth.items():
        if user_id not in user_rankings:
            continue
        ranking = user_rankings[user_id]
        recommended_items.update(ranking[:k])
        precision_scores.append(precision_at_k(truth, ranking, k))
        recall_scores.append(recall_at_k(truth, ranking, k))
        ndcg_scores.append(ndcg_at_k(truth, ranking, k))
    precision = float(np.mean(precision_scores)) if precision_scores else 0.0
    recall = float(np.mean(recall_scores)) if recall_scores else 0.0
    ndcg = float(np.mean(ndcg_scores)) if ndcg_scores else 0.0
    coverage = len(recommended_items) / catalog_size if catalog_size else 0.0
    return RankingMetrics(precision=precision, recall=recall, ndcg=ndcg, coverage=coverage)
