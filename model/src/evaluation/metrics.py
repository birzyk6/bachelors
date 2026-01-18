"""
Evaluation metrics for recommendation systems.

Implements regression metrics (for rating prediction) and ranking metrics (for top-K recommendation quality).
"""

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error.

    Args:
        y_true: True ratings
        y_pred: Predicted ratings

    Returns:
        RMSE score (lower is better)
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error.

    Args:
        y_true: True ratings
        y_pred: Predicted ratings

    Returns:
        MAE score (lower is better)
    """
    return mean_absolute_error(y_true, y_pred)


def evaluate_rating_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute all regression metrics.

    Args:
        y_true: True ratings
        y_pred: Predicted ratings

    Returns:
        Dictionary with RMSE and MAE
    """
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
    }


def precision_at_k(
    relevant_items: List[int],
    recommended_items: List[int],
    k: int,
) -> float:
    """
    Precision@K: Fraction of recommended items that are relevant.

    Args:
        relevant_items: List of relevant item IDs
        recommended_items: List of recommended item IDs (top-K)
        k: Cutoff position

    Returns:
        Precision@K score in [0, 1]
    """
    if k == 0 or len(recommended_items) == 0:
        return 0.0

    recommended_k = recommended_items[:k]
    relevant_set = set(relevant_items)

    num_relevant = sum(1 for item in recommended_k if item in relevant_set)

    return num_relevant / k


def recall_at_k(
    relevant_items: List[int],
    recommended_items: List[int],
    k: int,
) -> float:
    """
    Recall@K: Fraction of relevant items that are recommended.

    Args:
        relevant_items: List of relevant item IDs
        recommended_items: List of recommended item IDs (top-K)
        k: Cutoff position

    Returns:
        Recall@K score in [0, 1]
    """
    if len(relevant_items) == 0 or k == 0:
        return 0.0

    recommended_k = set(recommended_items[:k])
    relevant_set = set(relevant_items)

    num_relevant = len(recommended_k & relevant_set)

    return num_relevant / len(relevant_set)


def ndcg_at_k(
    relevant_items: List[int],
    recommended_items: List[int],
    k: int,
    relevance_scores: Dict[int, float] | None = None,
) -> float:
    """
    Normalized Discounted Cumulative Gain@K.

    Measures ranking quality with position-based discount.

    Args:
        relevant_items: List of relevant item IDs
        recommended_items: List of recommended item IDs (top-K)
        k: Cutoff position
        relevance_scores: Optional dict mapping item_id -> relevance score.
                         If None, binary relevance (1 for relevant, 0 otherwise).

    Returns:
        NDCG@K score in [0, 1]
    """
    if k == 0 or len(recommended_items) == 0:
        return 0.0

    recommended_k = recommended_items[:k]
    relevant_set = set(relevant_items)

    dcg = 0.0
    for i, item in enumerate(recommended_k):
        if item in relevant_set:
            rel = relevance_scores.get(item, 1.0) if relevance_scores else 1.0

            dcg += rel / np.log2(i + 2)

    if relevance_scores:
        ideal_items = sorted(
            relevant_items,
            key=lambda x: relevance_scores.get(x, 0.0),
            reverse=True,
        )[:k]
        ideal_rels = [relevance_scores.get(item, 1.0) for item in ideal_items]
    else:
        ideal_rels = [1.0] * min(len(relevant_items), k)

    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_rels))

    if idcg == 0:
        return 0.0

    return dcg / idcg


def mrr(
    relevant_items: List[int],
    recommended_items: List[int],
) -> float:
    """
    Mean Reciprocal Rank.

    Measures the rank of the first relevant item.

    Args:
        relevant_items: List of relevant item IDs
        recommended_items: List of recommended item IDs

    Returns:
        MRR score (higher is better)
    """
    relevant_set = set(relevant_items)

    for i, item in enumerate(recommended_items):
        if item in relevant_set:
            return 1.0 / (i + 1)

    return 0.0


def hit_rate_at_k(
    relevant_items: List[int],
    recommended_items: List[int],
    k: int,
) -> float:
    """
    Hit Rate@K: Binary indicator of whether any relevant item is in top-K.

    Args:
        relevant_items: List of relevant item IDs
        recommended_items: List of recommended item IDs (top-K)
        k: Cutoff position

    Returns:
        1.0 if hit, 0.0 otherwise
    """
    if k == 0 or len(recommended_items) == 0:
        return 0.0

    recommended_k = set(recommended_items[:k])
    relevant_set = set(relevant_items)

    return 1.0 if len(recommended_k & relevant_set) > 0 else 0.0


def evaluate_ranking(
    user_relevant_items: Dict[int, List[int]],
    user_recommendations: Dict[int, List[int]],
    k_values: List[int] = None,
    relevance_scores: Dict[int, Dict[int, float]] | None = None,
) -> Dict[str, Dict[int, float]]:
    """
    Evaluate ranking metrics for all users.

    Args:
        user_relevant_items: Dict mapping user_id -> list of relevant item IDs
        user_recommendations: Dict mapping user_id -> list of recommended item IDs
        k_values: List of K values to evaluate (e.g., [5, 10, 20])
        relevance_scores: Optional dict mapping user_id -> {item_id -> score}

    Returns:
        Dictionary with metrics at different K values:
        {
            "precision@5": 0.42,
            "recall@10": 0.68,
            "ndcg@10": 0.73,
            "mrr": 0.51,
            ...
        }
    """
    if k_values is None:
        k_values = [5, 10, 20]

    results = {}

    common_users = set(user_relevant_items.keys()) & set(user_recommendations.keys())

    if not common_users:
        print("Warning: No common users between relevant items and recommendations")
        return results

    for k in k_values:
        precision_scores = []
        recall_scores = []
        ndcg_scores = []
        hit_rates = []

        for user_id in common_users:
            relevant = user_relevant_items[user_id]
            recommended = user_recommendations[user_id]

            if len(relevant) == 0:
                continue

            user_rel_scores = relevance_scores.get(user_id) if relevance_scores else None

            precision_scores.append(precision_at_k(relevant, recommended, k))
            recall_scores.append(recall_at_k(relevant, recommended, k))
            ndcg_scores.append(ndcg_at_k(relevant, recommended, k, user_rel_scores))
            hit_rates.append(hit_rate_at_k(relevant, recommended, k))

        results[f"precision@{k}"] = np.mean(precision_scores) if precision_scores else 0.0
        results[f"recall@{k}"] = np.mean(recall_scores) if recall_scores else 0.0
        results[f"ndcg@{k}"] = np.mean(ndcg_scores) if ndcg_scores else 0.0
        results[f"hit_rate@{k}"] = np.mean(hit_rates) if hit_rates else 0.0

    mrr_scores = []
    for user_id in common_users:
        relevant = user_relevant_items[user_id]
        recommended = user_recommendations[user_id]

        if len(relevant) > 0:
            mrr_scores.append(mrr(relevant, recommended))

    results["mrr"] = np.mean(mrr_scores) if mrr_scores else 0.0

    return results


def get_relevant_items_from_test(
    test_data: pd.DataFrame,
    rating_threshold: float = 4.0,
) -> Dict[int, List[int]]:
    """
    Extract relevant items for each user from test data.

    Items are considered relevant if rating >= threshold.

    Args:
        test_data: DataFrame with [userId, movieId, rating]
        rating_threshold: Minimum rating to consider item relevant

    Returns:
        Dictionary mapping user_id -> list of relevant movie IDs
    """
    relevant_items = {}

    for user_id, group in test_data.groupby("userId"):
        relevant = group[group["rating"] >= rating_threshold]["movieId"].tolist()
        if relevant:
            relevant_items[user_id] = relevant

    return relevant_items


def catalog_coverage(
    recommended_items: List[List[int]],
    total_items: int,
) -> float:
    """
    Catalog Coverage: Percentage of items recommended at least once.

    Args:
        recommended_items: List of recommendation lists
        total_items: Total number of items in catalog

    Returns:
        Coverage percentage in [0, 1]
    """
    unique_recommended = set()
    for recs in recommended_items:
        unique_recommended.update(recs)

    return len(unique_recommended) / total_items if total_items > 0 else 0.0


def diversity_at_k(
    recommended_items: List[List[int]],
    item_features: Dict[int, set],
    k: int,
) -> float:
    """
    Diversity@K: Average pairwise dissimilarity in top-K recommendations.

    Args:
        recommended_items: List of recommendation lists
        item_features: Dict mapping item_id -> set of feature values (e.g., genres)
        k: Cutoff position

    Returns:
        Diversity score in [0, 1]
    """
    diversities = []

    for recs in recommended_items:
        recs_k = recs[:k]

        if len(recs_k) < 2:
            continue

        distances = []
        for i in range(len(recs_k)):
            for j in range(i + 1, len(recs_k)):
                item_i = recs_k[i]
                item_j = recs_k[j]

                if item_i not in item_features or item_j not in item_features:
                    continue

                features_i = item_features[item_i]
                features_j = item_features[item_j]

                intersection = len(features_i & features_j)
                union = len(features_i | features_j)

                distance = 1.0 - (intersection / union if union > 0 else 0.0)
                distances.append(distance)

        if distances:
            diversities.append(np.mean(distances))

    return np.mean(diversities) if diversities else 0.0
