"""Visualization utilities for thesis figures."""

from .plots import (
    plot_embedding_pca,
    plot_embedding_tsne,
    plot_genre_distribution,
    plot_learning_curves,
    plot_ndcg_comparison,
    plot_rating_distribution,
    plot_recall_at_k,
    plot_rmse_comparison,
    set_thesis_style,
)
from .predictions_comparison import load_model_predictions, plot_model_predictions_comparison

__all__ = [
    "plot_rmse_comparison",
    "plot_ndcg_comparison",
    "plot_learning_curves",
    "plot_recall_at_k",
    "plot_embedding_tsne",
    "plot_embedding_pca",
    "plot_rating_distribution",
    "plot_genre_distribution",
    "set_thesis_style",
    "plot_model_predictions_comparison",
    "load_model_predictions",
]
