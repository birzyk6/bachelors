"""Reusable plotting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.config import PLOT_COLORS, PLOTS_DIR


def _prepare_path(path: Path | None, filename: str) -> Path:
    target = path or PLOTS_DIR
    target.mkdir(parents=True, exist_ok=True)
    return target / filename


def plot_training_curves(
    history: Dict[str, List[float]],
    title: str,
    filename: str,
    path: Path | None = None,
) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    for key, values in history.items():
        if len(values) == 1:
            # For single-point grid searches, show as bar instead of line
            ax.bar([key], values, label=key, alpha=0.7)
        else:
            ax.plot(values, marker="o", label=key)

    if all(len(vals) == 1 for vals in history.values()):
        ax.set_ylabel("score")
        ax.set_title(title)
    else:
        ax.set_xlabel("parameter combination")
        ax.set_ylabel("score")
        ax.set_title(title)

    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_path = _prepare_path(path, filename)
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_loss_accuracy(loss: List[float], accuracy: List[float], filename: str, path: Path | None = None) -> Path:
    fig, ax1 = plt.subplots(figsize=(7, 4))
    epochs = range(1, len(loss) + 1)
    ax1.plot(epochs, loss, color=PLOT_COLORS["two_tower"], label="loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss", color=PLOT_COLORS["two_tower"])
    ax2 = ax1.twinx()
    ax2.plot(epochs, accuracy, color="#9467bd", label="accuracy")
    ax2.set_ylabel("accuracy", color="#9467bd")
    fig.suptitle("Two-tower loss vs accuracy")
    fig.tight_layout()
    save_path = _prepare_path(path, filename)
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_rating_predictions(
    titles: List[str],
    actual: List[float],
    predictions: Dict[str, List[float]],
    dataset_avg: List[float] | None,
    filename: str,
    path: Path | None = None,
) -> Path:
    x = np.arange(len(titles))
    fig, ax = plt.subplots(figsize=(10, 6))
    jitter_scale = 0.03
    ax.scatter(x, actual, label="actual", color="#7f7f7f", marker="o", s=70, zorder=3, alpha=0.8)
    offsets = np.linspace(-jitter_scale, jitter_scale, num=max(len(predictions), 1)) if predictions else []
    for offset, (label, preds) in zip(offsets, predictions.items()):
        color = PLOT_COLORS.get(label, None)
        ax.scatter(x + offset, preds, label=label, marker="o", s=70, zorder=3, color=color, alpha=0.75)
    if dataset_avg is not None:
        ax.scatter(
            x,
            dataset_avg,
            label="dataset avg",
            marker="x",
            s=100,
            color="#c70039",
            alpha=0.9,
            zorder=4,
            linewidths=2,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(titles, rotation=45, ha="right")
    ax.set_ylabel("rating")
    ax.set_title("Model predictions vs actual ratings")
    ax.set_ylim(0, 5.5)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    save_path = _prepare_path(path, filename)
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_metric_bars(metrics_df: pd.DataFrame, filename: str, path: Path | None = None) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    error_mask = metrics_df["metric"].isin(["rmse", "mae"])
    metrics_df[error_mask].pivot(index="model", columns="metric", values="value").plot(kind="bar", ax=axes[0])
    axes[0].set_ylabel("value")
    axes[0].set_title("Error metrics")
    ranking_mask = metrics_df["metric"].isin(["precision", "recall", "ndcg", "coverage"])
    metrics_df[ranking_mask].pivot(index="model", columns="metric", values="value").plot(kind="bar", ax=axes[1])
    axes[1].set_ylabel("value")
    axes[1].set_title("Ranking metrics")
    fig.tight_layout()
    save_path = _prepare_path(path, filename)
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_recommendation_overlap(
    labels: List[str],
    recommendations: List[List[int]],
    filename: str,
    path: Path | None = None,
) -> Path:
    unique_counts = [len(set(rec)) for rec in recommendations]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, unique_counts, color=[PLOT_COLORS.get(label.lower(), "#1f77b4") for label in labels])
    ax.set_ylabel("unique items in top-10")
    ax.set_title("Recommendation diversity")
    for idx, value in enumerate(unique_counts):
        ax.text(idx, value + 0.2, str(value), ha="center")
    fig.tight_layout()
    save_path = _prepare_path(path, filename)
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_rating_distribution(ratings_df: pd.DataFrame, filename: str, path: Path | None = None) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    ratings_df["rating"].hist(ax=ax, bins=20, color="#8c564b")
    ax.set_title("Rating distribution")
    ax.set_xlabel("rating")
    ax.set_ylabel("count")
    fig.tight_layout()
    save_path = _prepare_path(path, filename)
    fig.savefig(save_path)
    plt.close(fig)
    return save_path
