"""
Visualization utilities for generating thesis-quality plots.

All plots use consistent academic styling with publication-quality DPI.
Polish translations are used for academic thesis presentation.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Configure matplotlib for LaTeX-style plots with Polish character support
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",  # Supports Polish characters (ą, ę, ć, ł, etc.)
        "font.size": 10,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 16,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

# =============================================================================
# Polish Labels Dictionary - Centralized translations
# =============================================================================
POLISH_LABELS = {
    # Axis labels
    "model": "Model",
    "rating": "Ocena",
    "frequency": "Częstotliwość",
    "count": "Liczba",
    "genre": "Gatunek",
    "movie": "Film",
    "user": "Użytkownik",
    "epoch": "Epoka",
    "loss": "Strata",
    "similarity": "Podobieństwo",
    "score": "Wynik",
    # Metric names
    "rmse": "RMSE (Pierwiastek błędu średniokwadratowego)",
    "mae": "MAE (Średni błąd bezwzględny)",
    "ndcg": "NDCG (Znormalizowany skumulowany zysk)",
    "recall": "Recall (Pełność)",
    "precision": "Precision (Precyzja)",
    "mrr": "MRR (Średnia odwrotność pozycji)",
    # Plot titles
    "rating_distribution": "Rozkład ocen w zbiorze danych",
    "genre_distribution": "Rozkład gatunków filmowych",
    "rmse_comparison": "Porównanie błędu RMSE między modelami",
    "ndcg_comparison": "Porównanie jakości rankingu (NDCG@{k})",
    "recall_comparison": "Porównanie Recall@K między modelami",
    "precision_comparison": "Porównanie Precision@K między modelami",
    "learning_curves": "Krzywe uczenia - {model}",
    "embedding_tsne": "Przestrzeń osadzeń filmów (t-SNE)",
    "embedding_pca": "Przestrzeń osadzeń filmów (PCA)",
    "predictions_comparison": "Porównanie predykcji modeli dla użytkownika {user_id}",
    # Legend labels
    "mean": "Średnia",
    "median": "Mediana",
    "training_loss": "Strata treningowa",
    "validation_loss": "Strata walidacyjna",
    "actual_rating": "Rzeczywista ocena",
    "predicted_rating": "Przewidywana ocena",
    # Two-Tower specific
    "embedding_space": "Przestrzeń osadzeń modelu Two-Tower",
    "similarity_matrix": "Macierz podobieństwa filmów",
    "recommendations": "Rekomendacje",
    "affinity_score": "Wynik dopasowania",
    "cosine_similarity": "Podobieństwo kosinusowe",
    # Genre names (keep English but can add Polish if needed)
    "primary_genre": "Główny gatunek",
    "other": "Inne",
    # Misc
    "num_recommendations": "Liczba rekomendacji (K)",
    "top_n": "Top {n}",
    "movies": "filmy",
    "users": "użytkownicy",
    "ratings": "oceny",
}

# Model name translations
MODEL_NAMES_PL = {
    "collaborative": "Filtrowanie kolaboratywne",
    "content_based": "Filtrowanie treściowe",
    "knn": "k-NN",
    "ncf": "NCF",
    "two_tower": "Two-Tower",
}

# Color palette (colorblind-friendly)
COLORS = sns.color_palette("Set2", 8)
MODEL_COLORS = {
    "collaborative": COLORS[0],
    "content_based": COLORS[1],
    "knn": COLORS[2],
    "ncf": COLORS[3],
    "two_tower": COLORS[4],
}


def get_model_name_pl(model_name: str) -> str:
    """Get Polish model name, or format English name nicely."""
    key = model_name.lower().replace(" ", "_").replace("-", "_")
    return MODEL_NAMES_PL.get(key, model_name.replace("_", " ").title())


def set_thesis_style():
    """Apply thesis-style plot configuration."""
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)


def plot_rmse_comparison(results: Dict, output_path: Path):
    """
    Create bar chart comparing RMSE across all models.

    Args:
        results: Dictionary with rating_prediction results
        output_path: Path to save figure
    """
    set_thesis_style()

    # Extract RMSE values
    models = []
    rmse_values = []

    for model_name, metrics in results["rating_prediction"].items():
        if metrics.get("rmse") is not None:
            models.append(get_model_name_pl(model_name))
            rmse_values.append(metrics["rmse"])

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(models))
    bars = ax.bar(x, rmse_values, color=[MODEL_COLORS.get(m.lower().replace(" ", "_"), COLORS[0]) for m in models])

    # Styling with Polish labels
    ax.set_xlabel(POLISH_LABELS["model"], fontweight="bold")
    ax.set_ylabel(POLISH_LABELS["rmse"], fontweight="bold")
    ax.set_title(POLISH_LABELS["rmse_comparison"], fontweight="bold", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_ylim(0, max(rmse_values) * 1.2)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Grid
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"✓ Saved RMSE comparison to {output_path}")


def plot_ndcg_comparison(results: Dict, output_path: Path, k: int = 10):
    """
    Create bar chart comparing NDCG@K across all models.

    Args:
        results: Dictionary with ranking results
        output_path: Path to save figure
        k: K value for NDCG
    """
    set_thesis_style()

    # Extract NDCG@K values
    models = []
    ndcg_values = []

    for model_name, metrics in results["ranking"].items():
        metric_key = f"ndcg@{k}"
        if metric_key in metrics:
            models.append(get_model_name_pl(model_name))
            ndcg_values.append(metrics[metric_key])

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(models))
    bars = ax.bar(x, ndcg_values, color=[MODEL_COLORS.get(m.lower().replace(" ", "_"), COLORS[0]) for m in models])

    # Styling with Polish labels
    ax.set_xlabel(POLISH_LABELS["model"], fontweight="bold")
    ax.set_ylabel(f"NDCG@{k}", fontweight="bold")
    ax.set_title(POLISH_LABELS["ndcg_comparison"].format(k=k), fontweight="bold", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_ylim(0, 1.0)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Grid
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"✓ Saved NDCG@{k} comparison to {output_path}")


def plot_learning_curves(history, model_name: str, output_path: Path):
    """
    Plot training and validation loss curves.

    Args:
        history: Keras training history object
        model_name: Name of the model
        output_path: Path to save figure
    """
    set_thesis_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract history
    epochs = range(1, len(history.history["loss"]) + 1)
    train_loss = history.history["loss"]
    val_loss = history.history.get("val_loss")

    # Plot with Polish labels
    ax.plot(epochs, train_loss, "o-", label=POLISH_LABELS["training_loss"], color=COLORS[0], linewidth=2, markersize=6)

    if val_loss:
        ax.plot(epochs, val_loss, "s-", label=POLISH_LABELS["validation_loss"], color=COLORS[1], linewidth=2, markersize=6)

    # Styling with Polish labels
    ax.set_xlabel(POLISH_LABELS["epoch"], fontweight="bold")
    ax.set_ylabel(POLISH_LABELS["loss"], fontweight="bold")
    ax.set_title(POLISH_LABELS["learning_curves"].format(model=get_model_name_pl(model_name)), fontweight="bold", pad=20)
    ax.legend(loc="best", frameon=True, shadow=True)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"✓ Saved learning curves to {output_path}")


def plot_recall_at_k(results: Dict, output_path: Path, k_values: List[int] = None):
    """
    Plot Recall@K curves for all models.

    Args:
        results: Dictionary with ranking results
        output_path: Path to save figure
        k_values: List of K values to plot
    """
    if k_values is None:
        k_values = [5, 10, 20]

    set_thesis_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each model
    for model_name, metrics in results["ranking"].items():
        recalls = [metrics.get(f"recall@{k}", 0.0) for k in k_values]

        ax.plot(
            k_values,
            recalls,
            "o-",
            label=get_model_name_pl(model_name),
            color=MODEL_COLORS.get(model_name, COLORS[0]),
            linewidth=2,
            markersize=8,
        )

    # Styling with Polish labels
    ax.set_xlabel(POLISH_LABELS["num_recommendations"], fontweight="bold")
    ax.set_ylabel("Recall@K", fontweight="bold")
    ax.set_title(POLISH_LABELS["recall_comparison"], fontweight="bold", pad=20)
    ax.set_xticks(k_values)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="best", frameon=True, shadow=True)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"✓ Saved Recall@K curves to {output_path}")


def plot_embedding_tsne(
    embeddings: np.ndarray,
    labels: List[str],
    output_path: Path,
    title: str = None,
    perplexity: int = 30,
):
    """
    Create t-SNE visualization of embeddings.

    Args:
        embeddings: Array of shape (n_samples, embedding_dim)
        labels: List of labels for coloring (e.g., genres)
        output_path: Path to save figure
        title: Plot title (defaults to Polish)
        perplexity: t-SNE perplexity parameter
    """
    set_thesis_style()

    if title is None:
        title = POLISH_LABELS["embedding_tsne"]

    print(f"Computing t-SNE projection (perplexity={perplexity})...")

    # Compute t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Get unique labels and colors
    unique_labels = list(set(labels))
    color_map = {label: COLORS[i % len(COLORS)] for i, label in enumerate(unique_labels)}

    # Plot each label group
    for label in unique_labels:
        mask = [l == label for l in labels]
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[color_map[label]],
            label=label,
            alpha=0.6,
            s=30,
        )

    # Styling with Polish labels
    ax.set_xlabel("t-SNE wymiar 1", fontweight="bold")
    ax.set_ylabel("t-SNE wymiar 2", fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=20)
    ax.legend(loc="best", frameon=True, shadow=True, ncol=2, title=POLISH_LABELS["primary_genre"])
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"✓ Saved t-SNE visualization to {output_path}")


def plot_embedding_pca(
    embeddings: np.ndarray,
    labels: List[str],
    output_path: Path,
    title: str = None,
):
    """
    Create PCA visualization of embeddings.

    Args:
        embeddings: Array of shape (n_samples, embedding_dim)
        labels: List of labels for coloring
        output_path: Path to save figure
        title: Plot title (defaults to Polish)
    """
    set_thesis_style()

    if title is None:
        title = POLISH_LABELS["embedding_pca"]

    print("Computing PCA projection...")

    # Compute PCA
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(embeddings)

    # Explained variance
    explained_var = pca.explained_variance_ratio_

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Get unique labels and colors
    unique_labels = list(set(labels))
    color_map = {label: COLORS[i % len(COLORS)] for i, label in enumerate(unique_labels)}

    # Plot each label group
    for label in unique_labels:
        mask = [l == label for l in labels]
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[color_map[label]],
            label=label,
            alpha=0.6,
            s=30,
        )

    # Styling with Polish labels
    ax.set_xlabel(f"PC1 ({explained_var[0]:.1%} wariancji)", fontweight="bold")
    ax.set_ylabel(f"PC2 ({explained_var[1]:.1%} wariancji)", fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=20)
    ax.legend(loc="best", frameon=True, shadow=True, ncol=2, title=POLISH_LABELS["primary_genre"])
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"✓ Saved PCA visualization to {output_path}")


def plot_rating_distribution(ratings: pd.Series, output_path: Path):
    """
    Plot histogram of rating distribution.

    Args:
        ratings: Series of ratings
        output_path: Path to save figure
    """
    set_thesis_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Histogram
    ax.hist(ratings, bins=np.arange(0.5, 5.5, 0.5), color=COLORS[0], edgecolor="black", alpha=0.7)

    # Statistics
    mean_rating = ratings.mean()
    median_rating = ratings.median()

    # Add vertical lines for mean and median with Polish labels
    ax.axvline(mean_rating, color="red", linestyle="--", linewidth=2, label=f"{POLISH_LABELS['mean']}: {mean_rating:.2f}")
    ax.axvline(
        median_rating, color="orange", linestyle="--", linewidth=2, label=f"{POLISH_LABELS['median']}: {median_rating:.2f}"
    )

    # Styling with Polish labels
    ax.set_xlabel(POLISH_LABELS["rating"], fontweight="bold")
    ax.set_ylabel(POLISH_LABELS["frequency"], fontweight="bold")
    ax.set_title(POLISH_LABELS["rating_distribution"], fontweight="bold", pad=20)
    ax.legend(loc="best", frameon=True, shadow=True)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"✓ Saved rating distribution to {output_path}")


def plot_genre_distribution(genres: pd.Series, output_path: Path, top_n: int = 15):
    """
    Plot bar chart of genre frequencies.

    Args:
        genres: Series of genre strings (pipe-separated)
        output_path: Path to save figure
        top_n: Number of top genres to display
    """
    set_thesis_style()

    # Parse genres (pipe-separated)
    all_genres = []
    for genre_str in genres.dropna():
        all_genres.extend(genre_str.split("|"))

    # Count frequencies
    genre_counts = pd.Series(all_genres).value_counts().head(top_n)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(genre_counts))
    bars = ax.bar(x, genre_counts.values, color=COLORS[0], edgecolor="black", alpha=0.7)

    # Styling with Polish labels
    ax.set_xlabel(POLISH_LABELS["genre"], fontweight="bold")
    ax.set_ylabel(POLISH_LABELS["frequency"], fontweight="bold")
    ax.set_title(POLISH_LABELS["genre_distribution"], fontweight="bold", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(genre_counts.index, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"✓ Saved genre distribution to {output_path}")


def plot_model_predictions_comparison(
    models: Dict,
    user_id: int,
    movie_ids: List[int],
    movie_titles: Dict[int, str],
    actual_ratings: Dict[int, float],
    output_path: Path,
):
    """
    Compare predictions from different models for a specific user across multiple movies.

    Args:
        models: Dictionary of model names to loaded model objects
        user_id: User ID to generate predictions for
        movie_ids: List of movie IDs to predict
        movie_titles: Dictionary mapping movie IDs to titles
        actual_ratings: Dictionary mapping movie IDs to actual ratings (if available)
        output_path: Path to save the plot
    """
    set_thesis_style()

    # Prepare data for plotting
    predictions_data = []

    for model_name, model in models.items():
        for movie_id in movie_ids:
            try:
                # Create prediction input
                pred_input = pd.DataFrame({"userId": [user_id], "movieId": [movie_id]})

                # Get prediction
                prediction = model.predict(pred_input)[0]

                predictions_data.append(
                    {
                        "model": model_name,
                        "movie_id": movie_id,
                        "movie_title": movie_titles.get(movie_id, f"Movie {movie_id}"),
                        "prediction": prediction,
                    }
                )
            except Exception as e:
                print(f"Warning: Could not get prediction from {model_name} for movie {movie_id}: {e}")

    df = pd.DataFrame(predictions_data)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Get unique movies and models
    movies = df["movie_title"].unique()
    model_names = df["model"].unique()

    # Set up bar positions
    x = np.arange(len(movies))
    width = 0.15

    # Plot bars for each model
    for i, model_name in enumerate(model_names):
        model_data = df[df["model"] == model_name]
        predictions = [
            (
                model_data[model_data["movie_title"] == movie]["prediction"].values[0]
                if len(model_data[model_data["movie_title"] == movie]) > 0
                else 0
            )
            for movie in movies
        ]

        offset = width * (i - len(model_names) / 2)
        color = MODEL_COLORS.get(model_name, COLORS[i % len(COLORS)])
        ax.bar(
            x + offset,
            predictions,
            width,
            label=get_model_name_pl(model_name),
            color=color,
            edgecolor="black",
            alpha=0.8,
        )

    # Add actual ratings if available with Polish label
    if actual_ratings:
        actual_values = [actual_ratings.get(df[df["movie_title"] == movie]["movie_id"].values[0], None) for movie in movies]
        if any(v is not None for v in actual_values):
            # Plot actual ratings as line
            valid_actual = [(i, v) for i, v in enumerate(actual_values) if v is not None]
            if valid_actual:
                indices, values = zip(*valid_actual)
                ax.plot(
                    indices,
                    values,
                    "r--",
                    linewidth=2,
                    marker="o",
                    markersize=8,
                    label=POLISH_LABELS["actual_rating"],
                    zorder=10,
                )

    # Styling with Polish labels
    ax.set_xlabel(POLISH_LABELS["movie"], fontweight="bold")
    ax.set_ylabel(POLISH_LABELS["rating"], fontweight="bold")
    ax.set_title(POLISH_LABELS["predictions_comparison"].format(user_id=user_id), fontweight="bold", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([title[:30] + "..." if len(title) > 30 else title for title in movies], rotation=45, ha="right")
    ax.set_ylim(0, 5.5)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=2.5, color="gray", linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"✓ Saved model predictions comparison to {output_path}")
