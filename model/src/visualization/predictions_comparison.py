"""
Generate predictions comparison using only pre-trained saved models.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns

from ..models import CollaborativeFilteringModel, ContentBasedModel, KNNModel, NeuralCollaborativeFiltering

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 9,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 16,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

COLORS = sns.color_palette("Set2", 8)
MODEL_COLORS = {
    "Collaborative": COLORS[0],
    "Content-Based": COLORS[1],
    "KNN": COLORS[2],
    "NCF": COLORS[3],
    "Two-Tower": COLORS[4],
}
MODEL_MARKERS = {
    "Collaborative": "o",
    "Content-Based": "s",
    "KNN": "^",
    "NCF": "D",
    "Two-Tower": "v",
}


def load_model_predictions(
    model_name: str,
    model_class,
    model_path: Path,
    sample_user_id: int,
    movie_ids: List[int],
    init_args: Optional[Dict] = None,
) -> Optional[Dict[int, float]]:
    """Generic function to load a saved model and generate predictions."""
    if not model_path.exists():
        return None

    try:
        if init_args:
            model = model_class(**init_args)
        else:
            model = model_class()

        model.load(model_path)

        pred_input = pd.DataFrame({"userId": [sample_user_id] * len(movie_ids), "movieId": movie_ids})
        preds = model.predict(pred_input)
        return dict(zip(movie_ids, preds))
    except Exception as e:
        print(f"    ✗ Could not load {model_name}: {e}")
        return None


def load_sample_predictions(
    model_name: str, sample_user_id: int, movie_ids: List[int], metrics_dir: Path
) -> Optional[Dict[int, float]]:
    """Load sample predictions from training results if available."""
    results_file = metrics_dir / f"{model_name}_results.json"
    if not results_file.exists():
        return None

    try:
        with open(results_file) as f:
            results = json.load(f)

        if "sample_predictions" in results:
            sample_preds = results["sample_predictions"]
            if str(sample_user_id) in sample_preds:
                user_preds = sample_preds[str(sample_user_id)]
                preds = {}
                for movie_id in movie_ids:
                    for pred in user_preds:
                        if pred.get("movieId") == movie_id or pred.get("movie_id") == movie_id:
                            preds[movie_id] = pred.get("predicted_rating") or pred.get("score", 3.0)
                            break
                return preds if preds else None
        return None
    except Exception as e:
        print(f"    ✗ Could not load {model_name} predictions: {e}")
        return None


def generate_baseline_predictions(
    sample_user_id: int, movie_ids: List[int], test_data: pd.DataFrame, train_data: pd.DataFrame
) -> Dict[int, float]:
    """Generate simple baseline predictions when models aren't available."""
    predictions = {}

    user_ratings = train_data[train_data["userId"] == sample_user_id]["rating"]
    if len(user_ratings) > 0:
        user_avg = user_ratings.mean()
    else:
        user_avg = 3.5

    for movie_id in movie_ids:
        movie_ratings = train_data[train_data["movieId"] == movie_id]["rating"]
        if len(movie_ratings) > 0:
            movie_avg = movie_ratings.mean()
            predictions[movie_id] = (user_avg + movie_avg) / 2
        else:
            predictions[movie_id] = user_avg

    return predictions


def plot_model_predictions_comparison(
    data_dir: Path,
    saved_models_dir: Path,
    metrics_dir: Path,
    output_dir: Path,
    sample_user_id: int = 42,
    num_movies: int = 10,
) -> Path:
    """
    Generate model predictions comparison plot using saved models.

    Args:
        data_dir: Path to processed data directory
        saved_models_dir: Path to saved models directory
        metrics_dir: Path to metrics directory
        output_dir: Path to save plots
        sample_user_id: User ID to generate predictions for
        num_movies: Number of movies to compare

    Returns:
        Path to saved plot
    """
    print("=" * 80)
    print("Model Predictions Comparison (Saved Models)")
    print("=" * 80)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nLoading data...")
    test_data = pl.read_parquet(data_dir / "test.parquet").to_pandas()
    train_data = pl.read_parquet(data_dir / "train.parquet").to_pandas()
    movies_data = pl.read_parquet(data_dir / "movies.parquet").to_pandas()

    movie_titles = dict(zip(movies_data["movieId"], movies_data["title_ml"]))

    if "popularity" in test_data.columns:
        popularity_threshold = test_data["popularity"].quantile(0.5)
        popular_test_data = test_data[test_data["popularity"] >= popularity_threshold].copy()

        user_popular_counts = popular_test_data.groupby("userId").size()
        users_with_enough = user_popular_counts[user_popular_counts >= num_movies]

        if len(users_with_enough) > 0:
            if sample_user_id in users_with_enough.index:
                user_test_ratings = popular_test_data[popular_test_data["userId"] == sample_user_id].copy()
            else:
                sample_user_id = user_popular_counts.idxmax()
                user_test_ratings = popular_test_data[popular_test_data["userId"] == sample_user_id].copy()
        else:
            user_test_ratings = test_data[test_data["userId"] == sample_user_id].copy()
    else:
        user_test_ratings = test_data[test_data["userId"] == sample_user_id].copy()

    if len(user_test_ratings) == 0:
        sample_user_id = test_data["userId"].iloc[0]
        user_test_ratings = test_data[test_data["userId"] == sample_user_id].copy()

    if len(user_test_ratings) >= num_movies:
        sorted_ratings = user_test_ratings.sort_values("rating")

        rating_bins = pd.cut(sorted_ratings["rating"], bins=num_movies, duplicates="drop")

        if "popularity" in sorted_ratings.columns:
            selected = []
            for bin_label in rating_bins.cat.categories:
                bin_movies = sorted_ratings[rating_bins == bin_label]
                if len(bin_movies) > 0:
                    most_popular = bin_movies.nlargest(1, "popularity")
                    selected.append(most_popular)

            if len(selected) >= num_movies:
                user_test_ratings = pd.concat(selected).head(num_movies)
            else:
                indices = np.linspace(0, len(sorted_ratings) - 1, num_movies, dtype=int)
                user_test_ratings = sorted_ratings.iloc[indices]
        else:
            indices = np.linspace(0, len(sorted_ratings) - 1, num_movies, dtype=int)
            user_test_ratings = sorted_ratings.iloc[indices]
    else:
        user_test_ratings = user_test_ratings.head(num_movies)

    print(f"  Selected user: {sample_user_id}")
    print(f"  Movies: {len(user_test_ratings)}")
    print(f"  Rating range: {user_test_ratings['rating'].min():.1f} - {user_test_ratings['rating'].max():.1f}")

    movie_ids = user_test_ratings["movieId"].tolist()
    actual_ratings = dict(zip(user_test_ratings["movieId"], user_test_ratings["rating"]))

    predictions = {}

    print("\nLoading models and generating predictions...")

    print("  Loading Collaborative Filtering...")
    cf_path = saved_models_dir / "collaborative_model"
    cf_preds = load_model_predictions("Collaborative", CollaborativeFilteringModel, cf_path, sample_user_id, movie_ids)
    if cf_preds:
        predictions["Collaborative"] = cf_preds
        print("    ✓ Collaborative predictions loaded")

    print("  Loading Content-Based...")
    cb_path = saved_models_dir / "content_based_model"
    cb_preds = load_model_predictions("Content-Based", ContentBasedModel, cb_path, sample_user_id, movie_ids)
    if cb_preds:
        predictions["Content-Based"] = cb_preds
        print("    ✓ Content-Based predictions loaded")

    print("  Loading KNN...")
    knn_path = saved_models_dir / "knn_model"
    knn_preds = load_model_predictions("KNN", KNNModel, knn_path, sample_user_id, movie_ids)
    if knn_preds:
        predictions["KNN"] = knn_preds
        print("    ✓ KNN predictions loaded")

    print("  Loading NCF...")
    ncf_path = saved_models_dir / "ncf_model"
    if ncf_path.exists():
        try:
            with open(ncf_path / "mappings.json") as f:
                mappings = json.load(f)
            num_users = len(mappings["user_id_map"])
            num_movies_count = len(mappings["movie_id_map"])

            ncf_preds = load_model_predictions(
                "NCF",
                NeuralCollaborativeFiltering,
                ncf_path,
                sample_user_id,
                movie_ids,
                init_args={"num_users": num_users, "num_movies": num_movies_count},
            )
            if ncf_preds:
                predictions["NCF"] = ncf_preds
                print("    ✓ NCF predictions loaded")
        except Exception as e:
            print(f"    ✗ Could not load NCF: {e}")

    if len(predictions) == 0:
        print("\n✗ No predictions available. Please train models first.")
        raise ValueError("No model predictions available")

    print("\nGenerating plot...")
    fig, ax = plt.subplots(figsize=(12, 10))

    movie_labels = [movie_titles.get(mid, f"Movie {mid}") for mid in movie_ids]

    y = np.arange(len(movie_ids))

    for i in range(len(movie_ids)):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="#eeeeee", alpha=0.5, zorder=0)

    def get_error_color(predicted, actual):
        """Get color based on absolute error."""
        error = abs(predicted - actual)
        max_error = 2.0
        normalized = min(error / max_error, 1.0)

        if normalized < 0.5:
            r = normalized * 2
            g = 1.0
            b = 0.0
        else:
            r = 1.0
            g = 1.0 - (normalized - 0.5) * 2
            b = 0.0
        return (r, g, b)

    model_names = list(predictions.keys())
    stagger_width = 0.3
    stagger_step = stagger_width / (len(model_names) - 1) if len(model_names) > 1 else 0

    best_model_per_movie = []
    for mid in movie_ids:
        actual = actual_ratings[mid]
        errors = {m: abs(predictions[m].get(mid, 3.0) - actual) for m in model_names}
        best_model = min(errors, key=errors.get)
        best_model_per_movie.append(best_model)

    for i, model_name in enumerate(model_names):
        preds = [predictions[model_name].get(mid, 3.0) for mid in movie_ids]
        offset = -stagger_width / 2 + i * stagger_step
        y_offset = y + offset

        marker = MODEL_MARKERS.get(model_name, "o")

        colors = []
        sizes = []
        linewidths = []
        edgecolors = []

        for j, (pred, mid) in enumerate(zip(preds, movie_ids)):
            actual = actual_ratings[mid]
            colors.append(get_error_color(pred, actual))

            if model_name == best_model_per_movie[j]:
                sizes.append(280)
                linewidths.append(2.5)
                edgecolors.append("black")
            else:
                sizes.append(180)
                linewidths.append(1.0)
                edgecolors.append("black")

        ax.scatter(
            preds,
            y_offset,
            s=sizes,
            c=colors,
            marker=marker,
            edgecolors=edgecolors,
            linewidths=linewidths,
            alpha=0.9,
            zorder=5,
        )

    actual_values = [actual_ratings[mid] for mid in movie_ids]
    ax.scatter(
        actual_values,
        y,
        s=350,
        marker="*",
        label="Actual Rating",
        zorder=10,
        c="blue",
        edgecolors="black",
        linewidths=1.5,
    )

    ax.set_xlabel("Ocena", fontweight="bold", fontsize=12)
    ax.set_ylabel("Film", fontweight="bold", fontsize=12)
    ax.set_title(
        f"Porównanie Predykcji Modeli dla Użytkownika {sample_user_id}", fontweight="bold", pad=25, fontsize=20, loc="center"
    )

    ax.set_yticks(y)
    ax.set_yticklabels(movie_labels, fontsize=11)

    ax.set_xlim(0.5, 5.5)
    ax.set_xticks(np.arange(1, 6))

    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.grid(axis="y", alpha=0.1)

    ax.axvline(x=2.5, color="gray", linestyle=":", alpha=0.5)

    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            label="Rzeczywista Ocena",
            markerfacecolor="blue",
            markersize=15,
            markeredgecolor="black",
        ),
    ]

    for model_name in model_names:
        marker = MODEL_MARKERS.get(model_name, "o")
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                color="w",
                label=model_name,
                markerfacecolor="lightgray",
                markersize=12,
                markeredgecolor="black",
            )
        )

    legend_elements.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Niski Błąd (Dobry)",
            markerfacecolor="lime",
            markersize=10,
            markeredgecolor="black",
        )
    )
    legend_elements.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Wysoki Błąd (Zły)",
            markerfacecolor="red",
            markersize=10,
            markeredgecolor="black",
        )
    )

    legend_elements.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Najlepszy Model",
            markerfacecolor="white",
            markersize=14,
            markeredgecolor="black",
            markeredgewidth=2.5,
        )
    )

    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=4,
        frameon=True,
        framealpha=0.9,
        borderaxespad=1.0,
        columnspacing=1.5,
        labelspacing=1.0,
        borderpad=2.0,
        handletextpad=1.0,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)

    output_path = output_dir / "models" / "porownanie_predykcji_modeli.png"
    plt.savefig(output_path)
    plt.close()

    print(f"\n✓ Saved plot to {output_path}")

    print("\nPredictions Summary:")
    print("=" * 80)
    print(f"{'Movie':<35} {'Actual':<8} " + " ".join(f"{m:<10}" for m in model_names))
    print("-" * 80)
    for i, mid in enumerate(movie_ids):
        movie_title = movie_labels[i][:30]
        actual = actual_ratings[mid]
        pred_str = " ".join(f"{predictions[m].get(mid, 3.0):.2f}      " for m in model_names)
        print(f"{movie_title:<35} {actual:<8.2f} {pred_str}")
    print("=" * 80)

    return output_path
