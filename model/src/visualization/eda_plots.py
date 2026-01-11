"""
Exploratory Data Analysis plots for the movie recommender system.

Generates visualizations for understanding the MovieLens dataset structure,
rating distributions, user behavior patterns, and temporal trends.

All labels are in Polish for academic thesis presentation.
"""

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns

from .plots import COLORS, POLISH_LABELS, set_thesis_style

EDA_LABELS = {
    **POLISH_LABELS,
    "user_activity": "Rozkład aktywności użytkowników",
    "movie_popularity": "Rozkład popularności filmów",
    "ratings_over_time": "Liczba ocen w czasie",
    "ratings_by_year": "Rozkład ocen według roku wydania filmu",
    "sparsity_matrix": "Macierz rzadkości interakcji użytkownik-film",
    "rating_by_genre": "Rozkład ocen według gatunku",
    "top_rated_movies": "Najwyżej oceniane filmy",
    "most_rated_movies": "Najczęściej oceniane filmy",
    "num_ratings": "Liczba ocen",
    "num_users": "Liczba użytkowników",
    "num_movies": "Liczba filmów",
    "year": "Rok",
    "month": "Miesiąc",
    "avg_rating": "Średnia ocena",
    "rating_count": "Liczba ocen",
    "density": "Gęstość",
    "log_scale": "(skala logarytmiczna)",
    "power_law": "Rozkład potęgowy",
    "sparsity": "Rzadkość",
}


def plot_rating_distribution_detailed(
    ratings: pd.Series,
    output_path: Path,
    title: Optional[str] = None,
):
    """
    Plot detailed histogram of rating distribution with statistics.

    Args:
        ratings: Series of ratings
        output_path: Path to save figure
        title: Optional custom title
    """
    set_thesis_style()

    if title is None:
        title = EDA_LABELS["rating_distribution"]

    fig, ax = plt.subplots(figsize=(10, 6))

    mean_rating = ratings.mean()
    median_rating = ratings.median()
    std_rating = ratings.std()
    total_ratings = len(ratings)

    bins = np.arange(0.25, 5.75, 0.5)
    counts, _, patches = ax.hist(ratings, bins=bins, color=COLORS[0], edgecolor="black", alpha=0.7, rwidth=0.85)

    for i, (patch, count) in enumerate(zip(patches, counts)):
        rating_val = 0.5 + i * 0.5
        if rating_val >= 4.0:
            patch.set_facecolor("#2ecc71")
        elif rating_val >= 3.0:
            patch.set_facecolor("#f1c40f")
        else:
            patch.set_facecolor("#e74c3c")

    ax.axvline(mean_rating, color="navy", linestyle="--", linewidth=2, label=f"{EDA_LABELS['mean']}: {mean_rating:.2f}")
    ax.axvline(
        median_rating, color="darkred", linestyle=":", linewidth=2, label=f"{EDA_LABELS['median']}: {median_rating:.2f}"
    )

    stats_text = f"n = {total_ratings:,}\n" f"σ = {std_rating:.2f}"
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax.set_xlabel(EDA_LABELS["rating"], fontweight="bold")
    ax.set_ylabel(EDA_LABELS["frequency"], fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=20)
    ax.set_xticks(np.arange(0.5, 5.5, 0.5))
    ax.legend(loc="upper left", frameon=True, shadow=True)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"✓ Saved rating distribution to {output_path}")


def plot_user_activity_distribution(
    ratings_df: pd.DataFrame,
    output_path: Path,
    log_scale: bool = True,
):
    """
    Plot distribution of ratings per user (power-law distribution).

    Args:
        ratings_df: DataFrame with userId column
        output_path: Path to save figure
        log_scale: Whether to use log scale for axes
    """
    set_thesis_style()

    user_counts = ratings_df.groupby("userId").size()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.hist(user_counts, bins=50, color=COLORS[0], edgecolor="black", alpha=0.7)
    if log_scale:
        ax1.set_yscale("log")
    ax1.set_xlabel(EDA_LABELS["num_ratings"], fontweight="bold")
    ax1.set_ylabel(EDA_LABELS["num_users"] + (" " + EDA_LABELS["log_scale"] if log_scale else ""), fontweight="bold")
    ax1.set_title(EDA_LABELS["user_activity"], fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    stats_text = (
        f"{EDA_LABELS['mean']}: {user_counts.mean():.1f}\n"
        f"{EDA_LABELS['median']}: {user_counts.median():.1f}\n"
        f"Min: {user_counts.min()}\n"
        f"Max: {user_counts.max():,}"
    )
    ax1.text(
        0.95,
        0.95,
        stats_text,
        transform=ax1.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax2 = axes[1]
    counts_sorted = np.sort(user_counts.values)[::-1]
    ranks = np.arange(1, len(counts_sorted) + 1)

    ax2.loglog(ranks, counts_sorted, "o", markersize=2, alpha=0.5, color=COLORS[0])
    ax2.set_xlabel("Pozycja użytkownika (ranking)", fontweight="bold")
    ax2.set_ylabel(EDA_LABELS["num_ratings"], fontweight="bold")
    ax2.set_title(f"{EDA_LABELS['user_activity']} - {EDA_LABELS['power_law']}", fontweight="bold")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"✓ Saved user activity distribution to {output_path}")


def plot_movie_popularity_distribution(
    ratings_df: pd.DataFrame,
    output_path: Path,
    log_scale: bool = True,
):
    """
    Plot distribution of ratings per movie (long-tail distribution).

    Args:
        ratings_df: DataFrame with movieId column
        output_path: Path to save figure
        log_scale: Whether to use log scale
    """
    set_thesis_style()

    movie_counts = ratings_df.groupby("movieId").size()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.hist(movie_counts, bins=50, color=COLORS[1], edgecolor="black", alpha=0.7)
    if log_scale:
        ax1.set_yscale("log")
    ax1.set_xlabel(EDA_LABELS["num_ratings"], fontweight="bold")
    ax1.set_ylabel(EDA_LABELS["num_movies"] + (" " + EDA_LABELS["log_scale"] if log_scale else ""), fontweight="bold")
    ax1.set_title(EDA_LABELS["movie_popularity"], fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    stats_text = (
        f"{EDA_LABELS['mean']}: {movie_counts.mean():.1f}\n"
        f"{EDA_LABELS['median']}: {movie_counts.median():.1f}\n"
        f"Min: {movie_counts.min()}\n"
        f"Max: {movie_counts.max():,}"
    )
    ax1.text(
        0.95,
        0.95,
        stats_text,
        transform=ax1.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax2 = axes[1]
    counts_sorted = np.sort(movie_counts.values)[::-1]
    cumsum = np.cumsum(counts_sorted)
    cumsum_pct = cumsum / cumsum[-1] * 100
    pct_movies = np.arange(1, len(counts_sorted) + 1) / len(counts_sorted) * 100

    ax2.plot(pct_movies, cumsum_pct, color=COLORS[1], linewidth=2)
    ax2.fill_between(pct_movies, cumsum_pct, alpha=0.3, color=COLORS[1])

    idx_80 = np.searchsorted(cumsum_pct, 80)
    pct_80 = pct_movies[idx_80] if idx_80 < len(pct_movies) else 100
    ax2.axhline(80, color="red", linestyle="--", alpha=0.7)
    ax2.axvline(pct_80, color="red", linestyle="--", alpha=0.7)
    ax2.text(pct_80 + 2, 82, f"{pct_80:.1f}% filmów = 80% ocen", fontsize=9, color="red")

    ax2.set_xlabel("% filmów (posortowane według popularności)", fontweight="bold")
    ax2.set_ylabel("% wszystkich ocen (skumulowane)", fontweight="bold")
    ax2.set_title("Krzywa skumulowana - efekt długiego ogona", fontweight="bold")
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"✓ Saved movie popularity distribution to {output_path}")


def plot_genre_frequency(
    ratings_df: pd.DataFrame,
    output_path: Path,
    top_n: int = 15,
):
    """
    Plot bar chart of genre frequencies in ratings.

    Args:
        ratings_df: DataFrame with genres column
        output_path: Path to save figure
        top_n: Number of top genres to show
    """
    set_thesis_style()

    all_genres = []
    for genre_str in ratings_df["genres"].dropna():
        if isinstance(genre_str, str):
            if "|" in genre_str:
                all_genres.extend(genre_str.split("|"))
            else:
                all_genres.extend([g.strip() for g in genre_str.split(",")])

    genre_counts = Counter(all_genres)
    genre_counts.pop("(no genres listed)", None)
    top_genres = genre_counts.most_common(top_n)

    fig, ax = plt.subplots(figsize=(12, 6))

    genres = [g for g, _ in top_genres]
    counts = [c for _, c in top_genres]

    y_pos = np.arange(len(genres))
    bars = ax.barh(y_pos, counts, color=COLORS[2], edgecolor="black", alpha=0.7)

    max_count = max(counts) if counts else 1
    for bar, count in zip(bars, counts):
        if count > 0.75 * max_count:
            ax.text(
                bar.get_width() * 0.98,
                bar.get_y() + bar.get_height() / 2,
                f"{count:,}",
                va="center",
                ha="right",
                fontsize=9,
                fontweight="bold",
                color="black",
            )
        else:
            ax.text(
                bar.get_width() + max_count * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{count:,}",
                va="center",
                ha="left",
                fontsize=9,
                fontweight="bold",
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(genres)
    ax.invert_yaxis()
    ax.set_xlabel(EDA_LABELS["frequency"], fontweight="bold")
    ax.set_ylabel(EDA_LABELS["genre"], fontweight="bold")
    ax.set_title(EDA_LABELS["genre_distribution"], fontweight="bold", pad=20)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"✓ Saved genre frequency to {output_path}")


def plot_ratings_over_time(
    ratings_df: pd.DataFrame,
    output_path: Path,
    resample_freq: str = "M",
):
    """
    Plot number of ratings over time.

    Args:
        ratings_df: DataFrame with timestamp column
        output_path: Path to save figure
        resample_freq: Pandas resample frequency ('M' for month, 'Y' for year)
    """
    set_thesis_style()

    if "timestamp" in ratings_df.columns:
        df = ratings_df.copy()
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
        df.set_index("datetime", inplace=True)

        ratings_over_time = df.resample(resample_freq).size()

        fig, ax = plt.subplots(figsize=(14, 6))

        ax.fill_between(ratings_over_time.index, ratings_over_time.values, alpha=0.3, color=COLORS[0])
        ax.plot(ratings_over_time.index, ratings_over_time.values, color=COLORS[0], linewidth=1.5)

        ax.set_xlabel(EDA_LABELS["year"] if resample_freq == "Y" else "Data", fontweight="bold")
        ax.set_ylabel(EDA_LABELS["num_ratings"], fontweight="bold")
        ax.set_title(EDA_LABELS["ratings_over_time"], fontweight="bold", pad=20)
        ax.grid(True, alpha=0.3)

        fig.autofmt_xdate()

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        print(f"✓ Saved ratings over time to {output_path}")
    else:
        print("⚠ No timestamp column found, skipping temporal plot")


def plot_rating_by_genre(
    ratings_df: pd.DataFrame,
    output_path: Path,
    top_n: int = 12,
):
    """
    Plot average rating and distribution by genre.

    Args:
        ratings_df: DataFrame with genres and rating columns
        output_path: Path to save figure
        top_n: Number of genres to show
    """
    set_thesis_style()

    genre_ratings = []
    for _, row in ratings_df.iterrows():
        genre_str = row.get("genres", "")
        rating = row.get("rating", 0)
        if isinstance(genre_str, str) and genre_str:
            if "|" in genre_str:
                genres = genre_str.split("|")
            else:
                genres = [g.strip() for g in genre_str.split(",")]
            for genre in genres:
                if genre and genre != "(no genres listed)":
                    genre_ratings.append({"genre": genre, "rating": rating})

    if not genre_ratings:
        print("⚠ No genre data found")
        return

    genre_df = pd.DataFrame(genre_ratings)

    top_genres = genre_df["genre"].value_counts().head(top_n).index.tolist()
    genre_df = genre_df[genre_df["genre"].isin(top_genres)]

    genre_means = genre_df.groupby("genre")["rating"].mean().sort_values(ascending=False)
    ordered_genres = genre_means.index.tolist()

    fig, ax = plt.subplots(figsize=(12, 6))

    genre_df["genre"] = pd.Categorical(genre_df["genre"], categories=ordered_genres, ordered=True)
    sns.boxplot(data=genre_df, x="genre", y="rating", ax=ax, palette="Set2")

    ax.set_xlabel(EDA_LABELS["genre"], fontweight="bold")
    ax.set_ylabel(EDA_LABELS["rating"], fontweight="bold")
    ax.set_title(EDA_LABELS["rating_by_genre"], fontweight="bold", pad=20)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(
        y=ratings_df["rating"].mean(),
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Średnia globalna: {ratings_df['rating'].mean():.2f}",
    )
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"✓ Saved rating by genre to {output_path}")


def plot_sparsity_visualization(
    ratings_df: pd.DataFrame,
    output_path: Path,
    sample_users: int = 500,
    sample_movies: int = 500,
):
    """
    Visualize the sparsity of the user-movie interaction matrix.

    Args:
        ratings_df: DataFrame with userId and movieId columns
        output_path: Path to save figure
        sample_users: Number of users to sample for visualization
        sample_movies: Number of movies to sample for visualization
    """
    set_thesis_style()

    unique_users = ratings_df["userId"].unique()
    unique_movies = ratings_df["movieId"].unique()

    if len(unique_users) > sample_users:
        sampled_users = np.random.choice(unique_users, sample_users, replace=False)
    else:
        sampled_users = unique_users

    if len(unique_movies) > sample_movies:
        sampled_movies = np.random.choice(unique_movies, sample_movies, replace=False)
    else:
        sampled_movies = unique_movies

    filtered_df = ratings_df[ratings_df["userId"].isin(sampled_users) & ratings_df["movieId"].isin(sampled_movies)]

    user_map = {u: i for i, u in enumerate(sorted(sampled_users))}
    movie_map = {m: i for i, m in enumerate(sorted(sampled_movies))}

    rows = [user_map[u] for u in filtered_df["userId"]]
    cols = [movie_map[m] for m in filtered_df["movieId"]]

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(cols, rows, s=0.5, c=COLORS[0], alpha=0.5)

    total_cells = len(sampled_users) * len(sampled_movies)
    filled_cells = len(filtered_df)
    sparsity = 1 - (filled_cells / total_cells)
    density = filled_cells / total_cells * 100

    ax.set_xlabel(EDA_LABELS["movie"] + f" (próbka: {len(sampled_movies)})", fontweight="bold")
    ax.set_ylabel(EDA_LABELS["user"] + f" (próbka: {len(sampled_users)})", fontweight="bold")
    ax.set_title(
        f"{EDA_LABELS['sparsity_matrix']}\n" f"Rzadkość: {sparsity:.4%} | Gęstość: {density:.4f}%", fontweight="bold", pad=20
    )
    ax.set_xlim(0, len(sampled_movies))
    ax.set_ylim(0, len(sampled_users))
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"✓ Saved sparsity visualization to {output_path}")


def plot_top_movies(
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    output_path: Path,
    top_n: int = 20,
    min_ratings: int = 100,
    by: str = "rating",
):
    """
    Plot top movies by average rating or rating count.

    Args:
        ratings_df: DataFrame with movieId and rating columns
        movies_df: DataFrame with movieId and title columns
        output_path: Path to save figure
        top_n: Number of movies to show
        min_ratings: Minimum number of ratings required
        by: Sort by "rating" (average) or "count" (number of ratings)
    """
    set_thesis_style()

    movie_stats = (
        ratings_df.groupby("movieId").agg(avg_rating=("rating", "mean"), rating_count=("rating", "count")).reset_index()
    )

    movie_stats = movie_stats[movie_stats["rating_count"] >= min_ratings]

    if by == "rating":
        movie_stats = movie_stats.sort_values("avg_rating", ascending=False).head(top_n)
        title = EDA_LABELS["top_rated_movies"] + f" (min. {min_ratings} ocen)"
        value_col = "avg_rating"
        xlabel = EDA_LABELS["avg_rating"]
    else:
        movie_stats = movie_stats.sort_values("rating_count", ascending=False).head(top_n)
        title = EDA_LABELS["most_rated_movies"]
        value_col = "rating_count"
        xlabel = EDA_LABELS["num_ratings"]

    title_col = "title_ml" if "title_ml" in movies_df.columns else "title"
    movie_titles = dict(zip(movies_df["movieId"], movies_df[title_col]))
    movie_stats["title"] = movie_stats["movieId"].map(movie_titles)
    movie_stats["title"] = movie_stats["title"].fillna("Unknown").apply(lambda x: x[:40] + "..." if len(str(x)) > 40 else x)

    fig, ax = plt.subplots(figsize=(12, 8))

    y_pos = np.arange(len(movie_stats))
    values = movie_stats[value_col].values

    bars = ax.barh(y_pos, values, color=COLORS[3], edgecolor="black", alpha=0.7)

    max_val = max(values) if len(values) > 0 else 1
    for bar, val in zip(bars, values):
        if by == "rating":
            label = f"{val:.2f}"
        else:
            label = f"{val:,}"

        if val > 0.75 * max_val:
            ax.text(
                bar.get_width() * 0.98,
                bar.get_y() + bar.get_height() / 2,
                label,
                va="center",
                ha="right",
                fontsize=9,
                fontweight="bold",
                color="black",
            )
        else:
            ax.text(
                bar.get_width() + max_val * 0.01,
                bar.get_y() + bar.get_height() / 2,
                label,
                va="center",
                ha="left",
                fontsize=9,
                fontweight="bold",
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(movie_stats["title"])
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontweight="bold")
    ax.set_ylabel(EDA_LABELS["movie"], fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=20)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"✓ Saved top movies to {output_path}")


def generate_all_eda_plots(
    data_dir: Path,
    output_dir: Path,
):
    """
    Generate all EDA plots.

    Args:
        data_dir: Path to processed data directory
        output_dir: Path to save plots (will create 'eda' subdirectory)
    """
    print("\n" + "=" * 80)
    print("Generowanie wykresów eksploracyjnej analizy danych (EDA)")
    print("=" * 80)

    eda_dir = output_dir / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)

    print("\nŁadowanie danych...")
    train_file = data_dir / "train.parquet"
    movies_file = data_dir / "movies.parquet"

    if not train_file.exists():
        print(f"⚠ Nie znaleziono pliku treningowego: {train_file}")
        return

    train_df = pl.read_parquet(train_file).to_pandas()
    print(f"  Załadowano {len(train_df):,} ocen treningowych")

    movies_df = None
    if movies_file.exists():
        movies_df = pl.read_parquet(movies_file).to_pandas()
        print(f"  Załadowano {len(movies_df):,} filmów")

    print("\nGenerowanie wykresów...")

    plot_rating_distribution_detailed(train_df["rating"], eda_dir / "rozklad_ocen.png")

    plot_user_activity_distribution(train_df, eda_dir / "aktywnosc_uzytkownikow.png")

    plot_movie_popularity_distribution(train_df, eda_dir / "popularnosc_filmow.png")

    if "genres" in train_df.columns:
        plot_genre_frequency(train_df, eda_dir / "czestotliwosc_gatunkow.png")

    if "timestamp" in train_df.columns:
        plot_ratings_over_time(train_df, eda_dir / "oceny_w_czasie.png")

    if "genres" in train_df.columns:
        sample_df = train_df.sample(n=min(100000, len(train_df)), random_state=42)
        plot_rating_by_genre(sample_df, eda_dir / "oceny_wg_gatunku.png")

    if movies_df is not None:
        plot_top_movies(train_df, movies_df, eda_dir / "najwyzej_oceniane.png", by="rating")
        plot_top_movies(train_df, movies_df, eda_dir / "najczesciej_oceniane.png", by="count")

    print("\n" + "=" * 80)
    print(f"✓ Wykresy EDA zapisane w {eda_dir}")
    print("=" * 80)


if __name__ == "__main__":
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "model" / "data" / "processed"
    PLOTS_DIR = PROJECT_ROOT / "model" / "plots"

    generate_all_eda_plots(DATA_DIR, PLOTS_DIR)
