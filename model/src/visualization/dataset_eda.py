"""
Dataset Exploratory Data Analysis.

Generates comprehensive visualizations and metrics for understanding
the original datasets (MovieLens, TMDB) and their preprocessing pipeline.

All labels are in Polish for academic thesis presentation.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns

from .plots import COLORS, set_thesis_style

DATASET_LABELS = {
    "movielens": "MovieLens",
    "tmdb": "TMDB",
    "combined": "Połączone",
    "dataset_overview": "Przegląd zbiorów danych",
    "dataset_sizes": "Rozmiary zbiorów danych",
    "data_pipeline": "Proces przetwarzania danych",
    "rating_coverage": "Pokrycie ocen",
    "movie_overlap": "Wspólne filmy między zbiorami",
    "tmdb_enrichment": "Wzbogacenie metadanych TMDB",
    "genre_comparison": "Porównanie gatunków",
    "temporal_coverage": "Pokrycie czasowe",
    "data_quality": "Jakość danych",
    "preprocessing_stats": "Statystyki przetwarzania",
    "num_movies": "Liczba filmów",
    "num_ratings": "Liczba ocen",
    "num_users": "Liczba użytkowników",
    "dataset": "Zbiór danych",
    "year": "Rok",
    "count": "Liczba",
    "percentage": "Procent",
    "density": "Gęstość",
    "with_metadata": "Z metadanymi",
    "without_metadata": "Bez metadanych",
    "overlap": "Wspólne",
    "unique_ml": "Tylko MovieLens",
    "unique_tmdb": "Tylko TMDB",
    "original": "Oryginalne",
    "after_cleaning": "Po czyszczeniu",
    "removed": "Usunięte",
}


def plot_dataset_sizes_comparison(
    ml_stats: Dict,
    tmdb_stats: Dict,
    combined_stats: Dict,
    output_path: Path,
):
    """
    Compare sizes of MovieLens, TMDB, and combined datasets.

    Args:
        ml_stats: MovieLens statistics
        tmdb_stats: TMDB statistics
        combined_stats: Combined dataset statistics
        output_path: Path to save figure
    """
    set_thesis_style()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    datasets = ["MovieLens", "TMDB", "Połączone"]
    colors_list = [COLORS[0], COLORS[1], COLORS[2]]

    movies = [
        ml_stats.get("num_movies", 0),
        tmdb_stats.get("num_movies", 0),
        combined_stats.get("num_movies", 0),
    ]
    axes[0].bar(datasets, movies, color=colors_list, alpha=0.7, edgecolor="black", linewidth=1.5)
    axes[0].set_ylabel(DATASET_LABELS["num_movies"], fontweight="bold", fontsize=11)
    axes[0].set_title("Liczba filmów", fontweight="bold", pad=15, fontsize=12)
    max_movies = max(movies) if movies else 1
    for i, (d, m) in enumerate(zip(datasets, movies)):
        if m > 0.75 * max_movies:
            axes[0].text(i, m * 0.5, f"{m:,}", ha="center", va="center", fontweight="bold", fontsize=10, color="black")
        else:
            axes[0].text(i, m, f"{m:,}", ha="center", va="bottom", fontweight="bold", fontsize=10)

    ratings = [
        ml_stats.get("num_ratings", 0),
        0,
        combined_stats.get("num_ratings", 0),
    ]
    axes[1].bar(datasets, ratings, color=colors_list, alpha=0.7, edgecolor="black", linewidth=1.5)
    axes[1].set_ylabel(DATASET_LABELS["num_ratings"], fontweight="bold", fontsize=11)
    axes[1].set_title("Liczba ocen", fontweight="bold", pad=15, fontsize=12)
    max_ratings = max(ratings) if ratings else 1
    for i, (d, r) in enumerate(zip(datasets, ratings)):
        if r > 0:
            if r > 0.75 * max_ratings:
                axes[1].text(i, r * 0.5, f"{r:,}", ha="center", va="center", fontweight="bold", fontsize=10, color="black")
            else:
                axes[1].text(i, r, f"{r:,}", ha="center", va="bottom", fontweight="bold", fontsize=10)
        else:
            axes[1].text(i, 0, "brak", ha="center", va="center", fontweight="bold", fontsize=10, color="gray")

    users = [
        ml_stats.get("num_users", 0),
        0,
        combined_stats.get("num_users", 0),
    ]
    axes[2].bar(datasets, users, color=colors_list, alpha=0.7, edgecolor="black", linewidth=1.5)
    axes[2].set_ylabel(DATASET_LABELS["num_users"], fontweight="bold", fontsize=11)
    axes[2].set_title("Liczba użytkowników", fontweight="bold", pad=15, fontsize=12)
    max_users = max(users) if users else 1
    for i, (d, u) in enumerate(zip(datasets, users)):
        if u > 0:
            if u > 0.75 * max_users:
                axes[2].text(i, u * 0.5, f"{u:,}", ha="center", va="center", fontweight="bold", fontsize=10, color="black")
            else:
                axes[2].text(i, u, f"{u:,}", ha="center", va="bottom", fontweight="bold", fontsize=10)
        else:
            axes[2].text(i, 0, "brak", ha="center", va="center", fontweight="bold", fontsize=10, color="gray")

    plt.suptitle("Porównanie rozmiarów zbiorów danych", fontweight="bold", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"✓ Zapisano porównanie rozmiarów do {output_path}")


def plot_movie_overlap_venn(
    ml_movies: set,
    tmdb_movies: set,
    output_path: Path,
):
    """
    Visualize overlap between MovieLens and TMDB movie IDs.

    Args:
        ml_movies: Set of MovieLens movie IDs
        tmdb_movies: Set of TMDB movie IDs
        output_path: Path to save figure
    """
    set_thesis_style()

    fig, ax = plt.subplots(figsize=(10, 8))

    overlap = ml_movies & tmdb_movies
    ml_only = ml_movies - tmdb_movies
    tmdb_only = tmdb_movies - ml_movies

    categories = ["Tylko\nMovieLens", "Wspólne", "Tylko\nTMDB"]
    values = [len(ml_only), len(overlap), len(tmdb_only)]
    colors_list = [COLORS[0], COLORS[2], COLORS[1]]

    bars = ax.bar(categories, values, color=colors_list, alpha=0.7, edgecolor="black", linewidth=2)

    max_val = max(values) if values else 1
    for i, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()
        if val > 0.75 * max_val:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height * 0.5,
                f"{val:,}\n({val/sum(values)*100:.1f}%)",
                ha="center",
                va="center",
                fontweight="bold",
                fontsize=11,
                color="black",
            )
        else:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{val:,}\n({val/sum(values)*100:.1f}%)",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=11,
            )

    ax.set_ylabel(DATASET_LABELS["num_movies"], fontweight="bold", fontsize=12)
    ax.set_title("Wspólne filmy między zbiorami danych", fontweight="bold", fontsize=14, pad=20)

    total = len(ml_movies | tmdb_movies)
    summary_text = f"Razem unikalnych filmów: {total:,}\n"
    summary_text += f"Pokrycie: {len(overlap)/len(ml_movies)*100:.1f}% filmów ML ma metadane TMDB"
    ax.text(
        0.5,
        0.98,
        summary_text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.5),
        fontsize=10,
    )

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"✓ Zapisano wizualizację wspólnych filmów do {output_path}")


def plot_tmdb_enrichment(
    movies_df: pd.DataFrame,
    output_path: Path,
):
    """
    Show which MovieLens movies have TMDB metadata.

    Args:
        movies_df: Movies dataframe with tmdbId column
        output_path: Path to save figure
    """
    set_thesis_style()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    has_tmdb = movies_df["tmdbId"].notna().sum()
    no_tmdb = movies_df["tmdbId"].isna().sum()

    labels = ["Z metadanymi TMDB", "Bez metadanych TMDB"]
    sizes = [has_tmdb, no_tmdb]
    colors_list = [COLORS[2], COLORS[3]]
    explode = (0.05, 0)

    axes[0].pie(
        sizes,
        labels=labels,
        colors=colors_list,
        autopct=lambda pct: f"{pct:.1f}%\n({int(pct/100*sum(sizes)):,})",
        startangle=90,
        explode=explode,
        textprops={"fontweight": "bold"},
    )
    axes[0].set_title("Pokrycie metadanych TMDB", fontweight="bold", pad=15)

    if "overview" in movies_df.columns:
        has_overview = movies_df["overview"].notna().sum()
        has_nonempty_overview = (movies_df["overview"].fillna("").str.len() > 0).sum()

        categories = ["Wszystkie\nfilmy", "Z TMDB ID", "Z opisem", "Z niepustym\napisem"]
        values = [len(movies_df), has_tmdb, has_overview, has_nonempty_overview]

        bars = axes[1].bar(range(len(categories)), values, color=COLORS[:4], alpha=0.7, edgecolor="black")
        axes[1].set_xticks(range(len(categories)))
        axes[1].set_xticklabels(categories)
        axes[1].set_ylabel(DATASET_LABELS["num_movies"], fontweight="bold")
        axes[1].set_title("Dostępność metadanych", fontweight="bold", pad=15)

        max_val = max(values) if values else 1
        for bar, val in zip(bars, values):
            height = bar.get_height()
            if val > 0.75 * max_val:
                axes[1].text(
                    bar.get_x() + bar.get_width() / 2,
                    height * 0.5,
                    f"{val:,}",
                    ha="center",
                    va="center",
                    fontweight="bold",
                    color="black",
                )
            else:
                axes[1].text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{val:,}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"✓ Zapisano wizualizację wzbogacenia TMDB do {output_path}")


def plot_genre_distribution_comparison(
    ml_df: pd.DataFrame,
    tmdb_df: pd.DataFrame,
    output_path: Path,
    top_n: int = 15,
):
    """
    Compare genre distributions between MovieLens and TMDB.

    Args:
        ml_df: MovieLens movies dataframe
        tmdb_df: TMDB movies dataframe
        output_path: Path to save figure
        top_n: Number of top genres to show
    """
    set_thesis_style()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ml_genres = []
    for genres_str in ml_df["genres"].dropna():
        if genres_str and genres_str != "(no genres listed)":
            ml_genres.extend(genres_str.split("|"))
    ml_genre_counts = Counter(ml_genres).most_common(top_n)

    tmdb_genres = []
    for genres_str in tmdb_df["genres"].dropna():
        if genres_str and genres_str != "(no genres listed)":
            if "," in genres_str:
                tmdb_genres.extend([g.strip() for g in genres_str.split(",")])
            else:
                tmdb_genres.extend(genres_str.split("|"))
    tmdb_genre_counts = Counter(tmdb_genres).most_common(top_n)

    genres_ml, counts_ml = zip(*ml_genre_counts) if ml_genre_counts else ([], [])
    axes[0].barh(range(len(genres_ml)), counts_ml, color=COLORS[0], alpha=0.7, edgecolor="black")
    axes[0].set_yticks(range(len(genres_ml)))
    axes[0].set_yticklabels(genres_ml)
    axes[0].set_xlabel(DATASET_LABELS["num_movies"], fontweight="bold")
    axes[0].set_title("MovieLens - rozkład gatunków", fontweight="bold", pad=15)
    axes[0].invert_yaxis()

    max_count_ml = max(counts_ml) if counts_ml else 1
    for i, count in enumerate(counts_ml):
        if count > 0.75 * max_count_ml:
            axes[0].text(count * 0.98, i, f"{count:,}", va="center", ha="right", fontweight="bold", color="black")
        else:
            axes[0].text(count, i, f" {count:,}", va="center", ha="left", fontweight="bold")

    genres_tmdb, counts_tmdb = zip(*tmdb_genre_counts) if tmdb_genre_counts else ([], [])
    axes[1].barh(range(len(genres_tmdb)), counts_tmdb, color=COLORS[1], alpha=0.7, edgecolor="black")
    axes[1].set_yticks(range(len(genres_tmdb)))
    axes[1].set_yticklabels(genres_tmdb)
    axes[1].set_xlabel(DATASET_LABELS["num_movies"], fontweight="bold")
    axes[1].set_title("TMDB - rozkład gatunków", fontweight="bold", pad=15)
    axes[1].invert_yaxis()

    max_count_tmdb = max(counts_tmdb) if counts_tmdb else 1
    for i, count in enumerate(counts_tmdb):
        if count > 0.75 * max_count_tmdb:
            axes[1].text(count * 0.98, i, f"{count:,}", va="center", ha="right", fontweight="bold", color="black")
        else:
            axes[1].text(count, i, f" {count:,}", va="center", ha="left", fontweight="bold")

    plt.suptitle(f"Porównanie {top_n} najpopularniejszych gatunków", fontweight="bold", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"✓ Zapisano porównanie rozkładu gatunków do {output_path}")


def plot_temporal_coverage(
    ml_df: pd.DataFrame,
    tmdb_df: pd.DataFrame,
    output_path: Path,
):
    """
    Compare temporal coverage (release years) between datasets.

    Args:
        ml_df: MovieLens movies dataframe
        tmdb_df: TMDB movies dataframe
        output_path: Path to save figure
    """
    set_thesis_style()

    fig, ax = plt.subplots(figsize=(12, 6))

    ml_years = ml_df["year"].dropna().astype(int)

    if "release_date" in tmdb_df.columns:
        tmdb_years = pd.to_datetime(tmdb_df["release_date"], errors="coerce").dt.year.dropna().astype(int)
    elif "year" in tmdb_df.columns:
        tmdb_years = tmdb_df["year"].dropna().astype(int)
    else:
        tmdb_years = pd.Series([], dtype=int)

    bins = np.arange(1900, 2030, 5)
    ax.hist(ml_years, bins=bins, alpha=0.6, label="MovieLens", color=COLORS[0], edgecolor="black")
    if len(tmdb_years) > 0:
        ax.hist(tmdb_years, bins=bins, alpha=0.6, label="TMDB", color=COLORS[1], edgecolor="black")

    ax.set_xlabel(DATASET_LABELS["year"], fontweight="bold", fontsize=12)
    ax.set_ylabel(DATASET_LABELS["num_movies"], fontweight="bold", fontsize=12)
    ax.set_title("Pokrycie czasowe - rok wydania filmów", fontweight="bold", fontsize=14, pad=15)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    stats_text = f"MovieLens: {ml_years.min():.0f} - {ml_years.max():.0f}\n"
    if len(tmdb_years) > 0:
        stats_text += f"TMDB: {tmdb_years.min():.0f} - {tmdb_years.max():.0f}"
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        va="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.5),
        fontsize=10,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"✓ Zapisano wizualizację pokrycia czasowego do {output_path}")


def plot_preprocessing_pipeline(
    preprocessing_stats: Dict,
    output_path: Path,
):
    """
    Visualize the data preprocessing pipeline steps.

    Args:
        preprocessing_stats: Statistics from preprocessing
        output_path: Path to save figure
    """
    set_thesis_style()

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    splits = ["Treningowy", "Walidacyjny", "Testowy"]
    split_counts = [
        preprocessing_stats.get("train_ratings", 0),
        preprocessing_stats.get("val_ratings", 0),
        preprocessing_stats.get("test_ratings", 0),
    ]
    split_percentages = [c / sum(split_counts) * 100 for c in split_counts]

    axes[0].bar(splits, split_counts, color=COLORS[:3], alpha=0.7, edgecolor="black", linewidth=1.5)
    axes[0].set_ylabel(DATASET_LABELS["num_ratings"], fontweight="bold", fontsize=11)
    axes[0].set_title("Podział danych", fontweight="bold", pad=15, fontsize=12)
    max_split = max(split_counts) if split_counts else 1
    for i, (split, count, pct) in enumerate(zip(splits, split_counts, split_percentages)):
        if count > 0.75 * max_split:
            axes[0].text(
                i,
                count * 0.5,
                f"{count:,}\n({pct:.1f}%)",
                ha="center",
                va="center",
                fontweight="bold",
                fontsize=9,
                color="black",
            )
        else:
            axes[0].text(i, count, f"{count:,}\n({pct:.1f}%)", ha="center", va="bottom", fontweight="bold", fontsize=9)

    rating_stats = {
        "Min.": preprocessing_stats.get("rating_min", 0),
        "Średnia": preprocessing_stats.get("rating_mean", 0),
        "Maks.": preprocessing_stats.get("rating_max", 0),
    }
    axes[1].bar(rating_stats.keys(), rating_stats.values(), color=COLORS[0], alpha=0.7, edgecolor="black", linewidth=1.5)
    axes[1].set_ylabel("Ocena", fontweight="bold", fontsize=11)
    axes[1].set_title("Statystyki ocen", fontweight="bold", pad=15, fontsize=12)
    axes[1].set_ylim(0, 5.5)
    max_rating = max(rating_stats.values()) if rating_stats.values() else 1
    for i, (stat, val) in enumerate(rating_stats.items()):
        if val > 0.75 * 5.5:
            axes[1].text(i, val * 0.5, f"{val:.2f}", ha="center", va="center", fontweight="bold", fontsize=10, color="black")
        else:
            axes[1].text(i, val, f"{val:.2f}", ha="center", va="bottom", fontweight="bold", fontsize=10)

    plt.suptitle("Pipeline przetwarzania danych", fontweight="bold", fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"✓ Zapisano wizualizację pipeline przetwarzania do {output_path}")


def generate_dataset_statistics(
    data_dir: Path,
    output_dir: Path,
) -> Dict:
    """
    Generate comprehensive statistics for all datasets.

    Args:
        data_dir: Directory containing processed data
        output_dir: Directory to save statistics JSON files

    Returns:
        Dictionary with all statistics
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerowanie statystyk zbiorów danych...")

    train_df = pl.read_parquet(data_dir / "train.parquet")
    val_df = pl.read_parquet(data_dir / "val.parquet")
    test_df = pl.read_parquet(data_dir / "test.parquet")
    movies_df = pl.read_parquet(data_dir / "movies.parquet")

    ml_stats = {
        "num_movies": len(movies_df),
        "num_users": train_df.select("userId").unique().height,
        "num_ratings": len(train_df) + len(val_df) + len(test_df),
        "train_ratings": len(train_df),
        "val_ratings": len(val_df),
        "test_ratings": len(test_df),
        "rating_min": float(train_df["rating"].min()),
        "rating_max": float(train_df["rating"].max()),
        "rating_mean": float(train_df["rating"].mean()),
        "rating_std": float(train_df["rating"].std()),
        "movies_with_tmdb": int(movies_df.filter(pl.col("tmdbId").is_not_null()).height),
        "movies_without_tmdb": int(movies_df.filter(pl.col("tmdbId").is_null()).height),
    }

    with open(output_dir / "movielens_stats.json", "w") as f:
        json.dump(ml_stats, f, indent=2)
    print(f"  ✓ Saved movielens_stats.json")

    tmdb_csv_path = data_dir.parent / "raw" / "tmdb" / "TMDB_movie_dataset_v11.csv"
    tmdb_stats = {}
    if tmdb_csv_path.exists():
        tmdb_df = pd.read_csv(tmdb_csv_path)
        tmdb_stats = {
            "num_movies": len(tmdb_df),
            "num_with_overview": int((tmdb_df["overview"].notna() & (tmdb_df["overview"].str.len() > 10)).sum()),
            "num_with_genres": int(tmdb_df["genres"].notna().sum()),
            "year_min": int(pd.to_datetime(tmdb_df["release_date"], errors="coerce").dt.year.min()),
            "year_max": int(pd.to_datetime(tmdb_df["release_date"], errors="coerce").dt.year.max()),
            "popularity_mean": float(tmdb_df["popularity"].mean()),
            "vote_average_mean": float(tmdb_df["vote_average"].mean()),
        }

        with open(output_dir / "tmdb_stats.json", "w") as f:
            json.dump(tmdb_stats, f, indent=2)
        print(f"  ✓ Saved tmdb_stats.json")

    combined_stats = {
        "num_movies": ml_stats["num_movies"],
        "num_users": ml_stats["num_users"],
        "num_ratings": ml_stats["num_ratings"],
        "coverage_tmdb_metadata": ml_stats["movies_with_tmdb"] / ml_stats["num_movies"] * 100,
        "preprocessing": {
            "train_ratio": ml_stats["train_ratings"] / ml_stats["num_ratings"],
            "val_ratio": ml_stats["val_ratings"] / ml_stats["num_ratings"],
            "test_ratio": ml_stats["test_ratings"] / ml_stats["num_ratings"],
        },
    }

    with open(output_dir / "combined_stats.json", "w") as f:
        json.dump(combined_stats, f, indent=2)
    print(f"  ✓ Saved combined_stats.json")

    return {
        "movielens": ml_stats,
        "tmdb": tmdb_stats,
        "combined": combined_stats,
    }


def generate_all_dataset_eda(
    data_dir: Path,
    plots_dir: Path,
) -> None:
    """
    Generate all dataset EDA plots and statistics.

    Args:
        data_dir: Directory containing processed data
        plots_dir: Directory to save plots
    """
    print("=" * 80)
    print("Dataset Exploratory Data Analysis")
    print("=" * 80)

    eda_dir = plots_dir / "eda" / "datasets"
    eda_dir.mkdir(parents=True, exist_ok=True)

    metrics_dir = plots_dir.parent / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    all_stats = generate_dataset_statistics(data_dir, metrics_dir)
    ml_stats = all_stats["movielens"]
    tmdb_stats = all_stats["tmdb"]
    combined_stats = all_stats["combined"]

    print("\nŁadowanie danych do wizualizacji...")
    movies_df = pd.read_parquet(data_dir / "movies.parquet")
    print(f"  Załadowano {len(movies_df):,} filmów")

    tmdb_csv_path = data_dir.parent / "raw" / "tmdb" / "TMDB_movie_dataset_v11.csv"
    if tmdb_csv_path.exists():
        print(f"  Ładowanie danych TMDB z {tmdb_csv_path.name}...")
        tmdb_df = pd.read_csv(tmdb_csv_path)
        print(f"  Załadowano {len(tmdb_df):,} filmów TMDB")
    else:
        print(f"  ⚠ Nie znaleziono pliku TMDB")
        tmdb_df = pd.DataFrame()

    print("\nGenerowanie wykresów...")

    plot_dataset_sizes_comparison(ml_stats, tmdb_stats, combined_stats, eda_dir / "porownanie_rozmiarow.png")

    if not tmdb_df.empty:
        ml_movie_ids = set(movies_df["tmdbId"].dropna().astype(int))
        tmdb_movie_ids = set(tmdb_df["id"].dropna().astype(int))
        plot_movie_overlap_venn(ml_movie_ids, tmdb_movie_ids, eda_dir / "wspolne_filmy.png")

    plot_tmdb_enrichment(movies_df, eda_dir / "wzbogacenie_tmdb.png")

    if not tmdb_df.empty:
        plot_genre_distribution_comparison(movies_df, tmdb_df, eda_dir / "porownanie_gatunkow.png")

    if not tmdb_df.empty:
        plot_temporal_coverage(movies_df, tmdb_df, eda_dir / "pokrycie_czasowe.png")

    preprocessing_stats = {
        **ml_stats,
        "total_ratings": ml_stats["num_ratings"],
    }
    plot_preprocessing_pipeline(preprocessing_stats, eda_dir / "pipeline_przetwarzania.png")

    print("\n" + "=" * 80)
    print(f"✓ Analiza EDA zbiorów danych zakończona! Wykresy zapisane w {eda_dir}")
    print(f"✓ Statystyki zapisane w {metrics_dir}")
    print("=" * 80)


if __name__ == "__main__":
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "model" / "data" / "processed"
    PLOTS_DIR = PROJECT_ROOT / "model" / "plots"

    generate_all_dataset_eda(DATA_DIR, PLOTS_DIR)
