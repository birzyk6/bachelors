"""
Embedding visualization plots for the movie recommender system.

Precomputes and visualizes embeddings using t-SNE, PCA, and UMAP for:
- 80K movies with ratings (Two-Tower embeddings)
- 650K TMDB embeddings (full dataset)

For each dataset generates:
- Comparison plots (t-SNE vs PCA vs UMAP)
- Genre-colored visualizations
- Cluster analysis (with and without outlined edges, most popular movies)
- Density maps

All labels are in Polish for academic thesis presentation.
"""

import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from umap import UMAP

from .plots import set_thesis_style

# Polish labels for dimensionality reduction methods
METHOD_LABELS = {
    "tsne": {"dim1": "t-SNE wymiar 1", "dim2": "t-SNE wymiar 2", "name": "t-SNE"},
    "pca": {"dim1": "PCA składowa 1", "dim2": "PCA składowa 2", "name": "PCA"},
    "umap": {"dim1": "UMAP wymiar 1", "dim2": "UMAP wymiar 2", "name": "UMAP"},
}

# Dataset labels
DATASET_LABELS = {
    "rated": "80K filmów z ocenami",
    "tmdb": "650K filmów TMDB (cały zbiór)",
}

# Vibrant rainbow colormap for scatter plots
COLORMAP = "rainbow"

# Genre colors - grouped by theme with similar colors for related genres
GENRE_COLORS = {
    # Action/Adventure/Thriller family - Reds to Oranges
    "Action": "#FF5722",  # Deep orange
    "Adventure": "#FF9800",  # Orange
    "Thriller": "#F44336",  # Red
    # Drama/Crime/Mystery family - Deep blues to Purples
    "Drama": "#3F51B5",  # Indigo
    "Crime": "#283593",  # Dark indigo
    "Mystery": "#5E35B1",  # Deep purple
    "Film-Noir": "#1A237E",  # Very dark blue
    # Comedy/Family/Children family - Yellows to Light colors
    "Comedy": "#FFEB3B",  # Yellow
    "Family": "#FFC107",  # Amber
    "Children": "#FFD54F",  # Light amber
    # Sci-Fi/Fantasy family - Greens to Teals
    "Sci-Fi": "#00BCD4",  # Cyan
    "Science Fiction": "#00BCD4",  # Cyan (TMDB equivalent)
    "Fantasy": "#009688",  # Teal
    "IMAX": "#00ACC1",  # Light cyan
    # Romance/Musical family - Pinks
    "Romance": "#E91E63",  # Pink
    "Musical": "#F06292",  # Light pink
    "Music": "#F06292",  # Light pink (TMDB equivalent)
    # Horror - Dark purple
    "Horror": "#4A148C",  # Very dark purple
    # Animation - Bright magenta
    "Animation": "#FF4081",  # Bright pink/magenta
    # Documentary/History/War family - Grays to Browns
    "Documentary": "#607D8B",  # Blue gray
    "History": "#795548",  # Brown
    "War": "#5D4037",  # Dark brown
    "Western": "#8D6E63",  # Light brown
    "TV Movie": "#78909C",  # Light blue gray
    # Other/Unknown
    "Other": "#9E9E9E",  # Gray
}


def get_primary_genre(genres_str: str) -> str:
    """Extract primary genre from genre string."""
    if not genres_str or pd.isna(genres_str) or genres_str == "(no genres listed)":
        return "Other"
    if isinstance(genres_str, str):
        if "|" in genres_str:
            genre = genres_str.split("|")[0].strip()
            return "Other" if genre == "(no genres listed)" else genre
        elif "," in genres_str:
            genre = genres_str.split(",")[0].strip()
            return "Other" if genre == "(no genres listed)" else genre
        return "Other" if genres_str == "(no genres listed)" else genres_str.strip()
    return "Other"


def get_all_genres(genres_str: str) -> List[str]:
    """Extract all genres from genre string."""
    if not genres_str or pd.isna(genres_str) or genres_str == "(no genres listed)":
        return []
    if isinstance(genres_str, str):
        if "|" in genres_str:
            genres = [g.strip() for g in genres_str.split("|") if g.strip() and g.strip() != "(no genres listed)"]
            return genres if genres else []
        elif "," in genres_str:
            genres = [g.strip() for g in genres_str.split(",") if g.strip() and g.strip() != "(no genres listed)"]
            return genres if genres else []
        return [] if genres_str == "(no genres listed)" else [genres_str.strip()]
    return []


# =============================================================================
# CLUSTER EVALUATION FUNCTIONS
# =============================================================================


def get_or_calculate_optimal_k(
    embeddings: np.ndarray,
    cache_file: Path,
    output_dir: Path,
    dataset_name: str,
    max_k: int = 20,
) -> int:
    """
    Get cached optimal k or calculate it using elbow and silhouette methods.
    Saves the result for future use.
    """
    if cache_file.exists():
        print(f"  ✓ Wczytywanie zapisanego optymalnego k z {cache_file.name}")
        with open(cache_file) as f:
            data = json.load(f)
            optimal_k = data.get("optimal_k")
            print(f"  ✓ Użyto zapisanego k={optimal_k} (wynik sylwetkowy={data.get('silhouette_score', 'N/A'):.4f})")
            return optimal_k

    print(f"  Obliczanie optymalnej liczby klastrów...")

    # Generate elbow plot
    plot_elbow_method(
        embeddings,
        output_dir / f"{dataset_name}_elbow.png",
        max_k=max_k,
    )

    # Generate silhouette plot and get optimal k
    optimal_k = plot_silhouette_method(
        embeddings,
        output_dir / f"{dataset_name}_silhouette.png",
        max_k=max_k,
    )

    # Save optimal k for future use
    cache_data = {
        "optimal_k": optimal_k,
        "max_k_tested": max_k,
        "silhouette_score": float("NaN"),  # Will be updated in plot_silhouette_method
        "dataset_name": dataset_name,
    }

    with open(cache_file, "w") as f:
        json.dump(cache_data, f, indent=2)

    print(f"  💾 Zapisano optymalne k={optimal_k} do {cache_file}")

    return optimal_k


def plot_elbow_method(
    embeddings: np.ndarray,
    output_path: Path,
    max_k: int = 20,
):
    """
    Plot elbow curve to determine optimal number of clusters.
    """
    set_thesis_style()

    print(f"  Obliczanie metody łokcia dla k=2..{max_k}...")

    inertias = []
    k_range = range(2, max_k + 1)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(embeddings)
        inertias.append(kmeans.inertia_)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_range, inertias, "bo-", linewidth=2, markersize=8)
    ax.set_xlabel("Liczba klastrów (k)", fontweight="bold", fontsize=11)
    ax.set_ylabel("Suma kwadratów odległości wewnątrz klastrów", fontweight="bold", fontsize=11)
    ax.set_title("Metoda łokcia: wyznaczanie optymalnej liczby klastrów", fontweight="bold", fontsize=13, pad=15)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Zapisano wykres metody łokcia do {output_path.name}")


def plot_silhouette_method(
    embeddings: np.ndarray,
    output_path: Path,
    max_k: int = 20,
):
    """
    Plot silhouette scores to determine optimal number of clusters.
    """
    set_thesis_style()

    print(f"  Obliczanie współczynników sylwetkowych dla k=2..{max_k}...")

    silhouette_scores = []
    k_range = range(2, max_k + 1)

    # Use larger sample for more reliable scores
    sample_size = min(30000, len(embeddings))

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels, sample_size=sample_size)
        silhouette_scores.append(score)
        print(f"    k={k}: silhouette={score:.4f}")

    # Find best k - look for first significant local maximum in reasonable range (3-15)
    best_k_idx = 0
    max_score_in_range = -1

    for i, k in enumerate(k_range):
        if 3 <= k <= 15:  # Focus on reasonable range
            if silhouette_scores[i] > max_score_in_range:
                max_score_in_range = silhouette_scores[i]
                best_k_idx = i

    # If no good solution in range, use absolute maximum
    if max_score_in_range == -1:
        best_k_idx = np.argmax(silhouette_scores)

    best_k = list(k_range)[best_k_idx]
    best_score = silhouette_scores[best_k_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_range, silhouette_scores, "go-", linewidth=2, markersize=8)
    ax.axvline(best_k, color="red", linestyle="--", linewidth=2, label=f"Optimum: k={best_k} (wynik={best_score:.3f})")
    ax.set_xlabel("Liczba klastrów (k)", fontweight="bold", fontsize=11)
    ax.set_ylabel("Średni współczynnik sylwetkowy", fontweight="bold", fontsize=11)
    ax.set_title("Metoda sylwetkowa: wyznaczanie optymalnej liczby klastrów", fontweight="bold", fontsize=13, pad=15)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Zapisano wykres metody sylwetkowej do {output_path.name}")
    print(f"  ✓ Najlepsza liczba klastrów: k={best_k} (wynik={best_score:.4f})")

    # Update cache file with silhouette score if it exists
    cache_file = output_path.parent / f"{output_path.stem.replace('_silhouette', '')}_optimal_k.json"
    if cache_file.exists():
        try:
            with open(cache_file) as f:
                cache_data = json.load(f)
            cache_data["silhouette_score"] = float(best_score)
            with open(cache_file, "w") as f:
                json.dump(cache_data, f, indent=2)
        except Exception:
            pass


# =============================================================================
# PRECOMPUTATION FUNCTIONS
# =============================================================================


def precompute_dimensionality_reductions(
    embeddings: np.ndarray,
    cache_file: Path,
    dataset_name: str = "embeddings",
    genre_labels: np.ndarray = None,
) -> Dict[str, np.ndarray]:
    """
    Precompute t-SNE, PCA, and UMAP and cache to file.

    If genre_labels provided, also computes supervised UMAP for better genre separation.

    Returns dict with keys: 'tsne', 'pca', 'umap', and optionally 'umap_supervised'
    """
    # Load existing cache if available
    results = {}
    cache_updated = False

    if cache_file.exists():
        print(f"  ✓ Wczytywanie gotowych redukcji z {cache_file.name}")
        with open(cache_file, "rb") as f:
            results = pickle.load(f)

        # Check if we need to add supervised UMAP
        if genre_labels is not None and "umap_supervised" not in results:
            print(f"  Obliczanie nadzorowanej UMAP...")
            print(f"    - UMAP (nadzorowany - z etykietami gatunków)...")
            umap_supervised = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1, metric="cosine")
            results["umap_supervised"] = umap_supervised.fit_transform(embeddings, y=genre_labels)
            print(f"      ✓ UMAP nadzorowany gotowy")
            cache_updated = True
        elif "umap_supervised" in results:
            print(f"  ✓ Nadzorowana UMAP już obliczona")

        # Update cache if needed
        if cache_updated:
            print(f"  💾 Aktualizacja cache: {cache_file}")
            with open(cache_file, "wb") as f:
                pickle.dump(results, f)

        return results

    # Compute all reductions from scratch
    print(f"  Obliczanie redukcji wymiarów dla {len(embeddings):,} osadzeń...")

    # t-SNE
    print(f"    - t-SNE (może potrwać kilka minut)...")
    tsne = TSNE(n_components=2, perplexity=50, random_state=42, max_iter=1000)
    results["tsne"] = tsne.fit_transform(embeddings)
    print(f"      ✓ t-SNE gotowe")

    # PCA
    print(f"    - PCA...")
    pca = PCA(n_components=2, random_state=42)
    results["pca"] = pca.fit_transform(embeddings)
    print(f"      ✓ PCA gotowe")

    # UMAP (unsupervised)
    print(f"    - UMAP (nienadzorowany)...")
    umap = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    results["umap"] = umap.fit_transform(embeddings)
    print(f"      ✓ UMAP gotowe")

    # Supervised UMAP (if genre labels provided)
    if genre_labels is not None:
        print(f"    - UMAP (nadzorowany - z etykietami gatunków)...")
        umap_supervised = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1, metric="cosine")
        results["umap_supervised"] = umap_supervised.fit_transform(embeddings, y=genre_labels)
        print(f"      ✓ UMAP nadzorowany gotowy")

    # Save to cache
    print(f"  💾 Zapisywanie do cache: {cache_file}")
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "wb") as f:
        pickle.dump(results, f)

    return results


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================


def load_rated_movies_data(
    embeddings_file: Path,
    metadata_file: Path,
    movies_df: pd.DataFrame,
    rating_counts: Dict[int, int],
    avg_ratings: Dict[int, float],
) -> Tuple[np.ndarray, List[int], Dict]:
    """
    Load data for 80K rated movies.

    Returns: embeddings, movie_ids, metadata_dict
    """
    # Load embeddings
    embeddings_dict = np.load(embeddings_file, allow_pickle=True).item()
    with open(metadata_file) as f:
        metadata = json.load(f)

    movie_ids = metadata["movie_ids"]
    valid_ids = [mid for mid in movie_ids if mid in embeddings_dict]
    embeddings = np.array([embeddings_dict[mid] for mid in valid_ids])

    # Create metadata
    title_col = "title_ml" if "title_ml" in movies_df.columns else "title"
    id_to_title = dict(zip(movies_df["movieId"], movies_df[title_col]))
    id_to_genres = dict(zip(movies_df["movieId"], movies_df["genres"]))

    # Popularity from TMDB if available
    id_to_tmdb_pop = {}
    if "popularity" in movies_df.columns:
        id_to_tmdb_pop = dict(zip(movies_df["movieId"], movies_df["popularity"].fillna(0)))

    metadata_dict = {
        "movie_ids": valid_ids,
        "id_to_title": id_to_title,
        "id_to_genres": id_to_genres,
        "id_to_rating_count": {mid: rating_counts.get(mid, 0) for mid in valid_ids},
        "id_to_avg_rating": {mid: avg_ratings.get(mid, 0) for mid in valid_ids},
        "id_to_tmdb_pop": id_to_tmdb_pop,
    }

    return embeddings, valid_ids, metadata_dict


def load_tmdb_data(
    tmdb_file: Path,
    movies_df: pd.DataFrame,
) -> Tuple[np.ndarray, List[int], Dict]:
    """
    Load data for all TMDB movies using TMDB CSV for metadata.

    Returns: embeddings, movie_ids, metadata_dict
    """
    # Load TMDB embeddings
    tmdb_data = np.load(tmdb_file, allow_pickle=True)
    embeddings = tmdb_data["embeddings"]
    movie_ids = list(tmdb_data["movie_ids"])

    # Load TMDB CSV for better metadata coverage
    tmdb_csv_path = tmdb_file.parent.parent / "data" / "raw" / "tmdb" / "TMDB_movie_dataset_v11.csv"
    print(f"  Ładowanie metadanych TMDB z {tmdb_csv_path.name}...")
    tmdb_csv = pd.read_csv(tmdb_csv_path)

    # Create lookup dictionaries from TMDB CSV
    id_to_title = dict(zip(tmdb_csv["id"], tmdb_csv["title"]))

    # Convert TMDB genre format (comma-separated) to pipe-separated for consistency
    def convert_genres(genre_str):
        if pd.isna(genre_str) or genre_str == "":
            return "(no genres listed)"
        return genre_str.replace(", ", "|")

    id_to_genres = {row["id"]: convert_genres(row["genres"]) for _, row in tmdb_csv.iterrows()}
    id_to_tmdb_pop = dict(zip(tmdb_csv["id"], tmdb_csv["popularity"].fillna(0)))

    # Extract primary genre for each movie (for supervised learning)
    primary_genres = []
    for movie_id in movie_ids:
        genres_str = id_to_genres.get(movie_id, "(no genres listed)")
        if genres_str == "(no genres listed)":
            primary_genres.append("Other")
        else:
            # Take first genre as primary
            first_genre = genres_str.split("|")[0]
            primary_genres.append(first_genre)

    primary_genres = np.array(primary_genres)

    metadata_dict = {
        "movie_ids": movie_ids,
        "id_to_title": id_to_title,
        "id_to_genres": id_to_genres,
        "id_to_tmdb_pop": id_to_tmdb_pop,
        "primary_genre": primary_genres,
    }

    return embeddings, movie_ids, metadata_dict


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================


def plot_method_comparison(
    reductions: Dict[str, np.ndarray],
    output_path: Path,
    dataset_label: str,
    metadata: Dict = None,
):
    """
    Plot vertical comparison of t-SNE, PCA, and UMAP.
    Colors points by average rating if available, otherwise uses sequential coloring.
    Each subplot has its own colorbar on the left.
    """
    set_thesis_style()

    fig, axes = plt.subplots(3, 1, figsize=(10, 24))
    methods = ["tsne", "pca", "umap"]

    n_points = len(reductions["tsne"])

    # Get average ratings if available
    color_data = None
    cmap_name = COLORMAP
    vmin, vmax = None, None
    cbar_label = None

    if metadata and "id_to_avg_rating" in metadata:
        movie_ids = metadata["movie_ids"]
        avg_ratings = [metadata["id_to_avg_rating"].get(mid, 0) for mid in movie_ids]
        # Filter out zeros (movies without ratings)
        color_data = np.array(avg_ratings)
        # Replace zeros with NaN for better visualization
        color_data = np.where(color_data > 0, color_data, np.nan)
        cmap_name = "RdYlGn"  # Red (low) to Yellow to Green (high) - intuitive for ratings
        vmin, vmax = 0.5, 5.0
        cbar_label = "Średnia ocena"
    else:
        # Fallback to sequential coloring
        color_data = np.arange(n_points)

    for ax, method in zip(axes, methods):
        coords = reductions[method]
        labels = METHOD_LABELS[method]

        # Adjust alpha for more vibrant colors
        alpha = 0.3 if n_points > 100000 else 0.5 if n_points > 50000 else 0.8

        # Plot
        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=color_data,
            cmap=cmap_name,
            alpha=alpha,
            s=5,
            rasterized=True,
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_xlabel(labels["dim1"], fontweight="bold", fontsize=11)
        ax.set_ylabel(labels["dim2"], fontweight="bold", fontsize=11)
        ax.set_title(labels["name"], fontweight="bold", fontsize=13)
        ax.grid(alpha=0.2)

        # Add individual colorbar on the left with bigger margin
        if cbar_label:
            cbar = plt.colorbar(scatter, ax=ax, orientation="vertical", pad=0.1, aspect=20)
            cbar.set_label(cbar_label, fontweight="bold", fontsize=10)
            cbar.ax.yaxis.set_ticks_position("left")
            cbar.ax.yaxis.set_label_position("left")

    fig.suptitle(
        f"Porównanie metod redukcji wymiarów: {dataset_label}",
        fontweight="bold",
        fontsize=15,
        y=0.995,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Zapisano porównanie metod do {output_path.name}")


def plot_with_genres(
    reductions: Dict[str, np.ndarray],
    metadata: Dict,
    output_dir: Path,
    dataset_label: str,
):
    """
    Plot each dimensionality reduction method colored by genre.
    Plots "Other" category first in light gray background, then main genres on top.
    """
    set_thesis_style()

    # Get primary genre for each movie
    genres = []
    for mid in metadata["movie_ids"]:
        genre_str = metadata["id_to_genres"].get(mid, "")
        genres.append(get_primary_genre(genre_str))

    # Get unique genres and assign colors
    genres_array = np.array(genres)
    unique_genres = sorted(set(genres))
    genre_to_color = {g: GENRE_COLORS.get(g, GENRE_COLORS["Other"]) for g in unique_genres}

    methods = ["tsne", "pca", "umap"]

    for method in methods:
        fig, ax = plt.subplots(figsize=(14, 11))
        coords = reductions[method]
        labels = METHOD_LABELS[method]

        # Plot "Other" first as background with low alpha
        if "Other" in unique_genres:
            mask = genres_array == "Other"
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=GENRE_COLORS["Other"],
                label=f"Other ({mask.sum():,})",
                alpha=0.08,
                s=3,
                rasterized=True,
            )

        # Plot each main genre separately for legend (higher visibility)
        main_genres = [g for g in unique_genres if g != "Other"]
        for genre in main_genres:
            mask = genres_array == genre
            if mask.sum() == 0:
                continue

            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=genre_to_color[genre],
                label=f"{genre} ({mask.sum():,})",
                alpha=0.4,
                s=15,
                edgecolors="none",
                rasterized=True,
            )

        ax.set_xlabel(labels["dim1"], fontweight="bold", fontsize=11)
        ax.set_ylabel(labels["dim2"], fontweight="bold", fontsize=11)
        ax.set_title(
            f"{labels['name']}: kolorowanie według gatunku\n{dataset_label}",
            fontweight="bold",
            fontsize=13,
            pad=15,
        )
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1),
            frameon=True,
            shadow=True,
            fontsize=9,
            title="Gatunek",
        )
        ax.grid(alpha=0.2)

        plt.tight_layout()
        output_path = output_dir / f"{method}_genres.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"  ✓ Zapisano {method.upper()} z gatunkami do {output_path.name}")


def plot_with_top_genres(
    reductions: Dict[str, np.ndarray],
    metadata: Dict,
    output_dir: Path,
    dataset_label: str,
    top_n: int = 9,
):
    """
    Plot 3x3 grid for each reduction method showing top N genres.
    Each subplot highlights one genre in color while others are gray.
    """
    set_thesis_style()

    # Get primary genre for each movie
    genres = []
    for mid in metadata["movie_ids"]:
        genre_str = metadata["id_to_genres"].get(mid, "")
        genres.append(get_primary_genre(genre_str))

    # Count genres and get top N (excluding "Other")
    from collections import Counter

    genre_counts = Counter(genres)

    # Get top N genres (excluding "Other")
    top_genres = [g for g, count in genre_counts.most_common() if g != "Other"][:top_n]

    genres_array = np.array(genres)

    methods = ["tsne", "pca", "umap"]

    for method in methods:
        # Create 3x3 grid
        fig, axes = plt.subplots(3, 3, figsize=(20, 20))
        axes = axes.flatten()

        coords = reductions[method]
        labels = METHOD_LABELS[method]

        # Plot each genre in a separate subplot
        for idx, (ax, genre) in enumerate(zip(axes, top_genres)):
            # Plot all movies as gray background
            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c="#BDBDBD",
                alpha=0.1,
                s=5,
                rasterized=True,
            )

            # Highlight the current genre
            mask = genres_array == genre
            genre_count = mask.sum()

            if genre_count > 0:
                color = GENRE_COLORS.get(genre, GENRE_COLORS["Other"])
                ax.scatter(
                    coords[mask, 0],
                    coords[mask, 1],
                    c=color,
                    alpha=0.6,
                    s=20,
                    edgecolors="none",
                    rasterized=True,
                )

            ax.set_title(f"{genre} ({genre_count:,})", fontweight="bold", fontsize=11)
            ax.set_xlabel(labels["dim1"], fontsize=9)
            ax.set_ylabel(labels["dim2"], fontsize=9)
            ax.grid(alpha=0.2)

        fig.suptitle(
            f"{labels['name']}: top-{top_n} najpopularniejszych gatunków\n{dataset_label}",
            fontweight="bold",
            fontsize=16,
            y=0.995,
        )

        plt.tight_layout()
        output_path = output_dir / f"{method}_top{top_n}_genres_grid.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"  ✓ Zapisano {method.upper()} z siatką top-{top_n} gatunków do {output_path.name}")


def plot_density_maps(
    reductions: Dict[str, np.ndarray],
    output_dir: Path,
    dataset_label: str,
):
    """
    Plot density heatmaps for each dimensionality reduction method.
    """
    set_thesis_style()

    methods = ["tsne", "pca", "umap"]

    for method in methods:
        fig, ax = plt.subplots(figsize=(12, 10))
        coords = reductions[method]
        labels = METHOD_LABELS[method]

        # Create hexbin density plot with rainbow colors
        hb = ax.hexbin(
            coords[:, 0],
            coords[:, 1],
            gridsize=80,
            cmap="YlOrRd",
            mincnt=1,
        )

        # Colorbar
        cb = plt.colorbar(hb, ax=ax, label="Liczba filmów")
        cb.ax.tick_params(labelsize=10)

        ax.set_xlabel(labels["dim1"], fontweight="bold", fontsize=11)
        ax.set_ylabel(labels["dim2"], fontweight="bold", fontsize=11)
        ax.set_title(
            f"Mapa gęstości {labels['name']}: {dataset_label}",
            fontweight="bold",
            fontsize=13,
            pad=15,
        )

        plt.tight_layout()
        output_path = output_dir / f"{method}_density.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"  ✓ Zapisano mapę gęstości {method.upper()} do {output_path.name}")


def plot_supervised_umap(
    reductions: Dict[str, np.ndarray],
    metadata: Dict,
    output_dir: Path,
    dataset_label: str,
):
    """
    Generate plots specifically for supervised UMAP showing improved genre separation.

    Creates two plots:
    1. All genres colored (comparison with unsupervised)
    2. Top 9 genres in 3x3 grid
    """
    if "umap_supervised" not in reductions:
        return

    coords = reductions["umap_supervised"]
    genres_array = metadata["primary_genre"]

    # Count genre frequencies
    unique_genres, counts = np.unique(genres_array, return_counts=True)
    genre_counts = dict(zip(unique_genres, counts))
    top_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:9]
    top_genre_names = [g for g, _ in top_genres]

    # Plot 1: All genres colored
    fig, ax = plt.subplots(figsize=(14, 10))

    # Plot "Other" genres in background
    other_mask = ~np.isin(genres_array, top_genre_names)
    if other_mask.sum() > 0:
        ax.scatter(
            coords[other_mask, 0],
            coords[other_mask, 1],
            c="lightgray",
            alpha=0.08,
            s=10,
            edgecolors="none",
            rasterized=True,
            label=f"Other ({other_mask.sum():,})",
        )

    # Plot main genres on top
    genre_to_color = {g: GENRE_COLORS.get(g, "#808080") for g in top_genre_names}

    for genre in top_genre_names:
        mask = genres_array == genre
        if mask.sum() == 0:
            continue

        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=genre_to_color[genre],
            label=f"{genre} ({mask.sum():,})",
            alpha=0.4,
            s=15,
            edgecolors="none",
            rasterized=True,
        )

    ax.set_xlabel("UMAP Dimension 1", fontweight="bold", fontsize=11)
    ax.set_ylabel("UMAP Dimension 2", fontweight="bold", fontsize=11)
    ax.set_title(
        f"Supervised UMAP (z etykietami gatunków)\n{dataset_label}",
        fontweight="bold",
        fontsize=13,
        pad=15,
    )
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        frameon=True,
        shadow=True,
        fontsize=9,
        title="Gatunek",
    )

    plt.tight_layout()
    output_path = output_dir / "umap_supervised_genres.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Zapisano nadzorowaną UMAP (wszystkie gatunki) do {output_path.name}")

    # Plot 2: Top 9 genres in 3x3 grid
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    axes = axes.flatten()

    for idx, (genre, count) in enumerate(top_genres):
        ax = axes[idx]

        # Background: all other movies
        other_mask = genres_array != genre
        ax.scatter(
            coords[other_mask, 0],
            coords[other_mask, 1],
            c="lightgray",
            alpha=0.05,
            s=8,
            edgecolors="none",
            rasterized=True,
        )

        # Foreground: this genre
        genre_mask = genres_array == genre
        ax.scatter(
            coords[genre_mask, 0],
            coords[genre_mask, 1],
            c=GENRE_COLORS.get(genre, "#808080"),
            alpha=0.5,
            s=20,
            edgecolors="none",
            rasterized=True,
        )

        ax.set_xlabel("UMAP Dim 1", fontsize=9)
        ax.set_ylabel("UMAP Dim 2", fontsize=9)
        ax.set_title(
            f"{genre}\n({count:,} films)",
            fontweight="bold",
            fontsize=11,
            pad=8,
        )
        ax.tick_params(labelsize=8)

    fig.suptitle(
        f"Supervised UMAP: Top 9 Genres (z etykietami)\n{dataset_label}",
        fontweight="bold",
        fontsize=16,
        y=0.995,
    )

    plt.tight_layout()
    output_path = output_dir / "umap_supervised_top9.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Zapisano nadzorowaną UMAP (top-9 grid) do {output_path.name}")


# =============================================================================
# MAIN ORCHESTRATION FUNCTIONS
# =============================================================================


def generate_plots_for_dataset(
    embeddings: np.ndarray,
    metadata: Dict,
    output_dir: Path,
    cache_file: Path,
    dataset_name: str,
    dataset_label: str,
):
    """
    Generate all plots for a single dataset.

    Creates:
    - Method comparison plot (t-SNE vs PCA vs UMAP)
    - Genre-colored plots for each method
    - Top 9 genres plots
    - Density maps for each method
    - If TMDB dataset, also generates supervised UMAP visualizations
    """
    print(f"\n{'='*80}")
    print(f"Generowanie wykresów dla: {dataset_label}")
    print(f"{'='*80}")

    # Prepare genre labels for supervised learning (TMDB only)
    genre_labels = None
    if dataset_name == "tmdb" and "primary_genre" in metadata:
        # Encode genre strings to integers for UMAP
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        genre_labels = le.fit_transform(metadata["primary_genre"])
        print(f"  ✓ Przygotowano etykiety gatunków dla nadzorowanej UMAP ({len(np.unique(genre_labels))} unikalnych gatunków)")

    # Precompute dimensionality reductions
    reductions = precompute_dimensionality_reductions(embeddings, cache_file, dataset_name, genre_labels=genre_labels)

    # 1. Comparison plot
    print(f"\n  [1/4] Porównanie metod...")
    plot_method_comparison(
        reductions,
        output_dir / f"{dataset_name}_comparison.png",
        dataset_label,
        metadata,
    )

    # 2. Genre plots
    print(f"\n  [2/4] Wykresy z gatunkami...")
    plot_with_genres(reductions, metadata, output_dir, dataset_label)

    # 3. Top 9 genres plots
    print(f"\n  [3/4] Wykresy z top-9 gatunkami...")
    plot_with_top_genres(reductions, metadata, output_dir, dataset_label, top_n=9)

    # 4. Density maps
    print(f"\n  [4/4] Mapy gęstości...")
    plot_density_maps(reductions, output_dir, dataset_label)

    # 5. Supervised UMAP plots (if available)
    if "umap_supervised" in reductions:
        print(f"\n  [5/5] Nadzorowana UMAP (z etykietami gatunków)...")
        plot_supervised_umap(reductions, metadata, output_dir, dataset_label)

    print(f"\n{'='*80}")
    print(f"✓ Zakończono generowanie wykresów dla: {dataset_label}")
    print(f"{'='*80}")


def plot_cosine_similarity_heatmap(
    embeddings: np.ndarray,
    metadata: Dict,
    output_path: Path,
    n_movies: int = 15,
    dataset_name: str = "embeddings",
):
    """
    Plot lower triangular cosine similarity heatmap for diverse popular movies.

    Args:
        embeddings: Movie embeddings array
        metadata: Metadata dict with movie_ids, id_to_title, etc.
        output_path: Path to save plot
        n_movies: Number of movies to include
        dataset_name: Name of dataset for title
    """
    set_thesis_style()

    print(f"  Tworzenie heatmapy podobieństwa kosinusowego...")

    movie_ids = metadata["movie_ids"]
    id_to_title = metadata["id_to_title"]

    # Get popularity scores if available
    if "id_to_tmdb_pop" in metadata and metadata["id_to_tmdb_pop"]:
        popularity = [metadata["id_to_tmdb_pop"].get(mid, 0) for mid in movie_ids]
    elif "id_to_rating_count" in metadata:
        popularity = [metadata["id_to_rating_count"].get(mid, 0) for mid in movie_ids]
    else:
        # Use simple index as popularity
        popularity = list(range(len(movie_ids)))

    # Filter for popular movies with valid embeddings
    valid_movies = [
        (i, movie_ids[i], id_to_title.get(movie_ids[i], f"Film {movie_ids[i]}"), popularity[i])
        for i in range(len(movie_ids))
        if i < len(embeddings)
    ]

    # Sort by popularity and take top candidates
    valid_movies.sort(key=lambda x: x[3], reverse=True)
    top_candidates = valid_movies[: min(100, len(valid_movies))]

    # For TMDB dataset, just use most popular movies
    # For rated movies, use diversity selection
    if dataset_name == "tmdb":
        # Simply take the top n_movies most popular
        selected_movies = top_candidates[:n_movies]
        final_indices = [x[0] for x in selected_movies]
        final_embeddings = embeddings[final_indices]
        final_titles = [x[2][:30] for x in selected_movies]
    else:
        # Calculate pairwise similarity for candidates to find diverse set
        candidate_indices = [x[0] for x in top_candidates]
        candidate_embeddings = embeddings[candidate_indices]

        # Normalize embeddings
        norms = np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
        normalized = candidate_embeddings / (norms + 1e-8)

        # Compute full similarity matrix for candidates
        similarity_matrix = np.dot(normalized, normalized.T)

        # Select diverse movies using greedy selection
        selected_indices = [0]  # Start with most popular

        for _ in range(n_movies - 1):
            # Calculate minimum similarity to already selected movies
            min_sims = []
            for i in range(len(candidate_indices)):
                if i not in selected_indices:
                    sims_to_selected = [similarity_matrix[i, j] for j in selected_indices]
                    min_sim = min(sims_to_selected)
                    min_sims.append((i, min_sim))

            # Select the movie with lowest maximum similarity (most diverse)
            if min_sims:
                next_idx = min(min_sims, key=lambda x: x[1])[0]
                selected_indices.append(next_idx)

        # Get final selected movies
        selected_movies = [top_candidates[i] for i in selected_indices]
        final_indices = [x[0] for x in selected_movies]
        final_embeddings = embeddings[final_indices]
        final_titles = [x[2][:30] for x in selected_movies]

    # Compute final similarity matrix
    norms = np.linalg.norm(final_embeddings, axis=1, keepdims=True)
    normalized = final_embeddings / (norms + 1e-8)
    similarity_matrix = np.dot(normalized, normalized.T)

    # Create lower triangular mask (k=1 to show diagonal)
    mask = np.triu(np.ones_like(similarity_matrix, dtype=bool), k=1)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))

    # Use masked array for lower triangle + diagonal
    masked_sim = np.ma.array(similarity_matrix, mask=mask)

    sns.heatmap(
        masked_sim,
        xticklabels=final_titles,
        yticklabels=final_titles,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        square=True,
        ax=ax,
        vmin=0,
        vmax=1,
        annot_kws={"size": 8},
        cbar_kws={"label": "Podobieństwo kosinusowe"},
        mask=mask,
        linewidths=0,
        linecolor="none",
    )

    # Remove grid
    ax.grid(False)

    ax.set_title(
        f"Macierz podobieństwa kosinusowego - {DATASET_LABELS.get(dataset_name, dataset_name)}",
        fontweight="bold",
        fontsize=12,
        pad=15,
    )
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Zapisano heatmapę podobieństwa do {output_path}")


def generate_all_embedding_plots(
    data_dir: Path,
    saved_models_dir: Path,
    output_dir: Path,
):
    """
    Generate all embedding visualization plots.

    Creates two sets of plots:
    1. For 80K movies with ratings (Two-Tower embeddings)
    2. For all 650K TMDB movies

    Each set includes:
    - Method comparison (t-SNE, PCA, UMAP)
    - Genre-colored visualizations
    - Density maps

    All dimensionality reductions are precomputed and cached.
    """
    print("\n" + "=" * 80)
    print("GENEROWANIE WYKRESÓW WIZUALIZACJI OSADZEŃ")
    print("=" * 80)

    # Create output directories
    emb_dir = output_dir / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)

    rated_dir = emb_dir / "rated_movies"
    rated_dir.mkdir(exist_ok=True)

    tmdb_dir = emb_dir / "tmdb_full"
    tmdb_dir.mkdir(exist_ok=True)

    cache_dir = saved_models_dir / "reduction_cache"
    cache_dir.mkdir(exist_ok=True)

    # Load movie metadata
    movies_file = data_dir / "movies.parquet"
    if not movies_file.exists():
        print(f"⚠ Nie znaleziono pliku filmów: {movies_file}")
        return

    movies_df = pl.read_parquet(movies_file).to_pandas()
    print(f"\n✓ Załadowano {len(movies_df):,} filmów z metadanymi")

    # Load rating statistics
    train_file = data_dir / "train.parquet"
    rating_counts = {}
    avg_ratings = {}
    if train_file.exists():
        train_df = pl.read_parquet(train_file)
        counts = train_df.group_by("movieId").len()
        rating_counts = dict(counts.iter_rows())
        avgs = train_df.group_by("movieId").agg(pl.col("rating").mean())
        avg_ratings = dict(avgs.iter_rows())
        print(f"✓ Załadowano statystyki ocen dla {len(rating_counts):,} filmów")

    # =========================================================================
    # DATASET 1: 80K Movies with Ratings
    # =========================================================================
    embeddings_file = saved_models_dir / "movie_embeddings.npy"
    metadata_file = saved_models_dir / "movie_embeddings_metadata.json"

    if embeddings_file.exists() and metadata_file.exists():
        print(f"\n{'='*80}")
        print(f"DATASET 1: Filmy z ocenami (Two-Tower)")
        print(f"{'='*80}")

        embeddings, movie_ids, metadata = load_rated_movies_data(
            embeddings_file,
            metadata_file,
            movies_df,
            rating_counts,
            avg_ratings,
        )

        print(f"✓ Załadowano {len(embeddings):,} osadzeń ({embeddings.shape[1]}D)")

        generate_plots_for_dataset(
            embeddings,
            metadata,
            rated_dir,
            cache_dir / "rated_movies_reductions.pkl",
            "rated",
            DATASET_LABELS["rated"],
        )

        # Generate cosine similarity heatmap
        plot_cosine_similarity_heatmap(
            embeddings,
            metadata,
            rated_dir / "podobienstwo_kosinusowe.png",
            n_movies=15,
            dataset_name="rated",
        )
    else:
        print(f"\n⚠ Nie znaleziono osadzeń Two-Tower")

    # =========================================================================
    # DATASET 2: All TMDB Movies
    # =========================================================================
    tmdb_file = saved_models_dir / "tmdb_movie_embeddings.npz"

    if tmdb_file.exists():
        print(f"\n{'='*80}")
        print(f"DATASET 2: Wszystkie filmy TMDB")
        print(f"{'='*80}")

        embeddings, movie_ids, metadata = load_tmdb_data(
            tmdb_file,
            movies_df,
        )

        print(f"✓ Załadowano {len(embeddings):,} osadzeń TMDB ({embeddings.shape[1]}D)")

        generate_plots_for_dataset(
            embeddings,
            metadata,
            tmdb_dir,
            cache_dir / "tmdb_full_reductions.pkl",
            "tmdb",
            DATASET_LABELS["tmdb"],
        )

        # Generate cosine similarity heatmap
        plot_cosine_similarity_heatmap(
            embeddings,
            metadata,
            tmdb_dir / "podobienstwo_kosinusowe.png",
            n_movies=15,
            dataset_name="tmdb",
        )
    else:
        print(f"\n⚠ Nie znaleziono osadzeń TMDB: {tmdb_file}")

    print("\n" + "=" * 80)
    print(f"✓✓✓ WSZYSTKIE WYKRESY WYGENEROWANE POMYŚLNIE ✓✓✓")
    print(f"  - Filmy z ocenami: {rated_dir}")
    print(f"  - Wszystkie TMDB: {tmdb_dir}")
    print("=" * 80)


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "model" / "data" / "processed"
    SAVED_MODELS_DIR = PROJECT_ROOT / "model" / "saved_models"
    PLOTS_DIR = PROJECT_ROOT / "model" / "plots"

    generate_all_embedding_plots(DATA_DIR, SAVED_MODELS_DIR, PLOTS_DIR)
